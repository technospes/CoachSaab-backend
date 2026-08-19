from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from typing import Optional, List, Annotated, Literal, Any, Dict
import os
import json
import re
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI(title="CoachSaab API", version="5.0")

DATABASE_URL = os.getenv("DATABASE_URL", "")
engine = create_engine(DATABASE_URL)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ==========================================
# 1. STRICT SCHEMAS
# ==========================================
class ExerciseConfig(BaseModel):
    activity_key: str = Field(..., description="Canonical key, e.g., 'squat', 'push_up'")
    sets: int = Field(..., gt=0, description="Number of sets")
    reps: int = Field(..., gt=0, description="Number of reps per set")
    progression: str = Field(..., description="How to progress over the weeks")

class WorkoutPlanSchema(BaseModel):
    duration_weeks: int = Field(..., gt=0, le=12, description="Duration in weeks (max 12)")
    goal: str = Field(..., description="Main objective of the plan")
    exercises: List[ExerciseConfig] = Field(..., description="List of structured exercises")
    notes: str = Field(..., description="Additional coaching advice")

class UserProfileUpdate(BaseModel):
    age: Optional[int] = Field(None, description="User's age, if mentioned.")
    weight_kg: Optional[float] = Field(None, description="User's weight in kg, if mentioned.")
    goals: Optional[List[str]] = Field(None, description="User's fitness goals, if mentioned.")
    preferred_categories: Optional[List[str]] = Field(None, description="Preferred exercise types, if mentioned.")

class UserIntent(BaseModel):
    intent: Literal["generate_plan", "normal_chat"] = Field(
        ..., description="Does the user want to create/update a workout plan, or just chat normally?"
    )

# ==========================================
# 2. GRAPH STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    user_profile: dict
    missing_fields: List[str]
    extracted_profile: Optional[UserProfileUpdate]
    intent: Optional[str]

# ==========================================
# 3. LLM INITIALIZATION
# ==========================================
llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)

# Extractors (Forcing specific JSON outputs)
profile_extractor = llm.with_structured_output(UserProfileUpdate)
intent_classifier = llm.with_structured_output(UserIntent)
plan_generator = llm.with_structured_output(WorkoutPlanSchema)

# ==========================================
# 4. GRAPH NODES (Deterministic State Machine)
# ==========================================
def load_user_state_node(state: AgentState):
    """Reads DB, populates profile and calculates missing fields."""
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT name, gender, age, weight_kg, goals, preferred_categories FROM users WHERE user_id = :uid"),
            {"uid": state["user_id"]}
        ).mappings().fetchone()
        
    profile = dict(user_row) if user_row else {}
    missing = []
    if profile.get('age') is None: missing.append("Age")
    if profile.get('weight_kg') is None: missing.append("Weight (kg)")
    if not profile.get('goals'): missing.append("Fitness Goals")
    if not profile.get('preferred_categories'): missing.append("Preferred Exercise Types (e.g. bodyweight, legs)")
    
    return {"user_profile": profile, "missing_fields": missing}

def extract_profile_node(state: AgentState):
    """Uses LLM to extract any new profile info from the latest user message."""
    latest_message = state["messages"][-1].content
    prompt = f"Extract any fitness profile info from this message: '{latest_message}'. If a field isn't mentioned, leave it null."
    extracted = profile_extractor.invoke([HumanMessage(content=prompt)])
    return {"extracted_profile": extracted}

def update_database_node(state: AgentState):
    """Updates PostgreSQL if new data was extracted, merging lists safely."""
    extracted = state.get("extracted_profile")
    if not extracted:
        return {}

    user_id = state["user_id"]
    updates = []
    params = {"uid": user_id}
    
    if extracted.age is not None:
        updates.append("age = :age")
        params["age"] = extracted.age
    if extracted.weight_kg is not None:
        updates.append("weight_kg = :weight")
        params["weight"] = extracted.weight_kg

    # Safe Merging for Lists (Read first, then merge)
    if extracted.goals or extracted.preferred_categories:
        with engine.connect() as conn:
            current_data = conn.execute(
                text("SELECT goals, preferred_categories FROM users WHERE user_id = :uid"),
                {"uid": user_id}
            ).mappings().fetchone()
            
            if extracted.goals:
                existing_goals = list(current_data['goals'] or [])
                # Add only new goals
                new_goals = [g for g in extracted.goals if g not in existing_goals]
                merged_goals = existing_goals + new_goals
                updates.append("goals = :goals")
                params["goals"] = merged_goals
                
            if extracted.preferred_categories:
                existing_prefs = list(current_data['preferred_categories'] or [])
                new_prefs = [p for p in extracted.preferred_categories if p not in existing_prefs]
                merged_prefs = existing_prefs + new_prefs
                updates.append("preferred_categories = :prefs")
                params["prefs"] = merged_prefs
        
    if updates:
        with engine.begin() as conn:
            query = text(f"UPDATE users SET {', '.join(updates)}, updated_at = now() WHERE user_id = :uid")
            conn.execute(query, params)
            
    # Re-run load state to get fresh missing_fields directly from Truth (DB)
    return load_user_state_node(state)

def determine_intent_node(state: AgentState):
    """Classifies if the user is asking for a routine/plan or just chatting."""
    latest_message = state["messages"][-1].content
    intent_result = intent_classifier.invoke([HumanMessage(content=latest_message)])
    return {"intent": intent_result.intent}

def ask_question_node(state: AgentState):
    """Generates a polite question for exactly ONE missing field."""
    missing = state["missing_fields"][0]
    prompt = f"You are CoachSaab. The user wants to create a workout plan, but their profile is incomplete. Ask them politely and conversationally for their {missing}. Keep it to 1-2 sentences max. Do not ask for anything else."
    response = llm.invoke([SystemMessage(content=prompt)])
    return {"messages": [response]}

def generate_plan_node(state: AgentState):
    """Generates a structured plan via LLM, saves it to PostgreSQL, and replies."""
    profile = state["user_profile"]
    prompt = f"""Generate a 4-12 week structured workout plan based on this user:
    Name: {profile.get('name')}
    Age: {profile.get('age')}
    Weight: {profile.get('weight_kg')} kg
    Goals: {', '.join(profile.get('goals', []))}
    Preferences: {', '.join(profile.get('preferred_categories', []))}
    
    Return the plan exactly matching the required JSON schema."""
    
    plan_data = plan_generator.invoke([SystemMessage(content=prompt)])
    
    # Save to PostgreSQL (Truth)
    db_json = plan_data.model_dump()
    db_json["status"] = "active"
    
    try:
        with engine.begin() as conn:
            conn.execute(text("UPDATE workout_plans SET is_active = false WHERE user_id = :uid"), {"uid": state["user_id"]})
            query = text("INSERT INTO workout_plans (user_id, plan_name, plan_json, is_active) VALUES (:uid, :name, :data, true)")
            conn.execute(query, {"uid": state["user_id"], "name": f"{profile.get('name')}'s Custom Plan", "data": json.dumps(db_json)})
            
        msg = AIMessage(content=f"I've successfully generated and saved your '{plan_data.goal}' plan! Check your Home tab to see the full breakdown.")
    except Exception as e:
        msg = AIMessage(content="I generated the plan, but hit a database error saving it. Please try again.")

    return {"messages": [msg]}

def normal_chat_node(state: AgentState):
    """Standard RAG/Conversational response with full context."""
    profile = state["user_profile"]
    user_id = state["user_id"]
    
    # Fetch recent workouts for context
    with engine.connect() as conn:
        sessions_query = text("""
            SELECT activity_key, reps, duration_seconds, form_score, dominant_deviation 
            FROM workout_sessions 
            WHERE user_id = :uid ORDER BY created_at DESC LIMIT 3
        """)
        session_rows = conn.execute(sessions_query, {"uid": user_id}).mappings().fetchall()
        
    workout_context = "No recent workouts recorded."
    if session_rows:
        summaries = [
            f"- {s['activity_key'].capitalize()}: {s['reps']} reps, Score: {s['form_score']}%, Issue: {s['dominant_deviation'] or 'None'}"
            for s in session_rows
        ]
        workout_context = "\n".join(summaries)

    system_prompt = f"""You are CoachSaab, a smart AI fitness coach.
    
    Current User Profile:
    - Name: {profile.get('name')}
    - Gender: {profile.get('gender')}
    - Age: {profile.get('age') or 'Not set'}
    - Weight: {f"{profile.get('weight_kg')} kg" if profile.get('weight_kg') else 'Not set'}
    - Goals: {', '.join(profile.get('goals', [])) if profile.get('goals') else 'Not set'}
    - Preferences: {', '.join(profile.get('preferred_categories', [])) if profile.get('preferred_categories') else 'Not set'}

    Recent Workouts:
    {workout_context}

    Keep responses brief, helpful, and action-oriented (1-3 sentences)."""
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"][-6:]
    response = llm.invoke(messages)
    return {"messages": [response]}

# ==========================================
# 5. EDGE ROUTING LOGIC
# ==========================================
def route_intent(state: AgentState) -> str:
    """Routes to plan generation or normal chat based on intent."""
    if state["intent"] == "generate_plan":
        # If they want a plan, check if the profile is complete first
        if len(state["missing_fields"]) > 0:
            return "ask_question"
        return "generate_plan"
    return "normal_chat"

# ==========================================
# 6. GRAPH COMPILATION
# ==========================================
graph_builder = StateGraph(AgentState)

# Add Nodes
graph_builder.add_node("load_user_state", load_user_state_node)
graph_builder.add_node("extract_profile", extract_profile_node)
graph_builder.add_node("update_database", update_database_node)
graph_builder.add_node("determine_intent", determine_intent_node)
graph_builder.add_node("ask_question", ask_question_node)
graph_builder.add_node("generate_plan", generate_plan_node)
graph_builder.add_node("normal_chat", normal_chat_node)

# Add Edges (The State Machine Flow)
graph_builder.add_edge(START, "load_user_state")
graph_builder.add_edge("load_user_state", "extract_profile")
graph_builder.add_edge("extract_profile", "update_database")
graph_builder.add_edge("update_database", "determine_intent")

# Branch: Intent -> Plan (Complete vs Incomplete Profile) OR Chat
graph_builder.add_conditional_edges("determine_intent", route_intent)
graph_builder.add_edge("ask_question", END)
graph_builder.add_edge("generate_plan", END)
graph_builder.add_edge("normal_chat", END)

agent_graph = graph_builder.compile()

# ==========================================
# 7. FASTAPI ENDPOINTS
# ==========================================
def clean_ai_response(text_content: str) -> str:
    if not text_content: return "Hello!"
    cleaned = re.sub(r'<think>.*?</think>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

class ChatMessageCreate(BaseModel):
    role: str
    content: str

class WorkoutSessionCreate(BaseModel):
    user_id: str
    activity_key: str
    reps: int
    duration_seconds: int
    form_score: int
    dominant_deviation: Optional[str] = None
    deviations_json: Optional[dict] = {}

class UserCreate(BaseModel):
    name: str
    gender: Optional[str] = None
    goals: Optional[List[str]] = []

@app.get("/")
def read_root(): return {"message": "CoachSaab API (Deterministic Graph) Active 🚀"}

@app.post("/api/v1/users")
def create_user(user: UserCreate):
    with engine.begin() as conn:
        query = text("""
            INSERT INTO users (name, gender, goals) 
            VALUES (:name, :gender, :goals) 
            RETURNING user_id, name, gender, goals
        """)
        result = conn.execute(query, {
            "name": user.name, "gender": user.gender, "goals": user.goals
        }).mappings().fetchone()
        return dict(result)

@app.post("/api/v1/chat/conversations")
def create_conversation(user_id: str):
    with engine.begin() as conn:
        result = conn.execute(text("INSERT INTO chatbot_conversations (user_id) VALUES (:user_id) RETURNING conversation_id"), {"user_id": user_id}).mappings().fetchone()
        return dict(result)

@app.get("/api/v1/chat/conversations/{conversation_id}/messages")
def get_chat_history(conversation_id: str):
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT role, content, created_at FROM chatbot_messages WHERE conversation_id = :conv_id ORDER BY created_at ASC"), {"conv_id": conversation_id}).mappings().fetchall()
        return [{"role": r["role"], "content": clean_ai_response(r["content"]) if r["role"] == "assistant" else r["content"]} for r in rows]

@app.post("/api/v1/sessions")
def save_workout_session(session: WorkoutSessionCreate):
    with engine.begin() as conn:
        query = text("""
            INSERT INTO workout_sessions 
            (user_id, activity_key, reps, duration_seconds, form_score, dominant_deviation, deviations_json) 
            VALUES (:user_id, :activity_key, :reps, :duration_seconds, :form_score, :dominant_deviation, :deviations_json) 
            RETURNING session_id
        """)
        result = conn.execute(query, {
            "user_id": session.user_id, "activity_key": session.activity_key,
            "reps": session.reps, "duration_seconds": session.duration_seconds,
            "form_score": session.form_score, "dominant_deviation": session.dominant_deviation,
            "deviations_json": json.dumps(session.deviations_json)
        }).mappings().fetchone()
        return {"status": "success", "session_id": str(result['session_id'])}

@app.get("/api/v1/users/{user_id}/plan")
def get_active_plan(user_id: str):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT plan_name, plan_json, created_at FROM workout_plans WHERE user_id = :uid AND is_active = true ORDER BY created_at DESC LIMIT 1"), {"uid": user_id}).mappings().fetchone()
        return dict(row) if row else {"status": "no_active_plan"}

@app.post("/api/v1/chat/conversations/{conversation_id}/messages")
def add_chat_message(conversation_id: str, message: ChatMessageCreate):
    try:
        with engine.begin() as conn:
            conv_row = conn.execute(text("SELECT user_id FROM chatbot_conversations WHERE conversation_id = :conv_id"), {"conv_id": conversation_id}).mappings().fetchone()
            if not conv_row: raise HTTPException(status_code=404, detail="Conversation not found")
            user_id = conv_row['user_id']
            
            # Save User Message
            conn.execute(text("INSERT INTO chatbot_messages (conversation_id, role, content) VALUES (:conv_id, 'user', :content)"), {"conv_id": conversation_id, "content": message.content})
            
            # Load History for Graph
            history_rows = conn.execute(text("SELECT role, content FROM chatbot_messages WHERE conversation_id = :conv_id ORDER BY created_at ASC LIMIT 10"), {"conv_id": conversation_id}).mappings().fetchall()
            
        lg_messages = []
        for row in history_rows:
            if row['role'] == 'user': lg_messages.append(HumanMessage(content=row['content']))
            elif row['role'] == 'assistant': lg_messages.append(AIMessage(content=row['content']))

        # RUN THE STATE MACHINE
        initial_state = {"messages": lg_messages, "user_id": user_id, "missing_fields": []}
        result = agent_graph.invoke(initial_state)
        
        raw_reply = result["messages"][-1].content
        ai_reply = clean_ai_response(raw_reply)
        
    except Exception as e:
        print(f"Graph Error: {str(e)}")
        ai_reply = "I ran into a temporary issue processing that. Let's try again!"

    # Save Assistant Message
    with engine.begin() as conn:
        result = conn.execute(text("INSERT INTO chatbot_messages (conversation_id, role, content) VALUES (:conv_id, 'assistant', :content) RETURNING message_id, role, content"), {"conv_id": conversation_id, "content": ai_reply}).mappings().fetchone()
        
    return {"role": result["role"], "content": clean_ai_response(result["content"])}