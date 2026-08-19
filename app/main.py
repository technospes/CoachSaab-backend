from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from typing import Optional, List, Annotated, Literal
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

app = FastAPI(title="CoachSaab API", version="5.1")

DATABASE_URL = os.getenv("DATABASE_URL", "")
engine = create_engine(DATABASE_URL)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class DailyWorkout(BaseModel):
    day_number: int = Field(..., description="Day of the week (1 to 7)")
    focus: str = Field(..., description="Workout focus (e.g., 'UPPER BODY', 'LOWER BODY', 'ACTIVE RECOVERY', 'REST DAY')")
    exercises: List[str] = Field(..., description="List of exercise names (e.g., ['Push Up', 'Plank']). Empty if rest day.")
    is_rest: bool = Field(..., description="True if this is a rest day")

class WorkoutPlanSchema(BaseModel):
    duration_weeks: int = Field(..., gt=0, le=12, description="Duration in weeks (max 12)")
    goal: str = Field(..., description="Main objective of the plan")
    notes: str = Field(..., description="Additional coaching advice and progression instructions")
    schedule: List[DailyWorkout] = Field(..., description="A 7-day schedule template (Days 1-7) that repeats each week")

# ==========================================
# AGENT TOOLS
class ProfileExtractionSchema(BaseModel):
    age: Optional[int] = Field(None, description="Extracted age")
    weight_kg: Optional[float] = Field(None, description="Extracted weight in kg")
    goals: Optional[List[str]] = Field(None, description="Extracted fitness goals")
    preferred_categories: Optional[List[str]] = Field(None, description="Extracted preferred exercise types")

class IntentSchema(BaseModel):
    intent: Literal["chat", "plan"] = Field(description="Is the user asking for a workout plan/routine ('plan') or just having a conversation ('chat')?")

# ==========================================
# LANGGRAPH STATE & NODES (Deterministic Flow)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    user_profile: dict
    missing_fields: list
    extracted_updates: dict
    intent: str

def load_state_node(state: AgentState):
    """Loads the user's current profile from the database and calculates missing fields."""
    user_id = state["user_id"]
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT name, age, weight_kg, goals, preferred_categories FROM users WHERE user_id = :uid"), 
            {"uid": user_id}
        ).mappings().fetchone()
    
    if not user_row:
        return {"user_profile": {}, "missing_fields": ["Profile Error"]}

    profile = dict(user_row)
    
    # FIX: Convert Postgres Decimal to Python Float so json.dumps() doesn't crash!
    if profile.get("weight_kg") is not None:
        profile["weight_kg"] = float(profile["weight_kg"])

    missing = []
    if profile.get("age") is None: missing.append("Age")
    if profile.get("weight_kg") is None: missing.append("Weight (kg)")
    if not profile.get("goals"): missing.append("Fitness Goals")
    if not profile.get("preferred_categories"): missing.append("Preferred Exercises")

    return {"user_profile": profile, "missing_fields": missing}

def extract_profile_node(state: AgentState):
    """Passively extracts any profile data mentioned in the latest user message."""
    try:
        latest_msg = state["messages"][-1].content
        extractor = llm.with_structured_output(ProfileExtractionSchema)
        extracted = extractor.invoke([
            SystemMessage(content="Extract any age, weight, goals, or exercise preferences mentioned. Return null for fields not mentioned."),
            HumanMessage(content=latest_msg)
        ])
        updates = extracted.model_dump(exclude_none=True) if extracted else {}
        return {"extracted_updates": updates}
    except Exception as e:
        print(f"Extraction Error: {str(e)}")
        return {"extracted_updates": {}}

def update_database_node(state: AgentState):
    """Merges new profile extractions into the database (arrays are appended)."""
    updates = state.get("extracted_updates", {})
    if not updates:
        return {} # Nothing to update
    
    user_id = state["user_id"]
    profile = state["user_profile"]
    
    set_clauses = []
    params = {"uid": user_id}

    if "age" in updates:
        set_clauses.append("age = :age")
        params["age"] = updates["age"]
    if "weight_kg" in updates:
        set_clauses.append("weight_kg = :weight")
        params["weight"] = updates["weight_kg"]
        
    # MERGE ARRAYS INSTEAD OF OVERWRITING
    if "goals" in updates:
        existing = profile.get("goals") or []
        merged = list(set(existing + updates["goals"]))
        set_clauses.append("goals = CAST(:goals AS TEXT[])")
        params["goals"] = merged
        
    if "preferred_categories" in updates:
        existing = profile.get("preferred_categories") or []
        merged = list(set(existing + updates["preferred_categories"]))
        set_clauses.append("preferred_categories = CAST(:prefs AS TEXT[])")
        params["prefs"] = merged

    if set_clauses:
        try:
            with engine.begin() as conn:
                query = text(f"UPDATE users SET {', '.join(set_clauses)}, updated_at = now() WHERE user_id = :uid")
                conn.execute(query, params)
        except Exception as e:
            print(f"DB Update Error: {str(e)}")
                
    # Trigger a state reload for subsequent nodes so they have the fresh data
    return load_state_node(state)

def determine_intent_node(state: AgentState):
    """Determines if the user wants a plan or is just chatting."""
    try:
        # FIX: Look at the last 4 messages to maintain conversational context!
        recent_msgs = state["messages"][-4:]
        context_str = "\n".join([f"{'User' if m.type == 'human' else 'AI'}: {m.content}" for m in recent_msgs])
        
        intent_analyzer = llm.with_structured_output(IntentSchema)
        result = intent_analyzer.invoke([
            SystemMessage(content=(
                "Analyze the recent conversation context. "
                "Is the user's overarching goal right now to generate/create a workout plan? "
                "If they previously asked for a plan and are now providing their age, weight, goals, or preferences to complete the profile for it, the intent is STILL 'plan'. "
                "If they explicitly ask for a plan, the intent is 'plan'. "
                "If they are just asking general fitness questions, intent is 'chat'."
            )),
            HumanMessage(content=f"Recent Conversation:\n{context_str}")
        ])
        intent_val = result.intent if result else "chat"
        return {"intent": intent_val}
    except Exception as e:
        print(f"Intent Error: {str(e)}")
        return {"intent": "chat"}

def route_intent(state: AgentState) -> str:
    """Hard-routes the graph based on intent and missing fields."""
    intent = state.get("intent", "chat")
    missing = state.get("missing_fields", [])
    
    if intent == "plan":
        if missing:
            return "ask_question"
        else:
            return "generate_plan"
    return "normal_chat"

def ask_question_node(state: AgentState):
    """Politely asks the user for ONE missing profile field."""
    missing = state.get("missing_fields", [])
    target_field = missing[0] if missing else "information"
    
    prompt = f"The user asked for a workout plan, but we are missing: {target_field}. Ask them for this specific field politely and conversationally in 1 short sentence."
    response = llm.invoke([SystemMessage(content=prompt)])
    return {"messages": [response]}

def normal_chat_node(state: AgentState):
    """Handles standard fitness conversation with full context."""
    profile_str = json.dumps(state["user_profile"])
    sys_prompt = f"""You are CoachSaab, a smart AI fitness coach. 
    User Profile: {profile_str}
    Keep responses brief, actionable, and conversational (1-3 sentences max). Do not use markdown styling.
    CRITICAL RULE: NEVER generate a full day-by-day workout plan or routine in chat. If the user asks for a plan, tell them to say 'Generate my plan' so the system can build it officially."""
    
    response = llm.invoke([SystemMessage(content=sys_prompt)] + state["messages"])
    return {"messages": [response]}

def generate_plan_node(state: AgentState):
    """Forces the LLM to output a strict JSON plan, saves to DB, and returns confirmation."""
    try:
        profile_str = json.dumps(state["user_profile"])
        plan_generator = llm.with_structured_output(WorkoutPlanSchema)
        
        # 1. Generate Structured Plan
        plan_data = plan_generator.invoke([
            SystemMessage(content=f"Create a highly effective custom workout plan for this user. User Profile: {profile_str}"),
            HumanMessage(content="Generate my plan.")
        ])
        
        if not plan_data:
            return {"messages": [AIMessage(content="I'm having a little trouble formulating the perfect routine right now. Let's try again!")]}
        
        # 2. Save to Database
        user_id = state["user_id"]
        db_json = plan_data.model_dump()
        db_json["status"] = "active"
        
        plan_name = f"{state['user_profile'].get('name', 'My')} {plan_data.duration_weeks}-Week Plan"
        
        with engine.begin() as conn:
            conn.execute(text("UPDATE workout_plans SET is_active = false WHERE user_id = :uid"), {"uid": user_id})
            query = text("""
                INSERT INTO workout_plans (user_id, plan_name, plan_json, is_active)
                VALUES (:uid, :name, :data, true)
            """)
            conn.execute(query, {"uid": user_id, "name": plan_name, "data": json.dumps(db_json)})
            
        # 3. Return conversational confirmation
        msg = AIMessage(content=f"I've successfully generated your custom {plan_data.duration_weeks}-week '{plan_data.goal}' plan! Check your Home tab to see the breakdown.")
        return {"messages": [msg]}
    except Exception as e:
        print(f"Plan Gen Error: {str(e)}")
        return {"messages": [AIMessage(content=f"Uh oh! I ran into an error generating your plan: {str(e)}")]}

# Compile the Graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("load_state", load_state_node)
graph_builder.add_node("extract_profile", extract_profile_node)
graph_builder.add_node("update_database", update_database_node)
graph_builder.add_node("determine_intent", determine_intent_node)
graph_builder.add_node("ask_question", ask_question_node)
graph_builder.add_node("generate_plan", generate_plan_node)
graph_builder.add_node("normal_chat", normal_chat_node)

graph_builder.add_edge(START, "load_state")
graph_builder.add_edge("load_state", "extract_profile")
graph_builder.add_edge("extract_profile", "update_database")
graph_builder.add_edge("update_database", "determine_intent")
graph_builder.add_conditional_edges("determine_intent", route_intent)
graph_builder.add_edge("ask_question", END)
graph_builder.add_edge("generate_plan", END)
graph_builder.add_edge("normal_chat", END)

agent_graph = graph_builder.compile()

# ==========================================
# FASTAPI ENDPOINTS
# ==========================================
def clean_ai_response(text_content: str) -> str:
    if not text_content:
        return "I'm ready to help you train. What's our focus today?"
    cleaned = re.sub(r'<think>.*?</think>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("```json", "").replace("```markdown", "").replace("```", "")
    cleaned = cleaned.replace("**", "").replace("*", "").replace("### ", "").replace("## ", "").replace("# ", "")
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
def read_root():
    return {"message": "CoachSaab API Active 🚀"}

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
        query = text("INSERT INTO chatbot_conversations (user_id) VALUES (:user_id) RETURNING conversation_id")
        result = conn.execute(query, {"user_id": user_id}).mappings().fetchone()
        return dict(result)

@app.get("/api/v1/chat/conversations/{conversation_id}/messages")
def get_chat_history(conversation_id: str):
    with engine.connect() as conn:
        # Check if conversation exists first to prevent silent failures and foreign key errors
        conv = conn.execute(text("SELECT 1 FROM chatbot_conversations WHERE conversation_id = :c"), {"c": conversation_id}).fetchone()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        query = text("""
            SELECT role, content, created_at 
            FROM chatbot_messages 
            WHERE conversation_id = :conv_id 
            ORDER BY created_at ASC
        """)
        rows = conn.execute(query, {"conv_id": conversation_id}).mappings().fetchall()
        return [{"role": r["role"], "content": clean_ai_response(r["content"]) if r["role"] == "assistant" else r["content"], "created_at": r["created_at"]} for r in rows]

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
        query = text("""
            SELECT plan_name, plan_json, created_at 
            FROM workout_plans 
            WHERE user_id = :uid AND is_active = true 
            ORDER BY created_at DESC LIMIT 1
        """)
        row = conn.execute(query, {"uid": user_id}).mappings().fetchone()
        if row:
            return dict(row)
        return {"status": "no_active_plan"}

@app.post("/api/v1/chat/conversations/{conversation_id}/messages")
def add_chat_message(conversation_id: str, message: ChatMessageCreate):
    # 1. Validate conversation OUTSIDE the try block so it properly returns a 404
    with engine.begin() as conn:
        conv_query = text("SELECT user_id FROM chatbot_conversations WHERE conversation_id = :conv_id")
        conv_row = conn.execute(conv_query, {"conv_id": conversation_id}).mappings().fetchone()
        if not conv_row:
            raise HTTPException(status_code=404, detail="Conversation not found")
        user_id = conv_row['user_id']

    try:
        with engine.begin() as conn:
            # Insert incoming user message first
            conn.execute(
                text("INSERT INTO chatbot_messages (conversation_id, role, content) VALUES (:conv_id, 'user', :content)"),
                {"conv_id": conversation_id, "content": message.content}
            )

            # Fetch chronological message history
            history_query = text("""
                SELECT role, content FROM chatbot_messages 
                WHERE conversation_id = :conv_id 
                ORDER BY created_at ASC
            """)
            history_rows = conn.execute(history_query, {"conv_id": conversation_id}).mappings().fetchall()

            lg_messages = []
            for row in history_rows:
                if row['role'] == 'user':
                    lg_messages.append(HumanMessage(content=row['content']))
                elif row['role'] == 'assistant':
                    lg_messages.append(AIMessage(content=row['content']))

        # Initialize full explicit state to prevent LangGraph KeyErrors
        initial_state = {
            "messages": lg_messages,
            "user_id": user_id,
            "user_profile": {},
            "missing_fields": [],
            "extracted_updates": {},
            "intent": "chat"
        }
        
        # Invoke the Deterministic State Machine
        result = agent_graph.invoke(initial_state)
        raw_reply = result["messages"][-1].content
        ai_reply = clean_ai_response(raw_reply)
        
    except Exception as e:
        print(f"Backend Server Error: {str(e)}")
        # OUTPUT EXACT ERROR TO APP FOR DEBUGGING
        ai_reply = f"SYSTEM ERROR: {str(e)}"

    with engine.begin() as conn:
        insert_ai_msg = text("""
            INSERT INTO chatbot_messages (conversation_id, role, content) 
            VALUES (:conv_id, 'assistant', :content) RETURNING message_id, role, content
        """)
        result = conn.execute(insert_ai_msg, {"conv_id": conversation_id, "content": ai_reply}).mappings().fetchone()
        
    response_dict = dict(result)
    response_dict['content'] = clean_ai_response(response_dict['content'])
    return response_dict