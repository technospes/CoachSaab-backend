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

app = FastAPI(title="CoachSaab API", version="6.0")

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
    exercises: List[str] = Field(..., description="List of exercises WITH sets, reps, or time included in the string. Example: ['Barbell Squat (3 sets x 10 reps)', 'Plank (60 seconds)']. Empty if rest day.")
    is_rest: bool = Field(..., description="True if this is a rest day")

class WorkoutPlanSchema(BaseModel):
    duration_weeks: int = Field(..., gt=0, le=12, description="Duration in weeks. Keep it short (2 to 4 weeks max) for quick user milestones, unless the user explicitly requested a longer plan.")
    goal: str = Field(..., description="Main objective of the plan")
    notes: str = Field(..., description="Additional coaching advice and progression instructions")
    schedule: List[DailyWorkout] = Field(..., description="A 7-day schedule template (Days 1-7)")

# ==========================================
# HELPER: ROBUST JSON PARSER
# ==========================================
def extract_json_from_text(text_content: str) -> dict:
    """Strips <think> tags, markdown, and extracts the first valid JSON object."""
    try:
        # Strip <think> tags
        cleaned = re.sub(r'<think>.*?</think>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
        # Find everything between first { and last }
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = cleaned[start_idx:end_idx+1]
            return json.loads(json_str)
        return {}
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return {}

# ==========================================
# DETERMINISTIC ONBOARDING QUESTIONS
# ==========================================
QUESTIONS = {
    "Age": "Could you please let me know your age?",
    "Weight (kg)": "Could you please tell me your current weight in kilograms?",
    "Fitness Goals": "What are your main fitness goals right now? (e.g., lose fat, build muscle, improve endurance)",
    "Preferred Exercises": "What types of exercises or equipment do you prefer? (e.g., bodyweight, dumbbells, running, home workouts)"
}

# ==========================================
# LANGGRAPH STATE & NODES (Deterministic Flow)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    user_profile: dict
    missing_fields: list
    extracted_updates: dict
    intent: str # "chat" or "onboarding"

def load_state_node(state: AgentState):
    """Loads profile from DB and calculates missing fields deterministically."""
    user_id = state["user_id"]
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT name, age, weight_kg, goals, preferred_categories FROM users WHERE user_id = :uid"), 
            {"uid": user_id}
        ).mappings().fetchone()
    
    if not user_row:
        return {"user_profile": {}, "missing_fields": ["Profile Error"]}

    profile = dict(user_row)
    if profile.get("weight_kg") is not None:
        profile["weight_kg"] = float(profile["weight_kg"])

    missing = []
    if profile.get("age") is None: missing.append("Age")
    if profile.get("weight_kg") is None: missing.append("Weight (kg)")
    if not profile.get("goals"): missing.append("Fitness Goals")
    if not profile.get("preferred_categories"): missing.append("Preferred Exercises")

    return {"user_profile": profile, "missing_fields": missing}

def determine_intent_node(state: AgentState):
    """DETERMINISTIC ROUTER: No LLM used here. Checks chat history for intent."""
    latest_user_msg = state["messages"][-1].content.lower()
    
    # 1. Did the AI just ask an onboarding question?
    ai_asked_question = False
    if len(state["messages"]) >= 2:
        last_ai_msg = state["messages"][-2].content
        if any(q in last_ai_msg for q in QUESTIONS.values()):
            ai_asked_question = True

    # 2. Did the user explicitly trigger a plan request?
    plan_triggers = ["plan", "routine", "program", "schedule", "workout"]
    user_wants_plan = any(t in latest_user_msg for t in plan_triggers)

    # If the user wants a plan, or is currently answering an onboarding question, route to extraction
    if ai_asked_question or user_wants_plan:
        return {"intent": "onboarding"}
    
    return {"intent": "chat"}

def extract_profile_node(state: AgentState):
    """Uses LLM only for semantic extraction (and inferring numbers)."""
    try:
        latest_msg = state["messages"][-1].content
        missing_context = ", ".join(state.get("missing_fields", []))
        
        sys_prompt = f"""You are a strict data extractor. Extract profile data from the user's message.
        User is currently missing these fields: {missing_context}. 
        (CRITICAL: Use the missing fields to infer ambiguous numbers! If age is known but weight is missing, '75' means weight_kg).
        
        Respond ONLY with a valid JSON object containing these keys: 'age' (int), 'weight_kg' (float), 'goals' (list of str), 'preferred_categories' (list of str).
        Omit keys if data is not present. Do not include <think> tags in the final JSON."""
        
        response = llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=latest_msg)
        ])
        
        extracted = extract_json_from_text(response.content)
        valid_keys = {"age", "weight_kg", "goals", "preferred_categories"}
        updates = {k: v for k, v in extracted.items() if k in valid_keys and v is not None}
        
        return {"extracted_updates": updates}
    except Exception as e:
        print(f"Extraction Error: {str(e)}")
        return {"extracted_updates": {}}

def update_database_node(state: AgentState):
    """Merges new extractions into DB. If no updates, state remains unchanged."""
    updates = state.get("extracted_updates", {})
    if not updates:
        return load_state_node(state) # Reload state to check if we should ask the next question
    
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
                
    return load_state_node(state)

def ask_question_node(state: AgentState):
    """DETERMINISTIC: Outputs a hardcoded question. No LLM used."""
    missing = state.get("missing_fields", [])
    target_field = missing[0] if missing else "Age"
    return {"messages": [AIMessage(content=QUESTIONS[target_field])]}

def normal_chat_node(state: AgentState):
    profile_str = json.dumps(state["user_profile"])
    sys_prompt = f"""You are CoachSaab, a smart AI fitness coach. 
    User Profile: {profile_str}
    Keep responses brief, actionable, and conversational (1-3 sentences max). Do not use markdown styling.
    CRITICAL RULE: NEVER generate a full day-by-day workout plan or routine in chat. If they ask for one, just say 'Generate my plan' so the system can build it officially."""
    
    response = llm.invoke([SystemMessage(content=sys_prompt)] + state["messages"])
    return {"messages": [response]}

def generate_plan_node(state: AgentState):
    """Generates the plan and uses robust JSON parsing to bypass <think> tags."""
    try:
        profile_str = json.dumps(state["user_profile"])
        schema_json = WorkoutPlanSchema.model_json_schema()
        
        sys_prompt = f"""Create a highly effective custom workout plan for this user. 
        User Profile: {profile_str}
        
        CRITICAL RULES:
        1. Limit the plan duration to 2, 3, or 4 weeks MAX (unless user specifically asked for longer). We want quick milestones.
        2. In the `exercises` list, you MUST include the exact sets and reps OR time duration in the string itself (e.g., 'Barbell Back Squat (3 sets x 10 reps)').
        3. You MUST respond ONLY with a valid JSON object that strictly matches this JSON schema:
        {json.dumps(schema_json)}
        """
        
        response = llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content="Generate my plan based on my profile. Output pure JSON.")
        ])
        
        extracted_json = extract_json_from_text(response.content)
        if not extracted_json:
             return {"messages": [AIMessage(content="I'm having a little trouble formulating the perfect routine right now. Let's try again!")]}
             
        plan_data = WorkoutPlanSchema.model_validate(extracted_json)
        
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
            
        msg = AIMessage(content=f"I've successfully generated your custom {plan_data.duration_weeks}-week '{plan_data.goal}' plan! Check your Home tab to see the breakdown.")
        return {"messages": [msg]}
    except Exception as e:
        print(f"Plan Gen Error: {str(e)}")
        return {"messages": [AIMessage(content=f"SYSTEM ERROR: {str(e)}")]}

# ==========================================
# GRAPH ROUTING
# ==========================================
def route_intent(state: AgentState) -> str:
    if state.get("intent") == "onboarding":
        return "extract_profile"
    return "normal_chat"

def route_after_update(state: AgentState) -> str:
    if state.get("missing_fields"):
        return "ask_question"
    return "generate_plan"

# Compile the Graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("load_state", load_state_node)
graph_builder.add_node("determine_intent", determine_intent_node)
graph_builder.add_node("extract_profile", extract_profile_node)
graph_builder.add_node("update_database", update_database_node)
graph_builder.add_node("ask_question", ask_question_node)
graph_builder.add_node("generate_plan", generate_plan_node)
graph_builder.add_node("normal_chat", normal_chat_node)

graph_builder.add_edge(START, "load_state")
graph_builder.add_edge("load_state", "determine_intent")
graph_builder.add_conditional_edges("determine_intent", route_intent)
graph_builder.add_edge("extract_profile", "update_database")
graph_builder.add_conditional_edges("update_database", route_after_update)
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

class DayCompletionToggle(BaseModel):
    user_id: str
    week_number: int
    day_number: int
    is_completed: bool

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
            SELECT plan_id, user_id, plan_name, plan_json, created_at 
            FROM workout_plans 
            WHERE user_id = :uid AND is_active = true 
            ORDER BY created_at DESC LIMIT 1
        """)
        row = conn.execute(query, {"uid": user_id}).mappings().fetchone()
        if row:
            result_dict = dict(row)
            result_dict['plan_id'] = str(result_dict['plan_id'])
            result_dict['user_id'] = str(result_dict['user_id'])
            return result_dict
        return {"status": "no_active_plan"}

@app.get("/api/v1/plans/{plan_id}/completions")
def get_plan_completions(plan_id: str):
    with engine.connect() as conn:
        query = text("SELECT week_number, day_number FROM plan_day_completions WHERE plan_id = :pid AND is_completed = true")
        rows = conn.execute(query, {"pid": plan_id}).mappings().fetchall()
        return [f"w{r['week_number']}_d{r['day_number']}" for r in rows]

@app.post("/api/v1/plans/{plan_id}/completions/toggle")
def toggle_plan_completion(plan_id: str, payload: DayCompletionToggle):
    with engine.begin() as conn:
        # SECURITY PATCH: Verify the user actually owns this plan!
        plan_check = conn.execute(
            text("SELECT 1 FROM workout_plans WHERE plan_id = :pid AND user_id = :uid"),
            {"pid": plan_id, "uid": payload.user_id}
        ).fetchone()
        
        if not plan_check:
            raise HTTPException(status_code=403, detail="Unauthorized: Plan does not belong to user")
            
        if payload.is_completed:
            query = text("""
                INSERT INTO plan_day_completions (plan_id, user_id, week_number, day_number, is_completed)
                VALUES (:pid, :uid, :wn, :dn, true)
                ON CONFLICT (plan_id, week_number, day_number) 
                DO UPDATE SET is_completed = true, completed_at = now()
            """)
        else:
            query = text("""
                DELETE FROM plan_day_completions 
                WHERE plan_id = :pid AND week_number = :wn AND day_number = :dn
            """)
        
        conn.execute(query, {
            "pid": plan_id, 
            "uid": payload.user_id,
            "wn": payload.week_number, 
            "dn": payload.day_number
        })
    return {"status": "success"}

@app.post("/api/v1/chat/conversations/{conversation_id}/messages")
def add_chat_message(conversation_id: str, message: ChatMessageCreate):
    with engine.begin() as conn:
        conv_query = text("SELECT user_id FROM chatbot_conversations WHERE conversation_id = :conv_id")
        conv_row = conn.execute(conv_query, {"conv_id": conversation_id}).mappings().fetchone()
        if not conv_row:
            raise HTTPException(status_code=404, detail="Conversation not found")
        user_id = conv_row['user_id']

    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO chatbot_messages (conversation_id, role, content) VALUES (:conv_id, 'user', :content)"),
                {"conv_id": conversation_id, "content": message.content}
            )

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

        initial_state = {
            "messages": lg_messages,
            "user_id": user_id,
            "user_profile": {},
            "missing_fields": [],
            "extracted_updates": {},
            "intent": "chat"
        }
        
        result = agent_graph.invoke(initial_state)
        raw_reply = result["messages"][-1].content
        ai_reply = clean_ai_response(raw_reply)
        
    except Exception as e:
        print(f"Backend Server Error: {str(e)}")
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