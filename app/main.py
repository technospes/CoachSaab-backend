from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from typing import Optional, List, Annotated
import os
import json
import re
from dotenv import load_dotenv

# LangGraph & LangChain Imports
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI(title="CoachSaab API", version="4.0")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:mysecretpassword@localhost:5433/aicoach")
engine = create_engine(DATABASE_URL)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ==========================================
# STRICT PYDANTIC SCHEMAS (Notice: NO user_id!)
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
    age: Optional[int] = Field(None, ge=13, le=120, description="User's age")
    weight_kg: Optional[float] = Field(None, gt=20, lt=300, description="User's weight in kg")
    goals: Optional[List[str]] = Field(None, description="User's fitness goals (e.g. weight loss)")
    preferred_categories: Optional[List[str]] = Field(None, description="Preferred exercise categories")

# ==========================================
# LANGGRAPH TOOLS (Secure & Deterministic)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def update_user_profile(profile_data: UserProfileUpdate, config: RunnableConfig) -> str:
    """
    Updates the user's profile in the database. 
    ALWAYS use this tool when the user provides age, weight, preferences, or goals.
    """
    # 1. Securely extract the user_id injected by the backend
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "SYSTEM ERROR: User ID not found in context."

    updates = []
    params = {"uid": user_id}
    
    if profile_data.age is not None:
        updates.append("age = :age")
        params["age"] = profile_data.age
    if profile_data.weight_kg is not None:
        updates.append("weight_kg = :weight")
        params["weight"] = profile_data.weight_kg
    if profile_data.goals:
        updates.append("goals = :goals")
        params["goals"] = profile_data.goals
    if profile_data.preferred_categories:
        updates.append("preferred_categories = :prefs")
        params["prefs"] = profile_data.preferred_categories
        
    if not updates:
        return "No data provided to update."
        
    try:
        with engine.begin() as conn:
            query = text(f"UPDATE users SET {', '.join(updates)}, updated_at = now() WHERE user_id = :uid")
            conn.execute(query, params)
            
            # 2. Mid-Turn State Refresh (Cures Staleness)
            fresh_user = conn.execute(text("SELECT age, weight_kg, goals, preferred_categories FROM users WHERE user_id = :uid"), {"uid": user_id}).mappings().fetchone()
            
        return f"Profile successfully updated! New DB State: {dict(fresh_user)}"
    except Exception as e:
        return f"SYSTEM ERROR: Profile update failed in database: {str(e)}"

@tool
def save_workout_plan(plan_name: str, plan_data: WorkoutPlanSchema, config: RunnableConfig) -> str:
    """
    Generates and saves a structured workout plan. 
    Call this ONLY when the user asks for a routine AND their profile is complete.
    """
    user_id = config.get("configurable", {}).get("user_id")
    
    # 1. Deterministic Guardrail (Rejects the LLM if it tries to skip onboarding)
    try:
        with engine.connect() as conn:
            user_row = conn.execute(text("SELECT age, weight_kg, goals, preferred_categories FROM users WHERE user_id = :uid"), {"uid": user_id}).mappings().fetchone()
    except Exception as e:
        return f"SYSTEM ERROR: Failed to fetch profile from DB: {str(e)}"
        
    missing_fields = []
    if user_row['age'] is None: missing_fields.append("Age")
    if user_row['weight_kg'] is None: missing_fields.append("Weight")
    if not user_row['goals']: missing_fields.append("Goal")
    if not user_row['preferred_categories']: missing_fields.append("Preferences")
    
    if missing_fields:
        return f"SYSTEM REJECTION: Profile is incomplete. You MUST ask the user for their: {', '.join(missing_fields)}. Ask for exactly ONE field at a time. DO NOT generate the plan yet."

    # 2. Save to Database
    db_json = plan_data.model_dump()
    db_json["status"] = "active"
    
    try:
        with engine.begin() as conn:
            query = text("""
                INSERT INTO workout_plans (user_id, plan_name, plan_json)
                VALUES (:uid, :name, :data)
            """)
            conn.execute(query, {
                "uid": user_id, 
                "name": plan_name, 
                "data": json.dumps(db_json)
            })
            
        return f"Success! The '{plan_name}' plan has been securely saved to the database."
    except Exception as e:
        return f"SYSTEM ERROR: Failed to save to database. Tell the user an internal database error occurred: {str(e)}"

# Initialize Groq and bind tools
tools = [save_workout_plan, update_user_profile]
llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
llm_with_tools = llm.bind_tools(tools)

def chatbot_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(AgentState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
agent_graph = graph_builder.compile()

# ==========================================
# HELPER & ENDPOINTS
# ==========================================
def clean_ai_response(text_content: str) -> str:
    if not text_content:
        return "Hello! How can I assist you today?"
    cleaned = re.sub(r'<think>.*?</think>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("```json", "").replace("```markdown", "").replace("```", "")
    cleaned = cleaned.replace("**", "").replace("*", "").replace("### ", "").replace("## ", "").replace("# ", "")
    return cleaned.strip()

class ChatMessageCreate(BaseModel):
    role: str
    content: str
    context_snapshot: Optional[dict] = None

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
        query = text("""
            SELECT role, content, created_at 
            FROM chatbot_messages 
            WHERE conversation_id = :conv_id 
            ORDER BY created_at ASC
        """)
        rows = conn.execute(query, {"conv_id": conversation_id}).mappings().fetchall()
        
        formatted_messages = []
        for r in rows:
            row_dict = dict(r)
            if row_dict['role'] == 'assistant':
                row_dict['content'] = clean_ai_response(row_dict['content'])
            formatted_messages.append(row_dict)
            
        return formatted_messages

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
    """Fetches the latest active workout plan for the user."""
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
    try:
        with engine.begin() as conn:
            conv_query = text("SELECT user_id FROM chatbot_conversations WHERE conversation_id = :conv_id")
            conv_row = conn.execute(conv_query, {"conv_id": conversation_id}).mappings().fetchone()
            if not conv_row:
                raise HTTPException(status_code=404, detail="Conversation not found")
            user_id = conv_row['user_id']

            user_query = text("SELECT name, gender, age, weight_kg, goals, preferred_categories FROM users WHERE user_id = :uid")
            user_row = conn.execute(user_query, {"uid": user_id}).mappings().fetchone()
            
            user_name = user_row['name'] if user_row else "Athlete"
            user_gender = user_row['gender'] if user_row else "Not specified"
            user_age = user_row['age'] if user_row and user_row['age'] else "Unknown"
            user_weight = user_row['weight_kg'] if user_row and user_row['weight_kg'] else "Unknown"
            user_goals = user_row['goals'] if user_row and user_row['goals'] else "Unknown"
            user_prefs = user_row['preferred_categories'] if user_row and user_row['preferred_categories'] else "Unknown"

            sessions_query = text("""
                SELECT activity_key, reps, duration_seconds, form_score, dominant_deviation 
                FROM workout_sessions 
                WHERE user_id = :uid ORDER BY created_at DESC LIMIT 3
            """)
            session_rows = conn.execute(sessions_query, {"uid": user_id}).mappings().fetchall()
            
            workout_context_summary = "No recent workouts recorded."
            if session_rows:
                summaries = [
                    f"- {s['activity_key'].capitalize()}: {s['reps']} reps, Score: {s['form_score']}%, Issue: {s['dominant_deviation'] or 'None'}"
                    for s in session_rows
                ]
                workout_context_summary = "\n".join(summaries)

            insert_user_msg = text("INSERT INTO chatbot_messages (conversation_id, role, content) VALUES (:conv_id, 'user', :content)")
            conn.execute(insert_user_msg, {"conv_id": conversation_id, "content": message.content})

            history_query = text("""
                SELECT role, content FROM chatbot_messages 
                WHERE conversation_id = :conv_id ORDER BY created_at DESC LIMIT 10
            """)
            history_rows = conn.execute(history_query, {"conv_id": conversation_id}).mappings().fetchall()
            
            # The Progressive Collection Prompt
            system_prompt = f"""You are CoachSaab, an expert AI fitness coach and Agent Orchestrator.
Your goal is to guide the user, analyze their workout history, and generate structured workout plans.

User Profile:
Name: {user_name}
Gender: {user_gender}
Age: {user_age}
Weight: {user_weight}
Goals: {user_goals}
Preferences: {user_prefs}

Recent Workouts:
{workout_context_summary}

CRITICAL RULES:
1. Ask for EXACTLY ONE missing profile field at a time. Never ask multiple questions in one message.
2. Use `update_user_profile` to save information as soon as the user provides it.
3. Generate the plan and call `save_workout_plan` ONLY when Age, Weight, Goals, and Preferences are all known.
4. Keep conversational responses short, empathetic, and direct (1-3 sentences). No markdown styling.
"""

            lg_messages = [SystemMessage(content=system_prompt)]
            for row in reversed(history_rows):
                if row['role'] == 'user':
                    lg_messages.append(HumanMessage(content=row['content']))
                elif row['role'] == 'assistant':
                    lg_messages.append(AIMessage(content=row['content']))
                    
            lg_messages.append(HumanMessage(content=message.content))

        # Securely inject user_id into the LangGraph execution context
        runnable_config = {"configurable": {"user_id": user_id}}
        
        result = agent_graph.invoke({"messages": lg_messages}, config=runnable_config)
        raw_reply = result["messages"][-1].content
        ai_reply = clean_ai_response(raw_reply)
        
    except Exception as e:
        print(f"Backend Server Error: {str(e)}")
        ai_reply = f"Sorry {user_name}, I'm having trouble reasoning right now. Please try again in a moment."

    with engine.begin() as conn:
        insert_ai_msg = text("""
            INSERT INTO chatbot_messages (conversation_id, role, content) 
            VALUES (:conv_id, 'assistant', :content) RETURNING message_id, role, content
        """)
        result = conn.execute(insert_ai_msg, {"conv_id": conversation_id, "content": ai_reply}).mappings().fetchone()
        
    response_dict = dict(result)
    response_dict['content'] = clean_ai_response(response_dict['content'])
    return response_dict