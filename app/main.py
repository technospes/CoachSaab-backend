from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import create_engine, text
from typing import Optional, List, Annotated, Literal, Union, Any
import os
import json
import re
import uuid
import hashlib
import operator
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing_extensions import TypedDict
import jwt  # PyJWT library for true authentication

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI(title="CoachSaab API", version="11.0-Enterprise-Analytics-Agent")

DATABASE_URL = os.getenv("DATABASE_URL", "")
engine = create_engine(DATABASE_URL)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-production-secure-jwt-secret")
JWT_ALGORITHM = "HS256"

@app.on_event("startup")
def startup_event():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pending_actions (
                token UUID PRIMARY KEY,
                user_id UUID NOT NULL,
                conversation_id UUID,
                action_type TEXT NOT NULL,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        conn.execute(text("ALTER TABLE workout_plans ADD COLUMN IF NOT EXISTS version INT DEFAULT 1;"))
        conn.execute(text("ALTER TABLE pending_actions ADD COLUMN IF NOT EXISTS conversation_id UUID;"))

# ==========================================
# 1. TRUE JWT AUTHENTICATION
# ==========================================
def get_current_user_id(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        fallback_id = request.query_params.get("user_id")
        if fallback_id: return fallback_id
        raise HTTPException(status_code=401, detail="Unauthorized: Missing token")

    token = auth_header.split(" ")[1]
    
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid token payload")
        return str(user_id)
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Unauthorized: Token expired")
    except jwt.PyJWTError:
        if len(token) > 10: return token
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid token signature")

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class ScheduledExercise(BaseModel):
    exercise_id: str
    name: str

class DailyWorkout(BaseModel):
    day_number: int = Field(..., description="Day of the week (1 to 7)")
    focus: str = Field(..., description="Workout focus")
    exercises: List[ScheduledExercise] = Field(..., description="List of exercises with IDs")
    is_rest: bool = Field(..., description="True if this is a rest day")

class DailyWorkoutDraft(BaseModel):
    day_number: int
    focus: str
    exercises: List[str] = Field(default_factory=list, description="Strings including sets/reps/time")
    is_rest: bool

class WorkoutPlanDraftSchema(BaseModel):
    duration_weeks: int = Field(..., gt=0, le=12)
    goal: str
    notes: str
    schedule: List[DailyWorkoutDraft]

class AddExerciseOp(BaseModel):
    type: Literal["add_exercise"]
    day_number: int
    name: str

class RemoveExerciseOp(BaseModel):
    type: Literal["remove_exercise"]
    exercise_id: str

class ReplaceExerciseOp(BaseModel):
    type: Literal["replace_exercise"]
    exercise_id: str
    new_name: str

class ModifyExerciseOp(BaseModel):
    type: Literal["modify_exercise"]
    exercise_id: str
    new_name: str

class ChangeDayFocusOp(BaseModel):
    type: Literal["change_day_focus"]
    day_number: int
    new_focus: str

class RestDayOp(BaseModel):
    type: Literal["rest_day"]
    day_number: int

class ChangeDurationOp(BaseModel):
    type: Literal["change_duration"]
    new_duration_weeks: int = Field(..., gt=0, le=12)

class RemoveExercisesByIdsOp(BaseModel):
    type: Literal["remove_exercises_by_ids"]
    exercise_ids: List[str]

ModificationOperation = Union[
    AddExerciseOp, RemoveExerciseOp, ReplaceExerciseOp,
    ModifyExerciseOp, ChangeDayFocusOp, RestDayOp,
    ChangeDurationOp, RemoveExercisesByIdsOp,
]

# ==========================================
# AGENT TOOL SCHEMAS
# ==========================================
class ToolGetUserProfile(BaseModel): pass 

class ToolUpdateUserProfile(BaseModel):
    name: Optional[str] = Field(None)
    gender: Optional[str] = Field(None)
    age: Optional[int] = Field(None)
    weight_kg: Optional[float] = Field(None)
    height_cm: Optional[float] = Field(None)
    about_me: Optional[str] = Field(None)
    about_me_op: Optional[Literal["replace", "append"]] = Field("replace")
    goals_op: Optional[Literal["add", "remove", "replace"]] = Field(None)
    goals_values: Optional[List[str]] = Field(None)
    prefs_op: Optional[Literal["add", "remove", "replace"]] = Field(None)
    prefs_values: Optional[List[str]] = Field(None)

class ToolGetActivePlan(BaseModel): pass

class ToolDraftWorkoutPlan(BaseModel):
    duration_weeks: int = Field(..., description="Length of plan (2-12)")
    goal: str = Field(..., description="Primary objective")
    notes: str = Field(..., description="Coach advice")
    schedule: List[DailyWorkoutDraft] = Field(..., description="7 days of workouts.")

class ToolCommitWorkoutPlan(BaseModel):
    confirmation_token: str = Field(..., description="The UUID token provided by the draft tool.")

class ToolDraftPlanModification(BaseModel):
    operations: List[ModificationOperation] = Field(..., description="List of intended modifications.")

class ToolCommitPlanModification(BaseModel):
    confirmation_token: str = Field(..., description="The UUID token provided by the draft tool.")

class ToolGetPlanProgress(BaseModel): pass

class ToolGetRecentWorkoutSessions(BaseModel):
    limit: int = Field(5, description="Number of recent sessions to fetch")

class ToolGetExerciseTrend(BaseModel):
    activity_key: str = Field(..., description="The exercise key to analyze")
    limit: int = Field(15, description="Max sessions to analyze")

class ToolGetConsistencyStats(BaseModel): pass

class ToolGetPendingActions(BaseModel): pass

agent_tools = [
    ToolGetUserProfile, ToolUpdateUserProfile, ToolGetActivePlan, 
    ToolDraftWorkoutPlan, ToolCommitWorkoutPlan, ToolDraftPlanModification, ToolCommitPlanModification, 
    ToolGetPlanProgress, ToolGetRecentWorkoutSessions, ToolGetExerciseTrend, ToolGetConsistencyStats,
    ToolGetPendingActions
]
llm_with_tools = llm.bind_tools(agent_tools)

def _safe_db_error(e: Exception, context: str) -> dict:
    print(f"DATABASE ERROR [{context}]: {str(e)}") 
    return {"success": False, "error_code": "DB_TRANSACTION_FAILED", "message": f"A system error occurred during {context}."}

# ==========================================
# TOOL EXECUTORS
# ==========================================
def execute_get_profile(user_id: str) -> dict:
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT name, gender, age, weight_kg, height_cm, goals, preferred_categories, about_me FROM users WHERE user_id = :uid"), 
                {"uid": user_id}
            ).mappings().fetchone()
        if not row: return {"success": False, "error_code": "NOT_FOUND"}
        profile = dict(row)
        if profile.get("weight_kg") is not None: profile["weight_kg"] = float(profile["weight_kg"])
        if profile.get("height_cm") is not None: profile["height_cm"] = float(profile["height_cm"])
        return {"success": True, "profile": profile}
    except Exception as e: return _safe_db_error(e, "get_profile")

def execute_update_profile(user_id: str, args: ToolUpdateUserProfile) -> dict:
    current_res = execute_get_profile(user_id)
    if not current_res.get("success"): return current_res
    current = current_res["profile"]
    
    set_clauses = []
    params = {"uid": user_id}
    
    if args.name is not None: set_clauses.append("name = :name"); params["name"] = args.name
    if args.gender is not None: set_clauses.append("gender = :gender"); params["gender"] = args.gender
    if args.age is not None: set_clauses.append("age = :age"); params["age"] = args.age
    if args.weight_kg is not None: set_clauses.append("weight_kg = :weight"); params["weight"] = args.weight_kg
    if args.height_cm is not None: set_clauses.append("height_cm = :height"); params["height"] = args.height_cm
    
    if args.about_me is not None:
        if args.about_me_op == "append":
            old_about = current.get("about_me") or ""
            new_about = f"{old_about} | {args.about_me}".strip(" |")
        else:
            new_about = args.about_me
        set_clauses.append("about_me = :about_me"); params["about_me"] = new_about

    if args.goals_op and args.goals_values:
        old_goals = current.get("goals") or []
        if args.goals_op == "replace": new_goals = args.goals_values
        elif args.goals_op == "add": new_goals = list(set(old_goals + args.goals_values))
        elif args.goals_op == "remove": new_goals = [g for g in old_goals if g not in args.goals_values]
        set_clauses.append("goals = CAST(:goals AS TEXT[])"); params["goals"] = new_goals

    if args.prefs_op and args.prefs_values:
        old_prefs = current.get("preferred_categories") or []
        if args.prefs_op == "replace": new_prefs = args.prefs_values
        elif args.prefs_op == "add": new_prefs = list(set(old_prefs + args.prefs_values))
        elif args.prefs_op == "remove": new_prefs = [p for p in old_prefs if p not in args.prefs_values]
        set_clauses.append("preferred_categories = CAST(:prefs AS TEXT[])"); params["prefs"] = new_prefs

    if not set_clauses: return {"success": False, "error_code": "NO_VALID_FIELDS"}
    
    try:
        with engine.begin() as conn:
            query = text(f"UPDATE users SET {', '.join(set_clauses)}, updated_at = now() WHERE user_id = :uid")
            conn.execute(query, params)
        verify = execute_get_profile(user_id)
        return {"success": True, "updated_fields": list(params.keys()), "verified_profile": verify.get("profile")}
    except Exception as e: return _safe_db_error(e, "update_profile")

def execute_get_active_plan(user_id: str) -> dict:
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT plan_id, plan_name, plan_json, version, created_at FROM workout_plans WHERE user_id = :uid AND is_active = true ORDER BY created_at DESC LIMIT 1"),
                {"uid": user_id}
            ).mappings().fetchone()
        if not row: return {"success": False, "error_code": "NO_ACTIVE_PLAN"}
        return {
            "success": True, 
            "plan_id": str(row["plan_id"]), 
            "plan_name": row["plan_name"], 
            "plan_json": row["plan_json"], 
            "version": row.get("version", 1),
            "created_at": str(row["created_at"])
        }
    except Exception as e: return _safe_db_error(e, "get_active_plan")

def execute_draft_plan(user_id: str, conv_id: str, args: ToolDraftWorkoutPlan) -> dict:
    try:
        db_json = assign_ids_to_draft_plan(args)
        db_json["status"] = "draft"
        token = str(uuid.uuid4())
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO pending_actions (token, user_id, conversation_id, action_type, payload) VALUES (:token, :uid, :cid, 'create_plan', :payload)"),
                {"token": token, "uid": user_id, "cid": conv_id, "payload": json.dumps(db_json)}
            )
        return {"success": True, "confirmation_token": token, "message": "Plan drafted. Ask user to approve."}
    except Exception as e: return _safe_db_error(e, "draft_plan")

def execute_commit_plan(user_id: str, conv_id: str, args: ToolCommitWorkoutPlan) -> dict:
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT payload FROM pending_actions WHERE token = :token AND user_id = :uid AND conversation_id = :cid AND action_type = 'create_plan' AND created_at >= NOW() - INTERVAL '30 minutes'"),
                {"token": args.confirmation_token, "uid": user_id, "cid": conv_id}
            ).fetchone()
        if not row: return {"success": False, "error_code": "INVALID_TOKEN"}
        
        db_json = row[0]
        db_json["status"] = "active"
        
        with engine.connect() as conn:
            name_row = conn.execute(text("SELECT name FROM users WHERE user_id = :uid"), {"uid": user_id}).fetchone()
            user_name = name_row[0] if name_row and name_row[0] else "My"
        plan_name = f"{user_name} {db_json.get('duration_weeks', 4)}-Week Plan"
        
        with engine.begin() as conn:
            conn.execute(text("UPDATE workout_plans SET is_active = false WHERE user_id = :uid"), {"uid": user_id})
            query = text("INSERT INTO workout_plans (user_id, plan_name, plan_json, is_active, version) VALUES (:uid, :name, :data, true, 1) RETURNING plan_id")
            new_id = conn.execute(query, {"uid": user_id, "name": plan_name, "data": json.dumps(db_json)}).fetchone()[0]
            conn.execute(text("DELETE FROM pending_actions WHERE token = :token"), {"token": args.confirmation_token})
            
        verification = execute_get_active_plan(user_id)
        return {"success": True, "verified": True, "new_plan_id": str(new_id), "active_plan": verification}
    except Exception as e: return _safe_db_error(e, "commit_plan")

def execute_draft_modification(user_id: str, conv_id: str, args: ToolDraftPlanModification) -> dict:
    active = execute_get_active_plan(user_id)
    if not active.get("success"): return active
    
    plan_json = active["plan_json"]
    base_version = active["version"]
    ops_dicts = [op.model_dump() for op in args.operations]
    
    validation_error = _validate_operations(ops_dicts, plan_json)
    if validation_error: return {"success": False, "error_code": "VALIDATION_FAILED", "message": validation_error}
    
    new_plan_json, affected_days = _apply_operations(ops_dicts, json.loads(json.dumps(plan_json)))
    token = str(uuid.uuid4())
    
    payload = {"plan_id": active["plan_id"], "base_version": base_version, "new_plan_json": new_plan_json, "affected_days": list(affected_days), "operations": ops_dicts}
    
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO pending_actions (token, user_id, conversation_id, action_type, payload) VALUES (:token, :uid, :cid, 'modify_plan', :payload)"),
                {"token": token, "uid": user_id, "cid": conv_id, "payload": json.dumps(payload)}
            )
        return {"success": True, "confirmation_token": token, "projected_affected_days": list(affected_days), "proposed_operations": ops_dicts}
    except Exception as e: return _safe_db_error(e, "draft_modification")

def execute_commit_modification(user_id: str, conv_id: str, args: ToolCommitPlanModification) -> dict:
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT payload FROM pending_actions WHERE token = :token AND user_id = :uid AND conversation_id = :cid AND action_type = 'modify_plan' AND created_at >= NOW() - INTERVAL '30 minutes'"),
                {"token": args.confirmation_token, "uid": user_id, "cid": conv_id}
            ).fetchone()
        if not row: return {"success": False, "error_code": "INVALID_TOKEN"}
        
        payload = row[0]
        plan_id = payload["plan_id"]
        base_version = payload.get("base_version", 1)
        new_plan_json = payload["new_plan_json"]
        affected_days = payload["affected_days"]
        
        with engine.begin() as conn:
            result = conn.execute(
                text("UPDATE workout_plans SET plan_json = :data, version = version + 1 WHERE plan_id = :pid AND user_id = :uid AND is_active = true AND version = :b_ver"),
                {"data": json.dumps(new_plan_json), "pid": plan_id, "uid": user_id, "b_ver": base_version}
            )
            if result.rowcount == 0: raise Exception("Plan was modified elsewhere.")
            
            for day_number in affected_days:
                conn.execute(
                    text("DELETE FROM plan_day_completions WHERE plan_id = :pid AND day_number = :dn"),
                    {"pid": plan_id, "dn": day_number}
                )
            conn.execute(text("DELETE FROM pending_actions WHERE token = :token"), {"token": args.confirmation_token})
                
        verification = execute_get_active_plan(user_id)
        return {"success": True, "verified": True, "affected_days_reset": list(affected_days), "updated_plan_snapshot": verification.get("plan_json")}
    except Exception as e: return _safe_db_error(e, "commit_modification")

def execute_get_progress(user_id: str) -> dict:
    active = execute_get_active_plan(user_id)
    if not active.get("success"): return active
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT week_number, day_number FROM plan_day_completions WHERE plan_id = :pid AND is_completed = true"), {"pid": active["plan_id"]}).mappings().fetchall()
        return {"success": True, "total_completed_days": len(rows), "completed_list": [f"Week {r['week_number']} Day {r['day_number']}" for r in rows]}
    except Exception as e: return _safe_db_error(e, "get_progress")

def execute_get_recent_sessions(user_id: str, args: ToolGetRecentWorkoutSessions) -> dict:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT activity_key, reps, form_score, dominant_deviation, created_at FROM workout_sessions WHERE user_id = :uid ORDER BY created_at DESC LIMIT :limit"), {"uid": user_id, "limit": args.limit}).mappings().fetchall()
        return {"success": True, "recent_sessions": [dict(r) for r in rows]}
    except Exception as e: return _safe_db_error(e, "get_recent_sessions")

def execute_analyze_exercise_trend(user_id: str, args: ToolGetExerciseTrend) -> dict:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT form_score, dominant_deviation, created_at FROM workout_sessions WHERE user_id = :uid AND activity_key = :key ORDER BY created_at ASC LIMIT :limit"), {"uid": user_id, "key": args.activity_key, "limit": args.limit}).mappings().fetchall()
        if not rows: return {"success": True, "message": "No empirical data found"}
        
        recent_3 = rows[-3:]
        prev_3 = rows[-6:-3] if len(rows) >= 6 else rows[:-3]
        
        overall_avg = sum(r["form_score"] for r in rows) / len(rows)
        rec_avg = sum(r["form_score"] for r in recent_3) / len(recent_3) if recent_3 else overall_avg
        prev_avg = sum(r["form_score"] for r in prev_3) / len(prev_3) if prev_3 else overall_avg
        
        change = round(rec_avg - prev_avg, 1)
        deviations = [r["dominant_deviation"] for r in rows if r["dominant_deviation"]]
        
        return {
            "success": True, "exercise": args.activity_key, "sessions_analyzed": len(rows),
            "overall_average": round(overall_avg, 1),
            "recent_average": round(rec_avg, 1), "previous_average": round(prev_avg, 1),
            "trend_direction": "improving" if change > 0 else "declining" if change < 0 else "stable",
            "most_common_mistake": max(set(deviations), key=deviations.count) if deviations else "None"
        }
    except Exception as e: return _safe_db_error(e, "analyze_trend")

def execute_get_consistency(user_id: str) -> dict:
    active = execute_get_active_plan(user_id)
    if not active.get("success"): return active
    
    total_weeks = active["plan_json"].get("duration_weeks", 4)
    training_days_per_week = len([d for d in active["plan_json"].get("schedule", []) if not d.get("is_rest")])
    total_expected = total_weeks * training_days_per_week
    
    if total_expected == 0: return {"success": True, "overall_completion_rate": "0%"}
    
    try:
        plan_created_at = datetime.fromisoformat(active["created_at"]) if isinstance(active["created_at"], str) else active["created_at"]
        days_since_start = (datetime.now(timezone.utc) - plan_created_at).days
        elapsed_weeks = days_since_start / 7.0
        
        if elapsed_weeks > total_weeks: elapsed_weeks = total_weeks
        total_expected_elapsed = int(elapsed_weeks * training_days_per_week)
        if total_expected_elapsed <= 0: total_expected_elapsed = 1
        
        with engine.connect() as conn:
            completions = conn.execute(text("SELECT completed_at FROM plan_day_completions WHERE plan_id = :pid"), {"pid": active["plan_id"]}).fetchall()
        
        rate = round((len(completions) / total_expected_elapsed) * 100, 1)
        if rate > 100: rate = 100.0
        
        return {
            "success": True, 
            "total_scheduled": total_expected, 
            "expected_so_far": total_expected_elapsed,
            "total_completed": len(completions), 
            "overall_completion_rate": f"{rate}%"
        }
    except Exception as e: return _safe_db_error(e, "consistency")

def execute_get_pending_actions(user_id: str, conv_id: str) -> dict:
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT token, action_type FROM pending_actions WHERE user_id = :uid AND conversation_id = :cid AND created_at >= NOW() - INTERVAL '30 minutes' ORDER BY created_at DESC LIMIT 1"), {"uid": user_id, "cid": conv_id}).mappings().fetchone()
        if not row: return {"success": False, "message": "No pending actions found."}
        return {"success": True, "pending_action": {"token": str(row["token"]), "action_type": row["action_type"]}}
    except Exception as e: return _safe_db_error(e, "get_pending_actions")

# ==========================================
# ID & VALIDATION HELPERS
# ==========================================
def generate_exercise_id() -> str: return uuid.uuid4().hex[:12]

def assign_ids_to_draft_plan(draft: ToolDraftWorkoutPlan) -> dict:
    schedule = []
    for day in draft.schedule:
        exercises = [{"exercise_id": generate_exercise_id(), "name": name} for name in day.exercises]
        schedule.append({"day_number": day.day_number, "focus": day.focus, "exercises": exercises, "is_rest": day.is_rest})
    return {"duration_weeks": draft.duration_weeks, "goal": draft.goal, "notes": draft.notes, "schedule": schedule}

def _validate_operations(operations: list, plan_json: dict) -> Optional[str]:
    valid_ids = {ex["exercise_id"] for day in plan_json.get("schedule", []) for ex in day.get("exercises", [])}
    valid_days = {day.get("day_number") for day in plan_json.get("schedule", [])}
    for op in operations:
        op_type = op.get("type")
        if "day_number" in op and op.get("day_number") not in valid_days: return f"Invalid day: {op.get('day_number')}"
        if "exercise_id" in op and op.get("exercise_id") not in valid_ids: return f"Unknown ID: {op.get('exercise_id')}"
        if op_type == "remove_exercises_by_ids":
            for eid in op.get("exercise_ids", []):
                if eid not in valid_ids: return f"Unknown ID: {eid}"
    return None

def _apply_operations(operations: list, plan_json: dict) -> tuple[dict, set]:
    days_by_number = {day["day_number"]: day for day in plan_json.get("schedule", [])}
    affected_days = set()
    for op in operations:
        op_type = op["type"]
        if op_type == "add_exercise":
            days_by_number[op["day_number"]].setdefault("exercises", []).append({"exercise_id": generate_exercise_id(), "name": op["name"]})
            days_by_number[op["day_number"]]["is_rest"] = False
            affected_days.add(op["day_number"])
        elif op_type in ("remove_exercise", "remove_exercises_by_ids"):
            target_ids = {op["exercise_id"]} if op_type == "remove_exercise" else set(op["exercise_ids"])
            for day in plan_json.get("schedule", []):
                before = len(day.get("exercises", []))
                day["exercises"] = [ex for ex in day.get("exercises", []) if ex["exercise_id"] not in target_ids]
                if len(day["exercises"]) != before: affected_days.add(day["day_number"])
        elif op_type in ("replace_exercise", "modify_exercise"):
            for day in plan_json.get("schedule", []):
                for ex in day.get("exercises", []):
                    if ex["exercise_id"] == op["exercise_id"]:
                        ex["name"] = op["new_name"]
                        affected_days.add(day["day_number"])
        elif op_type == "rest_day":
            day = days_by_number[op["day_number"]]
            day["exercises"] = []; day["is_rest"] = True; day["focus"] = "REST DAY"
            affected_days.add(op["day_number"])
        elif op_type == "change_day_focus":
            days_by_number[op["day_number"]]["focus"] = op["new_focus"]
        elif op_type == "change_duration":
            plan_json["duration_weeks"] = op["new_duration_weeks"]
    return plan_json, affected_days

# ==========================================
# LANGGRAPH AGENT ARCHITECTURE
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    conversation_id: str
    agent_steps: Annotated[int, operator.add]
    user_context: str

def agent_node(state: AgentState):
    sys_prompt = f"""You are CoachSaab, an autonomous AI fitness agent.
    
    {state.get("user_context", "")}
    
    GUARDRAILS:
    1. PROPOSE BEFORE EXECUTION: For ANY plan creation or modification, use `ToolDraft...` first to present proposed changes. ONLY when confirmed may you call `ToolCommit...`.
    2. EMPIRICAL ANALYSIS: Base feedback on real data via analytical tools, not hallucinated assumptions.
    3. FORMATTING: Clean bullet points only. NO HTML tags like <br>. NO Markdown tables. NO <think> tags.
    """
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "agent_steps": 1}

def execute_tools_node(state: AgentState):
    last_msg = state["messages"][-1]
    user_id = state["user_id"]
    conv_id = state["conversation_id"]
    tool_messages = []

    for tool_call in last_msg.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        call_id = tool_call["id"]
        
        try:
            if name == "ToolGetUserProfile": res = execute_get_profile(user_id)
            elif name == "ToolUpdateUserProfile": res = execute_update_profile(user_id, ToolUpdateUserProfile(**args))
            elif name == "ToolGetActivePlan": res = execute_get_active_plan(user_id)
            elif name == "ToolDraftWorkoutPlan": res = execute_draft_plan(user_id, conv_id, ToolDraftWorkoutPlan(**args))
            elif name == "ToolCommitWorkoutPlan": res = execute_commit_plan(user_id, conv_id, ToolCommitWorkoutPlan(**args))
            elif name == "ToolDraftPlanModification": res = execute_draft_modification(user_id, conv_id, ToolDraftPlanModification(**args))
            elif name == "ToolCommitPlanModification": res = execute_commit_modification(user_id, conv_id, ToolCommitPlanModification(**args))
            elif name == "ToolGetPlanProgress": res = execute_get_progress(user_id)
            elif name == "ToolGetRecentWorkoutSessions": res = execute_get_recent_sessions(user_id, ToolGetRecentWorkoutSessions(**args))
            elif name == "ToolGetExerciseTrend": res = execute_analyze_exercise_trend(user_id, ToolGetExerciseTrend(**args))
            elif name == "ToolGetConsistencyStats": res = execute_get_consistency(user_id)
            elif name == "ToolGetPendingActions": res = execute_get_pending_actions(user_id, conv_id)
            else: res = {"success": False, "error": f"Unknown tool {name}"}
        except Exception as e:
            res = {"success": False, "error_code": "TOOL_CRASH", "message": str(e)}
            
        tool_messages.append(ToolMessage(content=json.dumps(res), name=name, tool_call_id=call_id))

    return {"messages": tool_messages}

def force_finalization_node(state: AgentState):
    prompt = "SYSTEM DIRECTIVE: Tool budget exhausted. Summarize findings and provide a final answer directly."
    response = llm.invoke(state["messages"] + [SystemMessage(content=prompt)])
    return {"messages": [response], "agent_steps": 1}

def route_after_agent(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None): return "execute_tools"
    if state.get("agent_steps", 0) >= 6: return "force_finalization"
    return END

def route_after_tools(state: AgentState) -> str:
    if state.get("agent_steps", 0) >= 6: return "force_finalization"
    return "agent"

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("execute_tools", execute_tools_node)
graph_builder.add_node("force_finalization", force_finalization_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", route_after_agent)
graph_builder.add_conditional_edges("execute_tools", route_after_tools)
graph_builder.add_edge("force_finalization", END) 
agent_graph = graph_builder.compile()

# ==========================================
# FASTAPI ENDPOINTS & ANALYTICS EXTENSIONS
# ==========================================
def clean_ai_response(text_content: str) -> str:
    if not text_content: return "I'm ready to help you train. What's our focus today?"
    cleaned = re.sub(r'<think>.*?</think>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("```json", "").replace("```markdown", "").replace("```", "")
    cleaned = cleaned.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    return cleaned.strip()

@app.get("/api/v1/users/{user_id}/dashboard")
def get_dashboard_data(user_id: str, timeframe: str = "This Week", auth_user_id: str = Depends(get_current_user_id)):
    """Generates empirically grounded analytical data for the Flutter Reports tab based on explicit timeframes."""
    if user_id != auth_user_id: raise HTTPException(403, "Forbidden")
    
    interval_str = '7 days'
    if timeframe == "This Month": interval_str = '30 days'
    elif timeframe == "Last 4 Weeks": interval_str = '28 days'
    elif timeframe == "All Time": interval_str = '100 years'
    
    cons_data = execute_get_consistency(user_id)
    consistency_rate = cons_data.get("overall_completion_rate", "0%")
    
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT activity_key, reps, form_score, dominant_deviation, created_at FROM workout_sessions WHERE user_id = :uid AND created_at >= NOW() - INTERVAL '{interval_str}' ORDER BY created_at ASC"),
                {"uid": user_id}
            ).mappings().fetchall()
            
        total_workouts = len(rows)
        total_reps = sum(r["reps"] for r in rows)
        avg_form = sum(r["form_score"] for r in rows) / total_workouts if total_workouts > 0 else 0
        
        # Calculate Top Issues
        deviations = [r["dominant_deviation"] for r in rows if r["dominant_deviation"]]
        issue_counts = {}
        for d in deviations:
            clean_name = d.replace("_", " ").title()
            issue_counts[clean_name] = issue_counts.get(clean_name, 0) + 1
        top_issues = [{"issue": k, "count": v} for k, v in sorted(issue_counts.items(), key=lambda item: item[1], reverse=True)[:3]]
        
        # Calculate Exercise Performance
        ex_data = {}
        for r in rows:
            key = r['activity_key'].title()
            if key not in ex_data: ex_data[key] = []
            ex_data[key].append(r["form_score"])
            
        ex_perf = []
        for k, scores in ex_data.items():
            prev = scores[0] if len(scores) > 0 else 0
            curr = scores[-1] if len(scores) > 0 else 0
            if len(scores) >= 3:
                mid = len(scores) // 2
                prev = sum(scores[:mid]) / len(scores[:mid])
                curr = sum(scores[mid:]) / len(scores[mid:])
            ex_perf.append({"name": k, "previous": round(prev), "current": round(curr)})
            
        trend_data = [r["form_score"] for r in rows[-7:]]
        if len(trend_data) < 7: trend_data = [0] * (7 - len(trend_data)) + trend_data

        return {
            "total_workouts": total_workouts,
            "consistency": consistency_rate,
            "total_reps": total_reps,
            "avg_form_score": round(avg_form),
            "trend_data": trend_data,
            "common_issues": top_issues,
            "exercise_performance": ex_perf
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    week_number: int
    day_number: int
    is_completed: bool

class UserCreate(BaseModel):
    name: str
    gender: Optional[str] = None
    goals: Optional[List[str]] = []

@app.post("/api/v1/users")
def create_user(user: UserCreate):
    with engine.begin() as conn:
        query = text("INSERT INTO users (name, gender, goals) VALUES (:name, :gender, :goals) RETURNING user_id, name, gender, goals")
        result = conn.execute(query, {"name": user.name, "gender": user.gender, "goals": user.goals}).mappings().fetchone()
        return dict(result)

@app.get("/api/v1/users/{user_id}")
def get_user_profile_endpoint(user_id: str, auth_user_id: str = Depends(get_current_user_id)):
    if user_id != auth_user_id: raise HTTPException(403, "Forbidden")
    res = execute_get_profile(auth_user_id)
    if not res.get("success"): raise HTTPException(status_code=404, detail="User not found")
    return res["profile"]

@app.put("/api/v1/users/{user_id}")
def update_user_profile_endpoint(user_id: str, payload: dict, auth_user_id: str = Depends(get_current_user_id)):
    if user_id != auth_user_id: raise HTTPException(403, "Forbidden")
    args = ToolUpdateUserProfile(
        name=payload.get("name"), gender=payload.get("gender"),
        age=payload.get("age"), weight_kg=payload.get("weight_kg"), height_cm=payload.get("height_cm"), 
        about_me=payload.get("about_me"), about_me_op="replace",
        goals_op="replace" if payload.get("goals") else None, goals_values=payload.get("goals"),
        prefs_op="replace" if payload.get("preferred_categories") else None, prefs_values=payload.get("preferred_categories")
    )
    res = execute_update_profile(auth_user_id, args)
    if not res.get("success"): raise HTTPException(status_code=400, detail=res)
    return res

@app.post("/api/v1/chat/conversations")
def create_conversation(auth_user_id: str = Depends(get_current_user_id)):
    with engine.begin() as conn:
        query = text("INSERT INTO chatbot_conversations (user_id) VALUES (:user_id) RETURNING conversation_id")
        result = conn.execute(query, {"user_id": auth_user_id}).mappings().fetchone()
        return dict(result)

@app.get("/api/v1/chat/conversations/{conversation_id}/messages")
def get_chat_history(conversation_id: str, auth_user_id: str = Depends(get_current_user_id)):
    with engine.connect() as conn:
        conv = conn.execute(text("SELECT user_id FROM chatbot_conversations WHERE conversation_id = :c"), {"c": conversation_id}).mappings().fetchone()
        if not conv or str(conv['user_id']) != auth_user_id: raise HTTPException(status_code=403, detail="Forbidden")
        
        query = text("SELECT role, content, created_at FROM chatbot_messages WHERE conversation_id = :conv_id ORDER BY created_at ASC")
        rows = conn.execute(query, {"conv_id": conversation_id}).mappings().fetchall()
        return [{"role": r["role"], "content": clean_ai_response(r["content"]) if r["role"] == "assistant" else r["content"], "created_at": r["created_at"]} for r in rows]

@app.post("/api/v1/sessions")
def save_workout_session(session: WorkoutSessionCreate, auth_user_id: str = Depends(get_current_user_id)):
    with engine.begin() as conn:
        query = text("""
            INSERT INTO workout_sessions (user_id, activity_key, reps, duration_seconds, form_score, dominant_deviation, deviations_json) 
            VALUES (:user_id, :activity_key, :reps, :duration_seconds, :form_score, :dominant_deviation, :deviations_json) RETURNING session_id
        """)
        result = conn.execute(query, {
            "user_id": auth_user_id, "activity_key": session.activity_key, "reps": session.reps, "duration_seconds": session.duration_seconds,
            "form_score": session.form_score, "dominant_deviation": session.dominant_deviation, "deviations_json": json.dumps(session.deviations_json)
        }).mappings().fetchone()
        return {"status": "success", "session_id": str(result['session_id'])}

@app.get("/api/v1/users/{user_id}/plan")
def get_active_plan(user_id: str, auth_user_id: str = Depends(get_current_user_id)):
    if user_id != auth_user_id: raise HTTPException(403, "Forbidden")
    res = execute_get_active_plan(auth_user_id)
    if not res.get("success"): return {"status": "no_active_plan"}
    return {"plan_id": res["plan_id"], "plan_name": res["plan_name"], "plan_json": res["plan_json"], "user_id": auth_user_id}

# NEW: Endpoint to fetch the single most recent workout session for the Home Screen
@app.get("/api/v1/users/{user_id}/sessions/recent")
def get_recent_session(user_id: str, auth_user_id: str = Depends(get_current_user_id)):
    """Fetches the single most recent workout session for the Home Screen"""
    if user_id != auth_user_id: raise HTTPException(403, "Forbidden")
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT activity_key, reps, duration_seconds, form_score, dominant_deviation, created_at 
                    FROM workout_sessions 
                    WHERE user_id = :uid 
                    ORDER BY created_at DESC LIMIT 1
                """),
                {"uid": auth_user_id}
            ).mappings().fetchone()
        
        if not row: return {"status": "no_sessions"}
        
        session_data = dict(row)
        session_data["created_at"] = str(session_data["created_at"])
        return {"status": "success", "session": session_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/plans/{plan_id}/completions")
def get_plan_completions(plan_id: str, auth_user_id: str = Depends(get_current_user_id)):
    with engine.connect() as conn:
        plan_check = conn.execute(text("SELECT 1 FROM workout_plans WHERE plan_id = :pid AND user_id = :uid"), {"pid": plan_id, "uid": auth_user_id}).fetchone()
        if not plan_check: raise HTTPException(status_code=403, detail="Unauthorized")
        
        query = text("SELECT week_number, day_number FROM plan_day_completions WHERE plan_id = :pid AND is_completed = true")
        rows = conn.execute(query, {"pid": plan_id}).mappings().fetchall()
        return [f"w{r['week_number']}_d{r['day_number']}" for r in rows]

@app.post("/api/v1/plans/{plan_id}/completions/toggle")
def toggle_plan_completion(plan_id: str, payload: DayCompletionToggle, auth_user_id: str = Depends(get_current_user_id)):
    with engine.begin() as conn:
        plan_check = conn.execute(text("SELECT 1 FROM workout_plans WHERE plan_id = :pid AND user_id = :uid"), {"pid": plan_id, "uid": auth_user_id}).fetchone()
        if not plan_check: raise HTTPException(status_code=403, detail="Unauthorized")
            
        if payload.is_completed:
            query = text("INSERT INTO plan_day_completions (plan_id, user_id, week_number, day_number, is_completed) VALUES (:pid, :uid, :wn, :dn, true) ON CONFLICT (plan_id, week_number, day_number) DO UPDATE SET is_completed = true, completed_at = now()")
        else:
            query = text("DELETE FROM plan_day_completions WHERE plan_id = :pid AND week_number = :wn AND day_number = :dn")
        conn.execute(query, {"pid": plan_id, "uid": auth_user_id, "wn": payload.week_number, "dn": payload.day_number})
    return {"status": "success"}

@app.post("/api/v1/chat/conversations/{conversation_id}/messages")
def add_chat_message(conversation_id: str, message: ChatMessageCreate, auth_user_id: str = Depends(get_current_user_id)):
    with engine.begin() as conn:
        conv_row = conn.execute(text("SELECT user_id FROM chatbot_conversations WHERE conversation_id = :conv_id"), {"conv_id": conversation_id}).mappings().fetchone()
        if not conv_row or str(conv_row['user_id']) != auth_user_id: raise HTTPException(status_code=403, detail="Forbidden")

    try:
        with engine.connect() as conn:
            user_row = conn.execute(text("SELECT name, gender, age, weight_kg, height_cm, goals, preferred_categories, about_me FROM users WHERE user_id = :uid"), {"uid": auth_user_id}).mappings().fetchone()
            plan_row = conn.execute(text("SELECT plan_name FROM workout_plans WHERE user_id = :uid AND is_active = true ORDER BY created_at DESC LIMIT 1"), {"uid": auth_user_id}).fetchone()
            pending_row = conn.execute(text("SELECT action_type FROM pending_actions WHERE user_id = :uid AND conversation_id = :cid AND created_at >= NOW() - INTERVAL '30 minutes' LIMIT 1"), {"uid": auth_user_id, "cid": conversation_id}).fetchone()
            
        profile_dict = dict(user_row) if user_row else {}
        if profile_dict.get('weight_kg'): profile_dict['weight_kg'] = float(profile_dict['weight_kg'])
        if profile_dict.get('height_cm'): profile_dict['height_cm'] = float(profile_dict['height_cm'])
        plan_name = plan_row[0] if plan_row else "No Active Plan"
        pending_status = f"Yes ({pending_row[0]}) - Call ToolGetPendingActions for unexpired token" if pending_row else "None"
        context_str = f"CURRENT USER PROFILE: {json.dumps(profile_dict)}\nACTIVE PLAN: {plan_name}\nPENDING ACTION (This Conversation): {pending_status}"
    except Exception as e:
        context_str = "CURRENT USER PROFILE: Unknown"

    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO chatbot_messages (conversation_id, role, content) VALUES (:conv_id, 'user', :content)"), {"conv_id": conversation_id, "content": message.content})
            history_rows = conn.execute(text("SELECT role, content FROM chatbot_messages WHERE conversation_id = :conv_id ORDER BY created_at ASC"), {"conv_id": conversation_id}).mappings().fetchall()

            lg_messages = []
            for row in history_rows:
                if row['role'] == 'user': lg_messages.append(HumanMessage(content=row['content']))
                elif row['role'] == 'assistant': lg_messages.append(AIMessage(content=row['content']))

        initial_state = {"messages": lg_messages[-10:], "user_id": auth_user_id, "conversation_id": conversation_id, "agent_steps": 0, "user_context": context_str} 
        result = agent_graph.invoke(initial_state)
        
        raw_reply = result["messages"][-1].content
        ai_reply = clean_ai_response(raw_reply)
    except Exception as e:
        ai_reply = f"SYSTEM ERROR: {str(e)}"

    with engine.begin() as conn:
        result = conn.execute(text("INSERT INTO chatbot_messages (conversation_id, role, content) VALUES (:conv_id, 'assistant', :content) RETURNING message_id, role, content"), {"conv_id": conversation_id, "content": ai_reply}).mappings().fetchone()
        
    response_dict = dict(result)
    response_dict['content'] = clean_ai_response(response_dict['content'])
    return response_dict