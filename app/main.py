from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import create_engine, text
from typing import Optional, List, Annotated, Literal, Union
import os
import json
import re
import uuid
import hashlib
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
class ScheduledExercise(BaseModel):
    exercise_id: str
    name: str

class DailyWorkout(BaseModel):
    day_number: int = Field(..., description="Day of the week (1 to 7)")
    focus: str = Field(..., description="Workout focus (e.g., 'UPPER BODY', 'LOWER BODY', 'ACTIVE RECOVERY', 'REST DAY')")
    exercises: List[ScheduledExercise] = Field(..., description="List of exercises, each with a stable exercise_id and a name string that includes sets/reps/time. Empty if rest day.")
    is_rest: bool = Field(..., description="True if this is a rest day")

class WorkoutPlanSchema(BaseModel):
    duration_weeks: int = Field(..., gt=0, le=12, description="Duration in weeks. Keep it short (2 to 4 weeks max) for quick user milestones, unless the user explicitly requested a longer plan.")
    goal: str = Field(..., description="Main objective of the plan")
    notes: str = Field(..., description="Additional coaching advice and progression instructions")
    schedule: List[DailyWorkout] = Field(..., description="A 7-day schedule template (Days 1-7)")

# ------------------------------------------
# v6.1: Internal schema the LLM actually fills in for NEW plans.
# The LLM still emits plain exercise name strings (no IDs) - the
# backend is solely responsible for minting exercise_ids afterwards.
# ------------------------------------------
class DailyWorkoutDraft(BaseModel):
    day_number: int
    focus: str
    exercises: List[str] = Field(default_factory=list)
    is_rest: bool

class WorkoutPlanDraftSchema(BaseModel):
    duration_weeks: int = Field(..., gt=0, le=12)
    goal: str
    notes: str
    schedule: List[DailyWorkoutDraft]

# ==========================================
# v6.1: STABLE EXERCISE ID HELPERS
# ==========================================
def generate_exercise_id() -> str:
    """New exercises always get a fresh random ID. The LLM never sees/sets this."""
    return uuid.uuid4().hex[:12]

def generate_legacy_exercise_id(plan_id: str, day_number: int, exercise_index: int) -> str:
    """
    Deterministic ID for legacy string-only exercises, derived from stable
    plan/day/index info. Same inputs -> same ID every time, so legacy plans
    don't get a new random ID minted on every load.
    """
    raw = f"{plan_id}:{day_number}:{exercise_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

def assign_ids_to_draft_plan(draft: WorkoutPlanDraftSchema) -> dict:
    """
    Converts a freshly-LLM-generated draft plan (plain exercise name strings)
    into the final stored object format, minting a new exercise_id for every
    exercise. The backend does this - never the LLM.
    """
    schedule = []
    for day in draft.schedule:
        exercises = [
            {"exercise_id": generate_exercise_id(), "name": name}
            for name in day.exercises
        ]
        schedule.append({
            "day_number": day.day_number,
            "focus": day.focus,
            "exercises": exercises,
            "is_rest": day.is_rest,
        })
    return {
        "duration_weeks": draft.duration_weeks,
        "goal": draft.goal,
        "notes": draft.notes,
        "schedule": schedule,
    }

def normalize_plan_json(plan_json: dict, plan_id: str) -> tuple[dict, bool]:
    """
    Ensures every exercise in every day of plan_json is in the
    {"exercise_id": ..., "name": ...} object format.

    Legacy plans store exercises as plain strings - these get a
    deterministic ID derived from (plan_id, day_number, exercise_index)
    so the ID is stable across repeated loads.

    Returns (normalized_plan_json, changed) where `changed` indicates
    whether any legacy strings were converted (so callers can decide
    whether to persist the normalized form back to the DB).
    """
    if not plan_json or "schedule" not in plan_json:
        return plan_json, False

    changed = False
    for day in plan_json.get("schedule", []):
        day_number = day.get("day_number")
        raw_exercises = day.get("exercises", []) or []
        normalized_exercises = []
        for idx, ex in enumerate(raw_exercises):
            if isinstance(ex, str):
                normalized_exercises.append({
                    "exercise_id": generate_legacy_exercise_id(plan_id, day_number, idx),
                    "name": ex,
                })
                changed = True
            elif isinstance(ex, dict) and "exercise_id" in ex and "name" in ex:
                normalized_exercises.append(ex)
            else:
                # Unrecognized shape - skip rather than silently corrupt data.
                continue
        day["exercises"] = normalized_exercises

    return plan_json, changed


# ==========================================
# v6.1: PLAN MODIFICATION SCHEMAS
# ==========================================
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
    new_name: str  # full replacement string incl. updated sets/reps/time, same exercise_id preserved

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
    AddExerciseOp,
    RemoveExerciseOp,
    ReplaceExerciseOp,
    ModifyExerciseOp,
    ChangeDayFocusOp,
    RestDayOp,
    ChangeDurationOp,
    RemoveExercisesByIdsOp,
]

class ModificationRequest(BaseModel):
    """
    LLM extraction output. Exactly one of `operations` / `clarification_question`
    must be present - never both, never neither.
    """
    operations: Optional[List[ModificationOperation]] = None
    clarification_question: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one_branch(self):
        has_ops = bool(self.operations)
        has_clarification = bool(self.clarification_question)
        if has_ops == has_clarification:
            raise ValueError(
                "ModificationRequest must have EITHER operations OR clarification_question, not both/neither."
            )
        return self

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
    intent: str  # "chat" | "onboarding" | "modification"
    # v6.1: plan-modification fields
    active_plan: Optional[dict]          # {"plan_id": ..., "plan_json": {...}} or None
    modification_result: Optional[dict]  # {"operations": [...]} | {"clarification_question": "..."}
    modification_error: Optional[str]

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

# v6.1: Modification-intent keyword sets. These only decide ROUTING
# (which node handles the message), never which exercise gets touched -
# exercise-level decisions are always deterministic ID validation
# downstream, resolved semantically by the LLM in extract_modification_node.
MODIFICATION_VERBS = [
    "remove", "delete", "replace", "swap", "substitute",
    "add", "modify", "update", "change",
]
PLAN_REFERENCE_TERMS = [
    "plan", "exercise", "workout", "routine", "schedule",
    "sets", "reps", "rest day", "duration",
]
INJURY_TERMS = [
    "injury", "injured", "hurt", "pain", "sore",
    "can't train", "cannot train", "unable to train",
    "don't have access to", "only have",
]

def _mentions_day_or_week_reference(text_lower: str) -> bool:
    """Matches things like 'day 1', 'day2', 'week 3' etc."""
    return bool(re.search(r"\b(day|week)\s*\d+\b", text_lower))

def load_active_plan_node(state: AgentState):
    """
    Loads the user's currently active plan (if any) and normalizes legacy
    string-exercise entries into stable-ID object form. Runs before intent
    routing so modification-intent can be gated on "does an active plan exist".
    """
    user_id = state["user_id"]
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT plan_id, plan_json
                FROM workout_plans
                WHERE user_id = :uid AND is_active = true
                ORDER BY created_at DESC LIMIT 1
            """),
            {"uid": user_id}
        ).mappings().fetchone()

    if not row:
        return {"active_plan": None}

    plan_id = str(row["plan_id"])
    plan_json = row["plan_json"]
    if isinstance(plan_json, str):
        plan_json = json.loads(plan_json)

    normalized_json, changed = normalize_plan_json(plan_json, plan_id)

    if changed:
        # Persist the normalized (now-object-format) legacy plan back once,
        # so future loads read the object format directly. Safe: we only
        # rewrite the exercises field shape, nothing else in plan_json.
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE workout_plans SET plan_json = :data WHERE plan_id = :pid"),
                    {"data": json.dumps(normalized_json), "pid": plan_id}
                )
        except Exception as e:
            print(f"Legacy plan normalization persist error: {str(e)}")
            # Non-fatal: normalized_json is still used in-memory for this request.

    return {"active_plan": {"plan_id": plan_id, "plan_json": normalized_json}}

def determine_intent_node(state: AgentState):
    """DETERMINISTIC ROUTER: No LLM used here. Checks chat history/message for intent."""
    latest_user_msg = state["messages"][-1].content.lower()
    has_active_plan = bool(state.get("active_plan"))

    # 1. Did the AI just ask an onboarding question?
    ai_asked_question = False
    if len(state["messages"]) >= 2:
        last_ai_msg = state["messages"][-2].content
        if any(q in last_ai_msg for q in QUESTIONS.values()):
            ai_asked_question = True

    # 2. Did the user explicitly trigger a plan request?
    plan_triggers = ["plan", "routine", "program", "schedule", "workout"]
    user_wants_plan = any(t in latest_user_msg for t in plan_triggers)

    # 3. v6.1 Modification intent (conservative, deterministic):
    #    Only selectable when an active plan actually exists.
    if has_active_plan:
        has_mod_verb = any(v in latest_user_msg for v in MODIFICATION_VERBS)
        has_plan_ref = (
            any(t in latest_user_msg for t in PLAN_REFERENCE_TERMS)
            or _mentions_day_or_week_reference(latest_user_msg)
        )
        has_injury_language = any(t in latest_user_msg for t in INJURY_TERMS)

        # verb + plan-reference (e.g. "remove squats from my plan") -> modification
        # bare injury statement (e.g. "my knee hurts") -> modification, so the
        #   extraction node can ask what they'd like changed
        if (has_mod_verb and has_plan_ref) or has_injury_language:
            return {"intent": "modification"}

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
    """Generates the plan and uses robust JSON parsing to bypass <think> tags.

    v6.1: The LLM still emits plain exercise name strings (WorkoutPlanDraftSchema).
    The backend - not the LLM - assigns a stable exercise_id to every exercise
    before the plan is validated/persisted in the new object format.
    """
    try:
        profile_str = json.dumps(state["user_profile"])
        schema_json = WorkoutPlanDraftSchema.model_json_schema()
        
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
             
        draft_plan = WorkoutPlanDraftSchema.model_validate(extracted_json)

        # Backend assigns exercise_ids and produces the final object-format plan.
        db_json = assign_ids_to_draft_plan(draft_plan)
        db_json["status"] = "active"
        
        user_id = state["user_id"]
        plan_name = f"{state['user_profile'].get('name', 'My')} {draft_plan.duration_weeks}-Week Plan"
        
        with engine.begin() as conn:
            conn.execute(text("UPDATE workout_plans SET is_active = false WHERE user_id = :uid"), {"uid": user_id})
            query = text("""
                INSERT INTO workout_plans (user_id, plan_name, plan_json, is_active)
                VALUES (:uid, :name, :data, true)
            """)
            conn.execute(query, {"uid": user_id, "name": plan_name, "data": json.dumps(db_json)})
            
        msg = AIMessage(content=f"I've successfully generated your custom {draft_plan.duration_weeks}-week '{draft_plan.goal}' plan! Check your Home tab to see the breakdown.")
        return {"messages": [msg]}
    except Exception as e:
        print(f"Plan Gen Error: {str(e)}")
        return {"messages": [AIMessage(content=f"SYSTEM ERROR: {str(e)}")]}

# ==========================================
# v6.1: PLAN MODIFICATION NODES
# ==========================================
def _render_plan_for_llm(plan_json: dict) -> str:
    """Compact human-readable representation of the active plan, WITH exercise_ids,
    so the LLM can resolve natural-language requests to real IDs."""
    lines = []
    for day in plan_json.get("schedule", []):
        lines.append(f"Day {day.get('day_number')} ({day.get('focus')}):")
        if day.get("is_rest") or not day.get("exercises"):
            lines.append("  (rest day / no exercises)")
        for ex in day.get("exercises", []):
            lines.append(f"  [{ex['exercise_id']}] {ex['name']}")
    return "\n".join(lines)

def extract_modification_node(state: AgentState):
    """
    LLM = semantic extraction ONLY. It sees the active plan (with real
    exercise_ids) and the user's natural-language request, and must return
    either a list of operations referencing ONLY IDs shown in the plan, or
    a clarification question if the request is ambiguous. It never invents
    IDs and never decides medical/safety questions.
    """
    try:
        active_plan = state.get("active_plan")
        if not active_plan:
            # Should not normally happen (router gates on active_plan existing),
            # but guard defensively.
            return {"modification_error": "no_active_plan"}

        plan_repr = _render_plan_for_llm(active_plan["plan_json"])
        latest_msg = state["messages"][-1].content
        schema_json = ModificationRequest.model_json_schema()

        sys_prompt = f"""You are a precise workout-plan modification extractor.

ACTIVE PLAN (the ONLY valid exercise_ids and day_numbers you may reference):
{plan_repr}

USER REQUEST:
"{latest_msg}"

RULES:
1. Return ONLY a JSON object matching this schema: {json.dumps(schema_json)}
2. You MUST use `operations` OR `clarification_question`, never both.
3. NEVER invent an exercise_id. Only use IDs shown above, exactly as written.
4. For remove/replace/modify operations, reference the exercise by its exercise_id, not by name.
5. If the request is ambiguous (e.g. could match more than one exercise, or you cannot tell which day/exercise is meant), you MUST return a `clarification_question` instead of guessing.
6. Do NOT diagnose any injury or decide if an exercise is medically safe. Only extract what the user explicitly asked to change and why (if they gave a reason).
7. For add_exercise, you do not need to invent an exercise_id - the backend assigns it.
8. Do not include <think> tags in the final JSON.
"""

        response = llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content="Extract the modification. Output pure JSON.")
        ])

        extracted_json = extract_json_from_text(response.content)
        if not extracted_json:
            return {"modification_error": "extraction_failed"}

        mod_request = ModificationRequest.model_validate(extracted_json)
        return {"modification_result": mod_request.model_dump(exclude_none=True), "modification_error": None}
    except Exception as e:
        print(f"Modification Extraction Error: {str(e)}")
        return {"modification_error": "extraction_failed"}

def _validate_operations(operations: list, plan_json: dict) -> Optional[str]:
    """
    Deterministic validation. Returns an error string if ANY operation is
    invalid (bad exercise_id, bad day_number, bad duration), else None.
    ALL-valid -> apply ALL. ANY-invalid -> apply NONE.
    """
    valid_ids = {
        ex["exercise_id"]
        for day in plan_json.get("schedule", [])
        for ex in day.get("exercises", [])
    }
    valid_days = {day.get("day_number") for day in plan_json.get("schedule", [])}

    for op in operations:
        op_type = op.get("type")

        if op_type == "add_exercise":
            if op.get("day_number") not in valid_days:
                return f"Invalid day_number in add_exercise: {op.get('day_number')}"
            if not op.get("name"):
                return "add_exercise missing exercise name"

        elif op_type in ("remove_exercise", "replace_exercise", "modify_exercise"):
            if op.get("exercise_id") not in valid_ids:
                return f"Unknown exercise_id: {op.get('exercise_id')}"
            if op_type in ("replace_exercise", "modify_exercise") and not op.get("new_name"):
                return f"{op_type} missing new_name"

        elif op_type == "remove_exercises_by_ids":
            ids = op.get("exercise_ids") or []
            if not ids:
                return "remove_exercises_by_ids missing exercise_ids"
            for eid in ids:
                if eid not in valid_ids:
                    return f"Unknown exercise_id: {eid}"

        elif op_type == "change_day_focus":
            if op.get("day_number") not in valid_days:
                return f"Invalid day_number in change_day_focus: {op.get('day_number')}"
            if not op.get("new_focus"):
                return "change_day_focus missing new_focus"

        elif op_type == "rest_day":
            if op.get("day_number") not in valid_days:
                return f"Invalid day_number in rest_day: {op.get('day_number')}"

        elif op_type == "change_duration":
            weeks = op.get("new_duration_weeks")
            if not isinstance(weeks, int) or weeks <= 0 or weeks > 12:
                return f"Invalid new_duration_weeks: {weeks}"

        else:
            return f"Unknown operation type: {op_type}"

    return None

def _apply_operations(operations: list, plan_json: dict) -> tuple[dict, set]:
    """
    Deterministically applies already-validated operations to plan_json.
    Returns (new_plan_json, affected_day_numbers) where affected_day_numbers
    is the set of day_numbers whose exercise LIST changed (and therefore
    need their checklist completion reset). change_day_focus and
    change_duration do NOT add to affected_day_numbers.
    """
    days_by_number = {day["day_number"]: day for day in plan_json.get("schedule", [])}
    affected_days = set()

    for op in operations:
        op_type = op["type"]

        if op_type == "add_exercise":
            day = days_by_number[op["day_number"]]
            day.setdefault("exercises", []).append({
                "exercise_id": generate_exercise_id(),
                "name": op["name"],
            })
            day["is_rest"] = False
            affected_days.add(op["day_number"])

        elif op_type == "remove_exercise":
            target_id = op["exercise_id"]
            for day in plan_json.get("schedule", []):
                before = len(day.get("exercises", []))
                day["exercises"] = [ex for ex in day.get("exercises", []) if ex["exercise_id"] != target_id]
                if len(day["exercises"]) != before:
                    affected_days.add(day["day_number"])

        elif op_type == "remove_exercises_by_ids":
            target_ids = set(op["exercise_ids"])
            for day in plan_json.get("schedule", []):
                before = len(day.get("exercises", []))
                day["exercises"] = [ex for ex in day.get("exercises", []) if ex["exercise_id"] not in target_ids]
                if len(day["exercises"]) != before:
                    affected_days.add(day["day_number"])

        elif op_type == "replace_exercise":
            target_id = op["exercise_id"]
            for day in plan_json.get("schedule", []):
                for ex in day.get("exercises", []):
                    if ex["exercise_id"] == target_id:
                        ex["name"] = op["new_name"]  # exercise_id preserved
                        affected_days.add(day["day_number"])

        elif op_type == "modify_exercise":
            target_id = op["exercise_id"]
            for day in plan_json.get("schedule", []):
                for ex in day.get("exercises", []):
                    if ex["exercise_id"] == target_id:
                        ex["name"] = op["new_name"]  # exercise_id preserved
                        affected_days.add(day["day_number"])

        elif op_type == "rest_day":
            day = days_by_number[op["day_number"]]
            day["exercises"] = []
            day["is_rest"] = True
            day["focus"] = "REST DAY"
            affected_days.add(op["day_number"])

        elif op_type == "change_day_focus":
            day = days_by_number[op["day_number"]]
            day["focus"] = op["new_focus"]
            # focus-only change does NOT reset completion

        elif op_type == "change_duration":
            plan_json["duration_weeks"] = op["new_duration_weeks"]
            # duration-only change does NOT reset completion

    return plan_json, affected_days

def apply_modification_node(state: AgentState):
    """
    Deterministically validates then applies the extracted operations.
    ALL-or-nothing: if any operation is invalid, NOTHING is modified.
    Plan update + affected-day completion cleanup happen in ONE atomic
    DB transaction (engine.begin()) using the SAME plan_id (no new plan
    row is ever created for a modification).
    """
    try:
        active_plan = state.get("active_plan")
        mod_result = state.get("modification_result") or {}
        operations = mod_result.get("operations")

        if not active_plan or not operations:
            return {"messages": [AIMessage(content="I couldn't find an active plan to modify.")]}

        plan_id = active_plan["plan_id"]
        user_id = state["user_id"]

        # SECURITY: re-verify ownership of this exact plan_id right before writing,
        # never trust an ID that wasn't loaded for this authenticated user.
        with engine.connect() as conn:
            owner_check = conn.execute(
                text("SELECT 1 FROM workout_plans WHERE plan_id = :pid AND user_id = :uid AND is_active = true"),
                {"pid": plan_id, "uid": user_id}
            ).fetchone()
        if not owner_check:
            return {"messages": [AIMessage(content="I couldn't verify that plan belongs to you, so no changes were made.")]}

        plan_json = json.loads(json.dumps(active_plan["plan_json"]))  # deep copy, don't mutate state in place

        validation_error = _validate_operations(operations, plan_json)
        if validation_error:
            print(f"Modification validation failed, applying NONE: {validation_error}")
            return {"messages": [AIMessage(content="I couldn't safely apply that change (it referenced something that doesn't match your current plan), so nothing was modified. Could you clarify what you'd like changed?")]}

        new_plan_json, affected_days = _apply_operations(operations, plan_json)

        # Single atomic transaction: plan update + affected-day completion cleanup.
        # Either both succeed or neither does - the plan is never left partially
        # modified and checklist records are never reset without a matching plan write.
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE workout_plans SET plan_json = :data WHERE plan_id = :pid AND user_id = :uid"),
                    {"data": json.dumps(new_plan_json), "pid": plan_id, "uid": user_id}
                )
                for day_number in affected_days:
                    conn.execute(
                        text("DELETE FROM plan_day_completions WHERE plan_id = :pid AND day_number = :dn"),
                        {"pid": plan_id, "dn": day_number}
                    )
        except Exception as e:
            print(f"Modification transaction failed, plan left unchanged: {str(e)}")
            return {"messages": [AIMessage(content="Something went wrong while saving your changes, so your plan was left unchanged. Please try again.")]}

        reply = "Done! I've updated your workout plan."
        if affected_days:
            reply += f" Progress for day(s) {', '.join(str(d) for d in sorted(affected_days))} was reset since those exercises changed."

        # Deterministic injury disclaimer - never LLM-generated, never a diagnosis.
        latest_user_msg = state["messages"][-1].content.lower()
        if any(t in latest_user_msg for t in INJURY_TERMS):
            reply += (
                " I can adjust your workout to avoid movements you've identified as uncomfortable. "
                "If the pain is significant or persistent, please consider getting advice from a "
                "qualified healthcare professional."
            )

        return {"messages": [AIMessage(content=reply)]}
    except Exception as e:
        print(f"Apply Modification Error: {str(e)}")
        return {"messages": [AIMessage(content="Something went wrong while modifying your plan, so it was left unchanged. Please try again.")]}

def clarify_modification_node(state: AgentState):
    """Returns the clarification question directly. Does NOT touch the database."""
    mod_result = state.get("modification_result") or {}
    question = mod_result.get("clarification_question")

    if not question:
        # extraction_failed or no_active_plan fallback
        question = "I wasn't able to understand exactly what you'd like changed in your plan. Could you rephrase that?"

    return {"messages": [AIMessage(content=question)]}

# ==========================================
# GRAPH ROUTING
# ==========================================
def route_intent(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "onboarding":
        return "extract_profile"
    if intent == "modification":
        return "extract_modification"
    return "normal_chat"

def route_after_update(state: AgentState) -> str:
    if state.get("missing_fields"):
        return "ask_question"
    return "generate_plan"

def route_after_modification_extraction(state: AgentState) -> str:
    """
    extract_modification -> clarification needed? -> clarify : validate/apply
    Both branches are real, existing nodes and both reach END.
    """
    mod_result = state.get("modification_result") or {}
    if mod_result.get("clarification_question") or state.get("modification_error"):
        return "clarify_modification"
    return "apply_modification"

# Compile the Graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("load_state", load_state_node)
graph_builder.add_node("load_active_plan", load_active_plan_node)
graph_builder.add_node("determine_intent", determine_intent_node)
graph_builder.add_node("extract_profile", extract_profile_node)
graph_builder.add_node("update_database", update_database_node)
graph_builder.add_node("ask_question", ask_question_node)
graph_builder.add_node("generate_plan", generate_plan_node)
graph_builder.add_node("normal_chat", normal_chat_node)
graph_builder.add_node("extract_modification", extract_modification_node)
graph_builder.add_node("clarify_modification", clarify_modification_node)
graph_builder.add_node("apply_modification", apply_modification_node)

graph_builder.add_edge(START, "load_state")
graph_builder.add_edge("load_state", "load_active_plan")
graph_builder.add_edge("load_active_plan", "determine_intent")
graph_builder.add_conditional_edges("determine_intent", route_intent)
graph_builder.add_edge("extract_profile", "update_database")
graph_builder.add_conditional_edges("update_database", route_after_update)
graph_builder.add_edge("ask_question", END)
graph_builder.add_edge("generate_plan", END)
graph_builder.add_edge("normal_chat", END)
graph_builder.add_conditional_edges("extract_modification", route_after_modification_extraction)
graph_builder.add_edge("clarify_modification", END)
graph_builder.add_edge("apply_modification", END)

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
            "intent": "chat",
            "active_plan": None,
            "modification_result": None,
            "modification_error": None,
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