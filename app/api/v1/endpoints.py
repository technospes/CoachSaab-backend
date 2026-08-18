from fastapi import APIRouter, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.core.database import AsyncSessionLocal

router = APIRouter()

@router.get("/activity-configs")
async def get_activity_configs(category: str = None):
    """
    Fetches all enabled activity configurations, optionally filtered by category.
    Matches PRD §5.1 & Backend Schema §6.1.
    """
    async with AsyncSessionLocal() as session:
        query = "SELECT * FROM activity_configs WHERE is_enabled = true"
        params = {}
        if category:
            query += " AND category = :cat"
            params["cat"] = category
            
        result = await session.execute(text(query), params)
        rows = result.mappings().all()
        return {"activities": [dict(row) for row in rows]}

@router.get("/activity-configs/{activity_key}")
async def get_activity_config_bundle(activity_key: str):
    """
    Returns the full config bundle including rules and feedback map for a specific activity.
    Matches Backend Schema §6.2.
    """
    async with AsyncSessionLocal() as session:
        # 1. Fetch Config
        config_res = await session.execute(
            text("SELECT * FROM activity_configs WHERE activity_key = :key"),
            {"key": activity_key}
        )
        config = config_res.mappings().first()
        if not config:
            raise HTTPException(status_code=404, detail="Activity not found")
        
        config_id = config["activity_config_id"]

        # 2. Fetch Rules
        rules_res = await session.execute(
            text("SELECT * FROM activity_rules WHERE activity_config_id = :id AND is_enabled = true"),
            {"id": config_id}
        )
        rules = [dict(row) for row in rules_res.mappings().all()]

        # 3. Fetch Feedback Map
        feedback_res = await session.execute(
            text("SELECT * FROM activity_feedback_map WHERE activity_config_id = :id"),
            {"id": config_id}
        )
        feedback = [dict(row) for row in feedback_res.mappings().all()]

        return {
            "config": dict(config),
            "rules": rules,
            "feedback_map": feedback
        }