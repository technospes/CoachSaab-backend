import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://postgres:mysecretpassword@localhost:5433/aicoach"

async def seed_rules():
    engine = create_async_engine(DATABASE_URL, echo=True)
    
    async with engine.begin() as conn:
        print("Seeding Squat Rules and Feedback...")
        
        # 1. Insert a Squat Depth Rule[cite: 4]
        await conn.execute(text("""
            INSERT INTO activity_rules 
            (activity_config_id, joint, phase, metric, target_min, target_max, deviation_type, severity_thresholds)
            SELECT activity_config_id, 'knee', 'bottom', 'angle', 85, 105, 'insufficient_depth', '{"minor": 5, "moderate": 10, "major": 20}'::jsonb
            FROM activity_configs WHERE activity_key = 'squat'
            AND NOT EXISTS (
                SELECT 1 FROM activity_rules WHERE deviation_type = 'insufficient_depth'
            );
        """))

        # 2. Insert the Corresponding TTS Feedback[cite: 4]
        await conn.execute(text("""
            INSERT INTO activity_feedback_map 
            (activity_config_id, deviation_type, locale, display_text, tts_text)
            SELECT activity_config_id, 'insufficient_depth', 'en', 'Go slightly deeper', 'Go slightly deeper'
            FROM activity_configs WHERE activity_key = 'squat'
            ON CONFLICT DO NOTHING;
        """))

        print("Rules seeded successfully!")

if __name__ == "__main__":
    asyncio.run(seed_rules())