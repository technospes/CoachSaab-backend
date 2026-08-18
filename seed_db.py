import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Using the asyncpg driver with your specific port and password
DATABASE_URL = "postgresql+asyncpg://postgres:mysecretpassword@localhost:5433/aicoach"

async def seed():
    engine = create_async_engine(DATABASE_URL, echo=True)
    
    async with engine.begin() as conn:
        print("Seeding MVP Activity Configs...")
        
        # 1. Active MVP Row: Squat[cite: 4]
        await conn.execute(text("""
            INSERT INTO activity_configs 
            (category, activity_key, display_name, required_landmarks, tracking_mode, hud_config, phases, is_enabled)
            VALUES 
            ('fitness', 'squat', 'Squat', '["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]', 
            'repetition', '{"counter": true, "timer": false, "phase": false}', '["top", "descent", "bottom", "ascent"]', true)
            ON CONFLICT (activity_key) DO NOTHING;
        """))

        # 2. Active MVP Row: Right-Arm Fast Bowling[cite: 4]
        await conn.execute(text("""
            INSERT INTO activity_configs 
            (category, subcategory, discipline, style, activity_key, display_name, required_landmarks, tracking_mode, hud_config, is_enabled)
            VALUES 
            ('sports', 'cricket', 'bowling', 'right_arm_fast', 'cricket_bowling_right_arm_fast', 'Right-Arm Fast Bowling', 
            '["right_shoulder", "right_elbow", "right_wrist", "left_ankle"]', 'event', '{"counter": true, "timer": false, "phase": false}', true)
            ON CONFLICT (activity_key) DO NOTHING;
        """))

        # 3. Disabled Future Row: Cricket Batting (Proving PRD §5.1 extensibility)[cite: 4]
        await conn.execute(text("""
            INSERT INTO activity_configs 
            (category, subcategory, discipline, activity_key, display_name, required_landmarks, tracking_mode, hud_config, is_enabled)
            VALUES 
            ('sports', 'cricket', 'batting', 'cricket_batting', 'Cricket Batting', 
            '["left_shoulder", "right_shoulder", "left_knee", "right_knee"]', 'event', '{"counter": true, "timer": false, "phase": false}', false)
            ON CONFLICT (activity_key) DO NOTHING;
        """))

        print("Database seeded successfully!")

if __name__ == "__main__":
    asyncio.run(seed())