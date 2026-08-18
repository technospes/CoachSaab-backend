# revision identifiers, used by Alembic.
# revision = '5f4172884c25'
"""init_core_tables

Revision ID: 001_init_core
Revises: 
Create Date: 2026-08-17 02:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '5f4172884c25'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # 3.1 users table
    op.execute("""
    CREATE TABLE users (
        user_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name              TEXT NOT NULL,
        email             TEXT UNIQUE,
        age               SMALLINT,
        height_cm         NUMERIC(5,1),
        weight_kg         NUMERIC(5,1),
        goals             TEXT[]  DEFAULT '{}',
        preferred_categories TEXT[] DEFAULT '{}',
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        deleted_at        TIMESTAMPTZ
    );
    """)

    # 3.2 activity_configs table
    op.execute("""
    CREATE TABLE activity_configs (
        activity_config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        category            TEXT NOT NULL,
        subcategory         TEXT,
        discipline          TEXT,
        style               TEXT,
        activity_key        TEXT NOT NULL UNIQUE,
        display_name         TEXT NOT NULL,
        icon_url             TEXT,
        required_landmarks   JSONB NOT NULL,
        camera_view          TEXT NOT NULL DEFAULT 'front',
        tracking_mode        TEXT NOT NULL,
        phases                JSONB,
        hud_config            JSONB NOT NULL,
        is_enabled            BOOLEAN NOT NULL DEFAULT true,
        version                INT NOT NULL DEFAULT 1,
        created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at             TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX idx_activity_configs_category ON activity_configs(category, is_enabled);
    """)

    # 3.3 activity_rules table
    op.execute("""
    CREATE TABLE activity_rules (
        rule_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        activity_config_id  UUID NOT NULL REFERENCES activity_configs(activity_config_id) ON DELETE CASCADE,
        joint                TEXT NOT NULL,
        phase                TEXT,
        metric                TEXT NOT NULL,
        target_min            NUMERIC,
        target_max            NUMERIC,
        deviation_type         TEXT NOT NULL,
        severity_thresholds     JSONB NOT NULL,
        is_enabled              BOOLEAN NOT NULL DEFAULT true
    );
    CREATE INDEX idx_activity_rules_config ON activity_rules(activity_config_id);
    """)

    # 3.4 activity_feedback_map table
    op.execute("""
    CREATE TABLE activity_feedback_map (
        feedback_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        activity_config_id   UUID NOT NULL REFERENCES activity_configs(activity_config_id) ON DELETE CASCADE,
        deviation_type         TEXT NOT NULL,
        locale                  TEXT NOT NULL DEFAULT 'en',
        display_text            TEXT NOT NULL,
        tts_text                TEXT NOT NULL,
        UNIQUE (activity_config_id, deviation_type, locale)
    );
    """)

    # Seed MVP Data for Squat
    op.execute("""
    WITH new_squat_config AS (
        INSERT INTO activity_configs (
            category, activity_key, display_name, required_landmarks, 
            tracking_mode, phases, hud_config
        ) VALUES (
            'fitness', 'squat', 'Squat',
            '["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]'::jsonb,
            'repetition',
            '["top", "descent", "bottom", "ascent"]'::jsonb,
            '{"counter": true, "timer": false, "phase": true}'::jsonb
        ) RETURNING activity_config_id
    ),
    insert_rules AS (
        INSERT INTO activity_rules (
            activity_config_id, joint, phase, metric, 
            target_min, target_max, deviation_type, severity_thresholds
        )
        SELECT 
            activity_config_id, 
            'knee', 'bottom', 'angle', 
            0, 105, 'insufficient_depth', 
            '{"minor": 5, "moderate": 10, "major": 20}'::jsonb
        FROM new_squat_config
        
        UNION ALL
        
        SELECT 
            activity_config_id, 
            'knee', NULL, 'visibility', 
            0.6, 1.0, 'obscured_knee', 
            '{"minor": 0.1, "moderate": 0.3, "major": 0.5}'::jsonb
        FROM new_squat_config
    )
    INSERT INTO activity_feedback_map (
        activity_config_id, deviation_type, locale, display_text, tts_text
    )
    SELECT 
        activity_config_id, 'insufficient_depth', 'en', 
        'Go Deeper! ⬇️', 'Watch your depth, go a bit lower.'
    FROM new_squat_config
    
    UNION ALL
    
    SELECT 
        activity_config_id, 'obscured_knee', 'en', 
        'Step back! I can''t see your legs.', 'Please step back so I can see your knees.'
    FROM new_squat_config;
    """)


def downgrade():
    op.execute("DROP TABLE IF EXISTS activity_feedback_map CASCADE;")
    op.execute("DROP TABLE IF EXISTS activity_rules CASCADE;")
    op.execute("DROP TABLE IF EXISTS activity_configs CASCADE;")
    op.execute("DROP TABLE IF EXISTS users CASCADE;")