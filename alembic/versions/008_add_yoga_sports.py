"""add_yoga_sports

Revision ID: 008_add_yoga_sports
Revises: 007_google_auth
Create Date: 2026-08-30 10:00:00.000000

"""
from alembic import op

revision = '008_add_yoga_sports'
down_revision = '007_google_auth'
branch_labels = None
depends_on = None

def upgrade():
    # 1. YOGA: Tree Pose (Asymmetric & Configurable Hold Time)
    op.execute("""
    WITH new_tree_config AS (
        INSERT INTO activity_configs (
            category, activity_key, display_name, required_landmarks, 
            tracking_mode, phases, hud_config, camera_view
        ) VALUES (
            'yoga', 'tree_pose', 'Tree Pose',
            '["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]'::jsonb,
            'hold', 
            '["setup", "hold"]'::jsonb,
            '{"timer": true, "phase": false, "target_duration_seconds": 30}'::jsonb,
            'front'
        ) RETURNING activity_config_id
    ),
    insert_tree_rules AS (
        INSERT INTO activity_rules (
            activity_config_id, joint, phase, metric, 
            target_min, target_max, deviation_type, severity_thresholds
        )
        -- Explicitly separating standing vs raised leg constraints
        SELECT activity_config_id, 'standing_knee', 'hold', 'angle', 160, 180, 'bent_standing_knee', '{"minor": 10, "moderate": 20, "major": 30}'::jsonb FROM new_tree_config
        UNION ALL
        SELECT activity_config_id, 'raised_knee', 'hold', 'angle', 30, 120, 'raised_leg_dropped', '{"minor": 15, "moderate": 25, "major": 35}'::jsonb FROM new_tree_config
        RETURNING rule_id
    )
    INSERT INTO activity_feedback_map (activity_config_id, deviation_type, locale, display_text, tts_text)
    SELECT activity_config_id, 'bent_standing_knee', 'en', 'Straighten your standing leg! 🧘', 'Keep your standing leg straight and strong.' FROM new_tree_config
    UNION ALL
    SELECT activity_config_id, 'raised_leg_dropped', 'en', 'Lift your raised leg higher!', 'Bring your raised foot higher up your standing leg.' FROM new_tree_config;
    """)

    # 2. SPORTS: Running
    op.execute("""
    WITH new_run_config AS (
        INSERT INTO activity_configs (
            category, activity_key, display_name, required_landmarks, 
            tracking_mode, phases, hud_config, camera_view
        ) VALUES (
            'sports', 'running', 'Running',
            '["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]'::jsonb,
            'continuous',
            '["stance", "swing"]'::jsonb,
            '{"timer": true, "cadence": true}'::jsonb,
            'side'
        ) RETURNING activity_config_id
    ),
    insert_run_rules AS (
        INSERT INTO activity_rules (
            activity_config_id, joint, phase, metric, target_min, target_max, deviation_type, severity_thresholds
        )
        SELECT activity_config_id, 'elbow', NULL, 'angle', 75, 110, 'improper_arm_swing', '{"minor": 15, "moderate": 25, "major": 35}'::jsonb FROM new_run_config
        RETURNING rule_id
    )
    INSERT INTO activity_feedback_map (activity_config_id, deviation_type, locale, display_text, tts_text)
    SELECT activity_config_id, 'improper_arm_swing', 'en', 'Bend elbows at 90°! 🏃', 'Keep your elbows bent at ninety degrees.' FROM new_run_config;
    """)

def downgrade():
    op.execute("DELETE FROM activity_configs WHERE activity_key IN ('tree_pose', 'running');")