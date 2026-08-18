"""add_workout_sessions

Revision ID: 003_workout_sessions
Revises: 002_user_chat
Create Date: 2026-08-18 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '003_workout_sessions'
down_revision = '002_user_chat'
branch_labels = None
depends_on = None

def upgrade():
    # Create the workout_sessions table
    op.execute("""
    CREATE TABLE workout_sessions (
        session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        activity_key TEXT NOT NULL,
        reps INT NOT NULL,
        duration_seconds INT NOT NULL,
        form_score INT NOT NULL,
        dominant_deviation TEXT,
        deviations_json JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    
    -- Add an index so we can quickly pull a user's recent workouts for the AI
    CREATE INDEX idx_workout_sessions_user ON workout_sessions(user_id, created_at DESC);
    """)

def downgrade():
    op.execute("DROP TABLE IF EXISTS workout_sessions CASCADE;")