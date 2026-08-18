"""add_workout_plans

Revision ID: 004_workout_plans
Revises: 003_workout_sessions
Create Date: 2026-08-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '004_workout_plans'
down_revision = '003_workout_sessions'
branch_labels = None
depends_on = None

def upgrade():
    # Create the workout_plans table so the Agent has a place to save data
    op.execute("""
    CREATE TABLE workout_plans (
        plan_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        plan_name TEXT NOT NULL,
        plan_json JSONB NOT NULL,
        is_active BOOLEAN DEFAULT true,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    
    CREATE INDEX idx_workout_plans_user ON workout_plans(user_id, is_active);
    """)

def downgrade():
    op.execute("DROP TABLE IF EXISTS workout_plans CASCADE;")