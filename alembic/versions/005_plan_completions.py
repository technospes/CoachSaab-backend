"""add_plan_completions

Revision ID: 005_plan_completions
Revises: 004_workout_plans
Create Date: 2026-08-20 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '005_plan_completions'
down_revision = '004_workout_plans'
branch_labels = None
depends_on = None

def upgrade():
    op.execute("""
    CREATE TABLE plan_day_completions (
        completion_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        plan_id UUID NOT NULL REFERENCES workout_plans(plan_id) ON DELETE CASCADE,
        user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        week_number INT NOT NULL,
        day_number INT NOT NULL,
        is_completed BOOLEAN NOT NULL DEFAULT true,
        completed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        -- Ensures a user cannot duplicate a completion for the same day
        UNIQUE(plan_id, week_number, day_number)
    );
    
    CREATE INDEX idx_plan_completions ON plan_day_completions(plan_id);
    """)

def downgrade():
    op.execute("DROP TABLE IF EXISTS plan_day_completions CASCADE;")