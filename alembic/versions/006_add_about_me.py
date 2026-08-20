"""add_about_me

Revision ID: 006_add_about_me
Revises: 005_plan_completions
Create Date: 2026-08-20 17:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '006_add_about_me'
down_revision = '005_plan_completions'
branch_labels = None
depends_on = None

def upgrade():
    # We use IF NOT EXISTS just in case it was already run manually via Supabase dashboard
    op.execute("""
    ALTER TABLE users ADD COLUMN IF NOT EXISTS about_me TEXT;
    """)

def downgrade():
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS about_me;")