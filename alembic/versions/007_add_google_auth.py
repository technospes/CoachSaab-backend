"""add_google_auth

Revision ID: 006_google_auth
Revises: 005_plan_completions
Create Date: 2026-08-22 17:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '007_google_auth'
down_revision = '006_add_about_me'
branch_labels = None
depends_on = None

def upgrade():
    # 1. Add auth_provider column (defaults to 'email' for existing users)
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider TEXT DEFAULT 'email';")
    
    # 2. Add google_sub column (unique identifier from Google)
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_sub TEXT UNIQUE;")
    
    # 3. Make password_hash nullable (Google users will not have a password hash)
    op.execute("ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL;")

def downgrade():
    # Revert the changes if needed
    op.execute("ALTER TABLE users ALTER COLUMN password_hash SET NOT NULL;")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS google_sub;")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS auth_provider;")