"""add_user_chat

Revision ID: 002_user_chat
Revises: 5f4172884c25
Create Date: 2026-08-17 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002_user_chat'
down_revision = '5f4172884c25' # Links this to your previous migration
branch_labels = None
depends_on = None

def upgrade():
    # 1. Add gender to the users table
    op.execute("ALTER TABLE users ADD COLUMN gender TEXT;")

    # 2. Create chatbot_conversations table
    op.execute("""
    CREATE TABLE chatbot_conversations (
        conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """)

    # 3. Create chatbot_messages table
    op.execute("""
    CREATE TABLE chatbot_messages (
        message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        conversation_id UUID NOT NULL REFERENCES chatbot_conversations(conversation_id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        context_snapshot JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX idx_chat_messages_conv ON chatbot_messages(conversation_id, created_at ASC);
    """)

def downgrade():
    op.execute("DROP TABLE IF EXISTS chatbot_messages CASCADE;")
    op.execute("DROP TABLE IF EXISTS chatbot_conversations CASCADE;")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS gender;")