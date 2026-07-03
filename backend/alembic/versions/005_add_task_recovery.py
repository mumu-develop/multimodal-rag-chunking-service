"""add task_args and task_type columns to tasks table

Revision ID: 005
Revises: 004
Create Date: 2026-05-28
"""
from alembic import op
import sqlalchemy as sa

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("task_args", sa.JSON(), nullable=True, comment='{"set_id": "...", "doc_ids": [...]}'))
    op.add_column("tasks", sa.Column("task_type", sa.String(64), nullable=True, comment="generate_testset | parse_and_chunk"))


def downgrade() -> None:
    op.drop_column("tasks", "task_type")
    op.drop_column("tasks", "task_args")
