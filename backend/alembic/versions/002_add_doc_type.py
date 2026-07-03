"""add doc_type column to doc_chunks and tasks

Revision ID: 002
Revises: init
Create Date: 2026-05-24
"""
from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add doc_type to doc_chunks
    op.add_column("doc_chunks", sa.Column("doc_type", sa.String(32), nullable=True, default=""))
    op.create_index("idx_doc_type", "doc_chunks", ["doc_type"])

    # Add doc_type to tasks
    op.add_column("tasks", sa.Column("doc_type", sa.String(32), nullable=True, default=""))


def downgrade() -> None:
    op.drop_column("tasks", "doc_type")
    op.drop_index("idx_doc_type", table_name="doc_chunks")
    op.drop_column("doc_chunks", "doc_type")
