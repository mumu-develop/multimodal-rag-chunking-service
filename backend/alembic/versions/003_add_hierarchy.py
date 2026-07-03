"""add hierarchy columns (heading_level, heading_text, sort_order) to doc_chunks

Revision ID: 003
Revises: 002
Create Date: 2026-05-25
"""
from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("doc_chunks", sa.Column("heading_level", sa.Integer(), nullable=True, server_default="0"))
    op.add_column("doc_chunks", sa.Column("heading_text", sa.String(255), nullable=True, default=""))
    op.add_column("doc_chunks", sa.Column("sort_order", sa.Integer(), nullable=True, server_default="0"))


def downgrade() -> None:
    op.drop_column("doc_chunks", "sort_order")
    op.drop_column("doc_chunks", "heading_text")
    op.drop_column("doc_chunks", "heading_level")
