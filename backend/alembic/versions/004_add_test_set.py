"""add test_set table for persisting RAGAS test cases

Revision ID: 004
Revises: 003
Create Date: 2026-05-28
"""
from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "test_set",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("set_id", sa.String(64), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("ground_truth", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.String(64), nullable=True, server_default=""),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("gmt_create", sa.DateTime(), server_default=sa.func.current_timestamp()),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_set_id", "test_set", ["set_id"])
    op.create_index("idx_doc_id", "test_set", ["doc_id"])
    op.create_index("idx_gmt_create", "test_set", ["gmt_create"])


def downgrade() -> None:
    op.drop_index("idx_gmt_create", "test_set")
    op.drop_index("idx_doc_id", "test_set")
    op.drop_index("idx_set_id", "test_set")
    op.drop_table("test_set")
