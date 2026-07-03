"""empty message

Revision ID: init
Revises:
Create Date: 2025-01-01
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import TINYINT

revision = "init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "doc_chunks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("chunk_id", sa.String(64), nullable=False),
        sa.Column("doc_id", sa.String(64), nullable=False),
        sa.Column(
            "chunk_type",
            sa.Enum("text", "image", "table", "code", name="chunk_type_enum"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("page_num", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), comment='{"bbox": [...], "context_before": "..."}'),
        sa.Column("status", TINYINT(), server_default="1"),
        sa.Column("audit_status", TINYINT(), server_default="0"),
        sa.Column("creator", sa.String(64), nullable=True),
        sa.Column(
            "gmt_create",
            sa.DateTime(),
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column(
            "gmt_modified",
            sa.DateTime(),
            server_default=sa.func.current_timestamp(),
            onupdate=sa.func.current_timestamp(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_doc_id", "doc_chunks", ["doc_id"])
    op.create_index("idx_audit", "doc_chunks", ["audit_status", "status"])

    op.create_table(
        "tasks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("doc_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), server_default="'pending'"),
        sa.Column("progress", sa.Integer(), server_default="0"),
        sa.Column("stage", sa.String(64), nullable=True),
        sa.Column("chunk_count", sa.Integer(), server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "gmt_create",
            sa.DateTime(),
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column(
            "gmt_modified",
            sa.DateTime(),
            server_default=sa.func.current_timestamp(),
            onupdate=sa.func.current_timestamp(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_task_id", "tasks", ["task_id"])
    op.create_index("idx_task_doc", "tasks", ["doc_id"])


def downgrade() -> None:
    op.drop_table("tasks")
    op.drop_table("doc_chunks")
