"""add evaluation_records table

Revision ID: 008
Revises: 007
Create Date: 2026-05-30
"""
from alembic import op
import sqlalchemy as sa


revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 创建 evaluation_records 表
    op.create_table(
        "evaluation_records",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("kb_id", sa.String(64), nullable=False, comment="知识库ID"),
        sa.Column("doc_id", sa.String(64), nullable=False, comment="文档ID"),
        sa.Column("set_id", sa.String(64), nullable=False, comment="测试集批次ID"),
        sa.Column("metrics", sa.JSON, nullable=False, comment="评估指标JSON"),
        sa.Column("gmt_create", sa.DateTime, server_default=sa.func.current_timestamp()),
        sa.PrimaryKeyConstraint("id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # 创建索引
    op.create_index("idx_eval_kb_doc_set", "evaluation_records", ["kb_id", "doc_id", "set_id"])


def downgrade() -> None:
    # 删除索引
    op.drop_index("idx_eval_kb_doc_set", "evaluation_records")

    # 删除表
    op.drop_table("evaluation_records")