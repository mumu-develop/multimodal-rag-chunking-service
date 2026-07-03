"""remove embedding_model from knowledge_bases

Revision ID: 007
Revises: 006
Create Date: 2026-05-29
"""
from alembic import op
import sqlalchemy as sa


revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 删除 knowledge_bases 表的 embedding_model 字段
    op.drop_column("knowledge_bases", "embedding_model")


def downgrade() -> None:
    # 回滚：重新添加 embedding_model 字段
    op.add_column(
        "knowledge_bases",
        sa.Column("embedding_model", sa.String(64), nullable=False, comment="Embedding模型名"),
    )