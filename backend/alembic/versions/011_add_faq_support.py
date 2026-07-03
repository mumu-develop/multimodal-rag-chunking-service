"""Add FAQ knowledge base support.

Revision ID: 011
Revises: 010
Create Date: 2026-06-14

This migration adds:
1. kb_type field to knowledge_bases table (faq/doc/hybrid)
2. faq_items table for storing FAQ question-answer pairs

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '011'
down_revision = '010'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add kb_type to knowledge_bases and create faq_items table."""
    # 1. Add kb_type column to knowledge_bases
    op.add_column(
        'knowledge_bases',
        sa.Column('kb_type', sa.String(32), server_default='doc', comment='知识库类型 faq/doc/hybrid'),
    )
    op.create_index('idx_kb_type', 'knowledge_bases', ['kb_type'])

    # 2. Create faq_items table
    op.create_table(
        'faq_items',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('faq_id', sa.String(64), nullable=False, comment='FAQ唯一标识（UUID）'),
        sa.Column('kb_id', sa.BigInteger(), nullable=False, comment='关联知识库ID'),
        sa.Column('question', sa.Text(), nullable=False, comment='问题文本'),
        sa.Column('answer', sa.Text(), nullable=False, comment='答案文本'),
        sa.Column('category', sa.String(64), nullable=True, default='', comment='分类'),
        sa.Column('keywords', sa.String(256), nullable=True, default='', comment='关键词标签'),
        sa.Column('source', sa.String(32), server_default='manual', comment='来源 manual/import'),
        sa.Column('hit_count', sa.Integer(), server_default='0', comment='命中次数'),
        sa.Column('sort_order', sa.Integer(), server_default='0', comment='排序权重'),
        sa.Column('is_enabled', sa.SmallInteger(), server_default='1', comment='是否启用 1=是 0=否'),
        sa.Column('creator', sa.String(64), nullable=True, comment='创建者'),
        sa.Column('gmt_create', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('gmt_modified', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('faq_id'),
    )

    # 3. Create indexes for faq_items
    op.create_index('idx_faq_kb_id', 'faq_items', ['kb_id'])
    op.create_index('idx_faq_category', 'faq_items', ['kb_id', 'category'])
    op.create_index('idx_faq_enabled', 'faq_items', ['is_enabled'])


def downgrade() -> None:
    """Remove kb_type column and drop faq_items table."""
    # Drop faq_items indexes
    op.drop_index('idx_faq_enabled', 'faq_items')
    op.drop_index('idx_faq_category', 'faq_items')
    op.drop_index('idx_faq_kb_id', 'faq_items')

    # Drop faq_items table
    op.drop_table('faq_items')

    # Drop kb_type index and column
    op.drop_index('idx_kb_type', 'knowledge_bases')
    op.drop_column('knowledge_bases', 'kb_type')