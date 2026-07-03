"""日活统计表（按天维度）。

Revision ID: 013
Revises: 012
Create Date: 2026-06-16

"""
from alembic import op
import sqlalchemy as sa


revision = '013'
down_revision = '012'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """创建日活统计表（按天维度）。"""
    op.create_table(
        'daily_stats',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('date', sa.Date(), nullable=False, comment='日期'),
        sa.Column('active_users', sa.Integer(), nullable=False, default=0, comment='日活用户数'),
        sa.Column('total_requests', sa.Integer(), nullable=False, default=0, comment='总请求数'),
        sa.Column('gmt_create', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('gmt_modified', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('date', name='uq_daily_stats_date'),  # 每天只有一条记录
    )

    op.create_index('idx_daily_stats_date', 'daily_stats', ['date'])


def downgrade() -> None:
    """删除日活统计表。"""
    op.drop_index('idx_daily_stats_date', 'daily_stats')
    op.drop_table('daily_stats')