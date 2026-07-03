"""Add audit logs and daily active users tables.

Revision ID: 012
Revises: 011
Create Date: 2026-06-16

This migration adds:
1. audit_logs table - 记录用户访问审计日志
2. daily_active_users table - 日活统计

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '012'
down_revision = '011'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create audit_logs and daily_active_users tables."""
    # 1. Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(64), nullable=False, comment='用户ID'),
        sa.Column('user_role', sa.String(32), nullable=True, default='guest', comment='用户角色'),
        sa.Column('path', sa.String(256), nullable=False, comment='请求路径'),
        sa.Column('method', sa.String(16), nullable=False, comment='请求方法 GET/POST/PUT/DELETE'),
        sa.Column('params', sa.Text(), nullable=True, comment='请求参数（JSON）'),
        sa.Column('status_code', sa.SmallInteger(), nullable=False, comment='响应状态码'),
        sa.Column('latency_ms', sa.Integer(), nullable=False, comment='请求耗时（毫秒）'),
        sa.Column('trace_id', sa.String(64), nullable=True, comment='追踪ID'),
        sa.Column('gmt_create', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
    )

    # 2. Create indexes for audit_logs
    op.create_index('idx_audit_user_id', 'audit_logs', ['user_id'])
    op.create_index('idx_audit_path', 'audit_logs', ['path'])
    op.create_index('idx_audit_gmt_create', 'audit_logs', ['gmt_create'])
    op.create_index('idx_audit_user_date', 'audit_logs', ['user_id', 'gmt_create'])

    # 3. Create daily_active_users table
    op.create_table(
        'daily_active_users',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(64), nullable=False, comment='用户ID'),
        sa.Column('date', sa.Date(), nullable=False, comment='日期'),
        sa.Column('gmt_create', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'date', name='uq_user_date'),  # 每天每个用户只记录一次
    )

    # 4. Create indexes for daily_active_users
    op.create_index('idx_dau_date', 'daily_active_users', ['date'])
    op.create_index('idx_dau_user_id', 'daily_active_users', ['user_id'])


def downgrade() -> None:
    """Drop audit_logs and daily_active_users tables."""
    # Drop daily_active_users indexes and table
    op.drop_index('idx_dau_user_id', 'daily_active_users')
    op.drop_index('idx_dau_date', 'daily_active_users')
    op.drop_table('daily_active_users')

    # Drop audit_logs indexes and table
    op.drop_index('idx_audit_user_date', 'audit_logs')
    op.drop_index('idx_audit_gmt_create', 'audit_logs')
    op.drop_index('idx_audit_path', 'audit_logs')
    op.drop_index('idx_audit_user_id', 'audit_logs')
    op.drop_table('audit_logs')