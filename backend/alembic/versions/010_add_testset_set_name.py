"""Add set_name column to test_set table.

Revision ID: 010
Revises: 009
Create Date: 2026-06-08

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '010'
down_revision = '009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add set_name column to test_set table."""
    op.add_column(
        'test_set',
        sa.Column('set_name', sa.String(255), nullable=False, comment='测试集名称'),
    )


def downgrade() -> None:
    """Remove set_name column from test_set table."""
    op.drop_column('test_set', 'set_name')