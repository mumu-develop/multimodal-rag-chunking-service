"""Add documents table for document-level config and management.

Revision ID: 009
Revises: 008
Create Date: 2026-06-08

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '009'
down_revision = '008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create documents table for document-level management and config."""
    op.create_table(
        'documents',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('doc_id', sa.String(64), nullable=False, comment='文档唯一标识（UUID）'),
        sa.Column('kb_id', sa.BigInteger(), nullable=False, comment='关联知识库ID'),
        sa.Column('doc_name', sa.String(255), nullable=False, comment='文档名称'),
        sa.Column('doc_type', sa.String(32), nullable=True, default='', comment='文档类型'),
        sa.Column('file_size', sa.Integer(), nullable=True, comment='文件大小(字节)'),
        sa.Column('file_path', sa.String(512), nullable=True, comment='文件存储路径'),
        sa.Column('source_type', sa.String(16), nullable=True, default='local_file', comment='来源类型'),
        sa.Column('source_url', sa.Text(), nullable=True, comment='URL来源地址'),
        sa.Column('status', sa.String(32), server_default='pending', comment='状态'),
        sa.Column('chunk_count', sa.Integer(), server_default='0', comment='切片数量'),
        sa.Column('is_enabled', sa.SmallInteger(), server_default='1', comment='是否启用'),
        sa.Column('config', sa.JSON(), nullable=True, comment='文档切分配置JSON'),
        sa.Column('error_message', sa.Text(), nullable=True, comment='错误信息'),
        sa.Column('creator', sa.String(64), nullable=True, comment='创建者'),
        sa.Column('gmt_create', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('gmt_modified', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('doc_id'),
    )

    # 创建索引
    op.create_index('idx_doc_kb_id', 'documents', ['kb_id'])
    op.create_index('idx_doc_doc_id', 'documents', ['doc_id'])
    op.create_index('idx_doc_status', 'documents', ['status'])


def downgrade() -> None:
    """Drop documents table."""
    op.drop_index('idx_doc_status', 'documents')
    op.drop_index('idx_doc_doc_id', 'documents')
    op.drop_index('idx_doc_kb_id', 'documents')
    op.drop_table('documents')