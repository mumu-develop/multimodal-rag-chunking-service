"""add knowledge_bases, intent_tree_nodes, sessions, messages, users tables
and add new columns to doc_chunks and tasks

Revision ID: 006
Revises: 005
Create Date: 2026-05-29
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import TINYINT


revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ----------------------------------------------------------
    # 新增表
    # ----------------------------------------------------------

    # knowledge_bases
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False, comment="知识库名称"),
        sa.Column("embedding_model", sa.String(64), nullable=False, comment="Embedding模型名"),
        sa.Column("collection_name", sa.String(128), nullable=False, comment="Milvus集合名"),
        sa.Column("owner", sa.String(64), nullable=False, comment="负责人"),
        sa.Column("description", sa.Text(), nullable=True, comment="描述"),
        sa.Column("is_enabled", TINYINT(), server_default="1", comment="是否启用"),
        sa.Column("gmt_create", sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column("gmt_modified", sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Index("idx_kb_owner", "owner"),
        sa.Index("idx_kb_name", "name"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # intent_tree_nodes
    op.create_table(
        "intent_tree_nodes",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False, comment="节点名称"),
        sa.Column("node_type", sa.Enum("DOMAIN", "CATEGORY", "TOPIC", name="intent_node_type_enum"), nullable=False, comment="节点类型"),
        sa.Column("parent_id", sa.BigInteger(), nullable=True, comment="父节点ID"),
        sa.Column("sort_order", sa.Integer(), server_default="0", comment="排序"),
        sa.Column("collection_name", sa.String(128), nullable=True, comment="绑定的集合名"),
        sa.Column("description", sa.Text(), nullable=True, comment="描述"),
        sa.Column("example_questions", sa.JSON(), nullable=True, comment="示例问题列表"),
        sa.Column("is_enabled", TINYINT(), server_default="1", comment="是否启用"),
        sa.Column("gmt_create", sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column("gmt_modified", sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Index("idx_intent_parent", "parent_id"),
        sa.Index("idx_intent_type", "node_type"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # sessions
    op.create_table(
        "sessions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False, comment="用户标识"),
        sa.Column("session_id", sa.String(128), unique=True, nullable=False, comment="会话ID"),
        sa.Column("message_count", sa.Integer(), server_default="0", comment="消息数"),
        sa.Column("gmt_create", sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Index("idx_session_id", "session_id"),
        sa.Index("idx_session_user", "user_id"),
        sa.Index("idx_session_gmt", "gmt_create"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # messages
    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(128), nullable=False, comment="会话ID"),
        sa.Column("query", sa.Text(), nullable=False, comment="用户查询"),
        sa.Column("response", sa.Text(), nullable=True, comment="AI回复"),
        sa.Column("latency_ms", sa.Integer(), nullable=True, comment="响应延迟(毫秒)"),
        sa.Column("is_success", TINYINT(), server_default="1", comment="是否成功"),
        sa.Column("error_msg", sa.Text(), nullable=True, comment="错误信息"),
        sa.Column("gmt_create", sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Index("idx_msg_session", "session_id"),
        sa.Index("idx_msg_gmt", "gmt_create"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # users
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("username", sa.String(64), unique=True, nullable=False, comment="用户名"),
        sa.Column("password_hash", sa.String(255), nullable=False, comment="密码哈希"),
        sa.Column("role", sa.Enum("admin", "user", name="user_role_enum"), server_default="user", comment="角色"),
        sa.Column("is_active", TINYINT(), server_default="1", comment="是否激活"),
        sa.Column("gmt_create", sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column("gmt_modified", sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Index("idx_user_username", "username"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # ----------------------------------------------------------
    # doc_chunks 新增字段
    # ----------------------------------------------------------
    op.add_column("doc_chunks", sa.Column("kb_id", sa.String(64), nullable=True, comment="关联知识库ID"))
    op.add_column("doc_chunks", sa.Column("is_enabled", TINYINT(), server_default="1", comment="是否启用"))
    op.add_column("doc_chunks", sa.Column("source_type", sa.String(16), nullable=True, comment="来源类型"))
    op.add_column("doc_chunks", sa.Column("source_url", sa.Text(), nullable=True, comment="URL来源地址"))
    op.add_column("doc_chunks", sa.Column("file_size", sa.Integer(), nullable=True, comment="文件大小(字节)"))
    op.add_column("doc_chunks", sa.Column("doc_name", sa.String(255), nullable=True, comment="文档名称"))
    op.create_index("idx_kb_id", "doc_chunks", ["kb_id"])

    # ----------------------------------------------------------
    # tasks 新增字段
    # ----------------------------------------------------------
    op.add_column("tasks", sa.Column("pipeline_nodes", sa.JSON(), nullable=True, comment="流水线节点详情"))
    op.add_column("tasks", sa.Column("task_params", sa.JSON(), nullable=True, comment="任务参数快照"))
    op.add_column("tasks", sa.Column("doc_name", sa.String(255), nullable=True, comment="文档名称"))


def downgrade() -> None:
    # tasks 字段
    op.drop_column("tasks", "doc_name")
    op.drop_column("tasks", "task_params")
    op.drop_column("tasks", "pipeline_nodes")

    # doc_chunks 字段
    op.drop_index("idx_kb_id", table_name="doc_chunks")
    op.drop_column("doc_chunks", "doc_name")
    op.drop_column("doc_chunks", "file_size")
    op.drop_column("doc_chunks", "source_url")
    op.drop_column("doc_chunks", "source_type")
    op.drop_column("doc_chunks", "is_enabled")
    op.drop_column("doc_chunks", "kb_id")

    # 新增表
    op.drop_table("users")
    op.drop_table("messages")
    op.drop_table("sessions")
    op.execute("DROP TYPE IF EXISTS intent_node_type_enum")
    op.drop_table("intent_tree_nodes")
    op.drop_table("knowledge_bases")
