from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.mysql import TINYINT
from sqlalchemy.orm import DeclarativeBase, Session

from app.config import settings


class Base(DeclarativeBase):
    pass


# ============================================================
# 现有表
# ============================================================

class DocChunk(Base):
    __tablename__ = "doc_chunks"
    __table_args__ = (
        Index("idx_doc_id", "doc_id"),
        Index("idx_audit", "audit_status", "status"),
        Index("idx_doc_type", "doc_type"),
        Index("idx_kb_id", "kb_id"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    chunk_id = Column(String(64), unique=True, nullable=False)
    doc_id = Column(String(64), nullable=False, index=True, comment="关联文档ID（documents.doc_id）")
    doc_type = Column(String(32), nullable=True, default="")
    chunk_type = Column(SAEnum("text", "image", "table", "code", name="chunk_type_enum"), nullable=False)
    content = Column(Text, nullable=False)
    caption = Column(Text, nullable=True)
    page_num = Column(Integer, nullable=False)
    heading_level = Column(Integer, server_default="0", comment="0=content, 1=h1, 2=h2...")
    heading_text = Column(String(255), nullable=True, default="")
    sort_order = Column(Integer, server_default="0")
    meta_data = Column("metadata", JSON, comment='{"bbox": [...], "context_before": "..."}')
    status = Column(TINYINT, server_default="1")
    audit_status = Column(TINYINT, server_default="0")
    creator = Column(String(64), nullable=True)
    # 新增字段（007 迁移）
    kb_id = Column(String(64), nullable=True, comment="关联知识库ID")
    is_enabled = Column(TINYINT, server_default="1", comment="是否启用 1=是 0=否")
    source_type = Column(String(16), nullable=True, comment="来源类型 local_file/url")
    source_url = Column(Text, nullable=True, comment="URL来源地址")
    file_size = Column(Integer, nullable=True, comment="文件大小(字节)")
    doc_name = Column(String(255), nullable=True, comment="文档名称")
    gmt_create = Column(DateTime, server_default=func.current_timestamp())
    gmt_modified = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())


class Document(Base):
    """文档表：关联知识库，存储文档元数据和切分配置。"""
    __tablename__ = "documents"
    __table_args__ = (
        Index("idx_doc_kb_id", "kb_id"),
        Index("idx_doc_doc_id", "doc_id"),
        Index("idx_doc_status", "status"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    doc_id = Column(String(64), unique=True, nullable=False, comment="文档唯一标识（UUID）")
    kb_id = Column(BigInteger, nullable=False, comment="关联知识库ID（knowledge_bases.id）")
    doc_name = Column(String(255), nullable=False, comment="文档名称")
    doc_type = Column(String(32), nullable=True, default="", comment="文档类型 pdf/docx/pptx/md")
    file_size = Column(Integer, nullable=True, comment="文件大小(字节)")
    file_path = Column(String(512), nullable=True, comment="文件存储路径")
    source_type = Column(String(16), nullable=True, default="local_file", comment="来源类型 local_file/url")
    source_url = Column(Text, nullable=True, comment="URL来源地址")
    status = Column(String(32), server_default="pending", comment="状态 pending/processing/success/failed")
    chunk_count = Column(Integer, server_default="0", comment="切片数量")
    is_enabled = Column(TINYINT, server_default="1", comment="是否启用 1=是 0=否")
    config = Column(JSON, nullable=True, comment="文档切分配置JSON")
    error_message = Column(Text, nullable=True, comment="错误信息")
    creator = Column(String(64), nullable=True, comment="创建者")
    gmt_create = Column(DateTime, server_default=func.current_timestamp())
    gmt_modified = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())


class TaskStatus(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        Index("idx_task_id", "task_id"),
        Index("idx_task_doc", "doc_id"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    task_id = Column(String(64), unique=True, nullable=False)
    doc_id = Column(String(64), nullable=True)
    doc_type = Column(String(32), nullable=True, default="")
    status = Column(String(32), server_default="pending")  # pending/processing/completed/failed
    progress = Column(Integer, server_default="0")
    stage = Column(String(64), nullable=True)
    chunk_count = Column(Integer, server_default="0")
    error_message = Column(Text, nullable=True)
    task_args = Column(JSON, nullable=True, comment='{"set_id": "...", "doc_ids": [...], "method": "fast"}')
    task_type = Column(String(64), nullable=True, comment="generate_testset | parse_and_chunk")
    # 新增字段（007 迁移）
    pipeline_nodes = Column(JSON, nullable=True, comment="[{\"name\":\"fetcher-1\",\"type\":\"fetcher\",\"status\":\"success\",\"duration_ms\":881,\"message\":\"...\"}]")
    task_params = Column(JSON, nullable=True, comment="{\"pipeline_id\":\"...\",\"source\":\"url\",\"url\":\"...\",\"chunks\":2}")
    doc_name = Column(String(255), nullable=True, comment="文档名称")
    gmt_create = Column(DateTime, server_default=func.current_timestamp())
    gmt_modified = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())


class TestSet(Base):
    __tablename__ = "test_set"
    __table_args__ = (
        Index("idx_set_id", "set_id"),
        Index("idx_doc_id", "doc_id"),
        Index("idx_gmt_create", "gmt_create"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    set_id = Column(String(64), nullable=False)
    set_name = Column(String(255), nullable=False, comment="测试集名称")
    query = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=False)
    doc_id = Column(String(64), nullable=True, default="")
    tags = Column(JSON, nullable=True)
    gmt_create = Column(DateTime, server_default=func.current_timestamp())


# ============================================================
# 新增表（007 迁移）
# ============================================================

class KnowledgeBase(Base):
    """知识库"""
    __tablename__ = "knowledge_bases"
    __table_args__ = (
        Index("idx_kb_owner", "owner"),
        Index("idx_kb_name", "name"),
        Index("idx_kb_type", "kb_type"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, comment="知识库名称")
    collection_name = Column(String(128), nullable=False, comment="Milvus集合名")
    owner = Column(String(64), nullable=False, comment="负责人")
    description = Column(Text, nullable=True, comment="描述")
    kb_type = Column(String(32), server_default="doc", comment="知识库类型 faq/doc/hybrid")
    is_enabled = Column(TINYINT, server_default="1", comment="是否启用")
    gmt_create = Column(DateTime, server_default=func.current_timestamp())
    gmt_modified = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())


class FAQItem(Base):
    """FAQ问答对"""
    __tablename__ = "faq_items"
    __table_args__ = (
        Index("idx_faq_kb_id", "kb_id"),
        Index("idx_faq_category", "kb_id", "category"),
        Index("idx_faq_enabled", "is_enabled"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    faq_id = Column(String(64), unique=True, nullable=False, comment="FAQ唯一标识（UUID）")
    kb_id = Column(BigInteger, nullable=False, comment="关联知识库ID")
    question = Column(Text, nullable=False, comment="问题文本")
    answer = Column(Text, nullable=False, comment="答案文本")
    category = Column(String(64), nullable=True, default="", comment="分类")
    keywords = Column(String(256), nullable=True, default="", comment="关键词标签")
    source = Column(String(32), server_default="manual", comment="来源 manual/import")
    hit_count = Column(Integer, server_default="0", comment="命中次数")
    sort_order = Column(Integer, server_default="0", comment="排序权重")
    is_enabled = Column(TINYINT, server_default="1", comment="是否启用 1=是 0=否")
    creator = Column(String(64), nullable=True, comment="创建者")
    gmt_create = Column(DateTime, server_default=func.current_timestamp())
    gmt_modified = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())


class IntentTreeNode(Base):
    """意图树节点"""
    __tablename__ = "intent_tree_nodes"
    __table_args__ = (
        Index("idx_intent_parent", "parent_id"),
        Index("idx_intent_type", "node_type"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, comment="节点名称")
    node_type = Column(SAEnum("DOMAIN", "CATEGORY", "TOPIC", name="intent_node_type_enum"), nullable=False, comment="节点类型")
    parent_id = Column(BigInteger, nullable=True, comment="父节点ID")
    sort_order = Column(Integer, server_default="0", comment="排序")
    collection_name = Column(String(128), nullable=True, comment="绑定的集合名")
    description = Column(Text, nullable=True, comment="描述")
    example_questions = Column(JSON, nullable=True, comment="示例问题列表")
    is_enabled = Column(TINYINT, server_default="1", comment="是否启用")
    gmt_create = Column(DateTime, server_default=func.current_timestamp())
    gmt_modified = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())


class ChatSession(Base):
    """会话记录"""
    __tablename__ = "sessions"
    __table_args__ = (
        Index("idx_session_id", "session_id"),
        Index("idx_session_user", "user_id"),
        Index("idx_session_gmt", "gmt_create"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, comment="用户标识")
    session_id = Column(String(128), unique=True, nullable=False, comment="会话ID")
    message_count = Column(Integer, server_default="0", comment="消息数")
    gmt_create = Column(DateTime, server_default=func.current_timestamp())


class Message(Base):
    """消息记录"""
    __tablename__ = "messages"
    __table_args__ = (
        Index("idx_msg_session", "session_id"),
        Index("idx_msg_gmt", "gmt_create"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(128), nullable=False, comment="会话ID")
    query = Column(Text, nullable=False, comment="用户查询")
    response = Column(Text, nullable=True, comment="AI回复")
    latency_ms = Column(Integer, nullable=True, comment="响应延迟(毫秒)")
    is_success = Column(TINYINT, server_default="1", comment="是否成功")
    error_msg = Column(Text, nullable=True, comment="错误信息")
    gmt_create = Column(DateTime, server_default=func.current_timestamp())


class UserProfile(Base):
    """用户"""
    __tablename__ = "users"
    __table_args__ = (
        Index("idx_user_username", "username"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, comment="用户名")
    password_hash = Column(String(255), nullable=False, comment="密码哈希")
    role = Column(SAEnum("admin", "user", name="user_role_enum"), server_default="user", comment="角色")
    is_active = Column(TINYINT, server_default="1", comment="是否激活")
    gmt_create = Column(DateTime, server_default=func.current_timestamp())
    gmt_modified = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())


class EvaluationRecord(Base):
    """评估记录"""
    __tablename__ = "evaluation_records"
    __table_args__ = (
        Index("idx_eval_kb_doc_set", "kb_id", "doc_id", "set_id"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    kb_id = Column(String(64), nullable=False, comment="知识库ID")
    doc_id = Column(String(64), nullable=False, comment="文档ID")
    set_id = Column(String(64), nullable=False, comment="测试集批次ID")
    metrics = Column(JSON, nullable=False, comment="评估指标JSON")
    gmt_create = Column(DateTime, server_default=func.current_timestamp())


# ============================================================
# 数据库连接
# ============================================================

_engine = None


def get_engine():
    """获取 SQLAlchemy Engine 单例。"""
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine
        _engine = create_engine(
            settings.DATABASE_URI,
            pool_recycle=3600,
            pool_size=10,
            max_overflow=5,
            echo=False,
        )
    return _engine


def get_session() -> Session:
    return Session(bind=get_engine())
