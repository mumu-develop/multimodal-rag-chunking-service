"""Repository: MySQL document data access."""
from typing import Optional

import structlog

from app.models import Document, get_session

logger = structlog.get_logger()


# 默认文档切分配置（当 config 为空时使用）
DEFAULT_DOC_CONFIG = {
    "chunk_token_limit": 1024,
    "chunk_token_overlap": 100,
    "chunk_size_table": 1600,
    "parsing_strategy": "hi_res",
    "extract_images": True,
    "garbage_filter_enabled": True,
}


def get_doc_config_with_defaults(config: dict | None) -> dict:
    """合并文档配置与默认值，确保所有配置项都有值。

    Args:
        config: 文档的 config 字段（可能为 None 或部分字段）

    Returns:
        dict: 合并后的完整配置
    """
    if config is None:
        return DEFAULT_DOC_CONFIG.copy()

    merged = DEFAULT_DOC_CONFIG.copy()
    merged.update(config)
    return merged


class DocumentRepository:
    """MySQL documents table repository."""

    def __init__(self):
        self.last_error: str = ""

    def create(
        self,
        doc_id: str,
        kb_id: int,
        doc_name: str,
        doc_type: str = "",
        file_size: int = 0,
        file_path: str = "",
        source_type: str = "local_file",
        source_url: str = "",
        config: dict | None = None,
        creator: str = "",
        status: str = "success",
        chunk_count: int = 0,
    ) -> Optional[Document]:
        """创建文档记录。"""
        session = get_session()
        try:
            record = Document(
                doc_id=doc_id,
                kb_id=kb_id,
                doc_name=doc_name,
                doc_type=doc_type,
                file_size=file_size,
                file_path=file_path,
                source_type=source_type,
                source_url=source_url,
                config=config,
                creator=creator,
                status=status,
                chunk_count=chunk_count,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.info("doc_created", doc_id=doc_id, kb_id=kb_id, name=doc_name, status=status, chunk_count=chunk_count)
            self.last_error = ""
            return record
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("doc_create_failed", doc_id=doc_id, error=self.last_error)
            return None
        finally:
            session.close()

    @staticmethod
    def get_by_doc_id(doc_id: str) -> Optional[Document]:
        """根据 doc_id 获取文档。"""
        session = get_session()
        try:
            return session.query(Document).filter(Document.doc_id == doc_id).first()
        finally:
            session.close()

    @staticmethod
    def get_by_id(id: int) -> Optional[Document]:
        """根据 id 获取文档。"""
        session = get_session()
        try:
            return session.query(Document).filter(Document.id == id).first()
        finally:
            session.close()

    @staticmethod
    def list_by_kb(
        kb_id: int,
        page: int = 1,
        size: int = 20,
        keyword: str = "",
        status: str = "",
    ) -> tuple[int, list[dict]]:
        """查询知识库下的文档列表，实时统计 chunk_count。"""
        from sqlalchemy import func as sql_func
        from app.models import DocChunk
        session = get_session()
        try:
            # 子查询：统计每个 doc_id 的 chunk 数量
            chunk_count_subq = session.query(
                DocChunk.doc_id,
                sql_func.count(DocChunk.id).label("actual_chunk_count")
            ).filter(DocChunk.kb_id == str(kb_id)).group_by(DocChunk.doc_id).subquery()

            # 主查询：关联 documents 表和 chunk_count 子查询
            q = session.query(
                Document,
                chunk_count_subq.c.actual_chunk_count
            ).outerjoin(
                chunk_count_subq, Document.doc_id == chunk_count_subq.c.doc_id
            ).filter(Document.kb_id == kb_id)

            if keyword:
                q = q.filter(Document.doc_name.contains(keyword))
            if status:
                q = q.filter(Document.status == status)

            total = q.count()
            results = q.order_by(Document.gmt_create.desc()) \
                       .offset((page - 1) * size).limit(size).all()

            # 转换为 dict 列表，包含实时 chunk_count
            items = []
            for record, actual_chunk_count in results:
                item = {
                    "id": record.id,
                    "doc_id": record.doc_id,
                    "kb_id": record.kb_id,
                    "doc_name": record.doc_name,
                    "doc_type": record.doc_type or "",
                    "file_size": record.file_size or 0,
                    "source_type": record.source_type or "local_file",
                    "source_url": record.source_url or "",
                    "status": record.status or "pending",
                    "chunk_count": actual_chunk_count or 0,  # 实时统计
                    "is_enabled": record.is_enabled,
                    "config": record.config,
                    "error_message": record.error_message,
                    "creator": record.creator,
                    "gmt_create": record.gmt_create.isoformat() if record.gmt_create else "",
                    "gmt_modified": record.gmt_modified.isoformat() if record.gmt_modified else "",
                }
                items.append(item)

            return total, items
        finally:
            session.close()

    def update(self, doc_id: str, **kwargs) -> bool:
        """更新文档记录。"""
        session = get_session()
        try:
            row = session.query(Document).filter(Document.doc_id == doc_id).update(kwargs)
            session.commit()
            logger.info("doc_updated", doc_id=doc_id, fields=list(kwargs.keys()))
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("doc_update_failed", doc_id=doc_id, error=self.last_error)
            return False
        finally:
            session.close()

    def update_status(self, doc_id: str, status: str, chunk_count: int = 0, error_message: str = "") -> bool:
        """更新文档状态。"""
        return self.update(
            doc_id,
            status=status,
            chunk_count=chunk_count,
            error_message=error_message if error_message else None,
        )

    def delete(self, doc_id: str) -> bool:
        """删除文档记录。"""
        session = get_session()
        try:
            row = session.query(Document).filter(Document.doc_id == doc_id).delete()
            session.commit()
            logger.info("doc_deleted", doc_id=doc_id)
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("doc_delete_failed", doc_id=doc_id, error=self.last_error)
            return False
        finally:
            session.close()

    @staticmethod
    def count_by_kb(kb_id: int) -> int:
        """统计知识库下的文档数量。"""
        session = get_session()
        try:
            return session.query(Document).filter(Document.kb_id == kb_id).count()
        finally:
            session.close()


# 单例
doc_repo = DocumentRepository()