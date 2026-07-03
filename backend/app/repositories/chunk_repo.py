"""Repository: MySQL doc_chunks data access."""
import json
from typing import Optional

import structlog
from sqlalchemy import text

from app.config import settings
from app.models import DocChunk, get_session

logger = structlog.get_logger()


class ChunkRepository:
    """MySQL doc_chunks table repository."""

    def __init__(self):
        self.last_error: str = ""

    def insert_chunks(self, chunks: list[dict]) -> bool:
        """批量插入 chunk 到 MySQL。"""
        session = get_session()
        try:
            for c in chunks:
                stmt = text(
                    """
                    INSERT IGNORE INTO doc_chunks
                        (chunk_id, doc_id, chunk_type, content, caption, page_num,
                         heading_level, heading_text, sort_order,
                         metadata, creator, doc_type, kb_id, doc_name, file_size, source_type, source_url)
                    VALUES
                        (:chunk_id, :doc_id, :chunk_type, :content, :caption, :page_num,
                         :heading_level, :heading_text, :sort_order,
                         :metadata, :creator, :doc_type, :kb_id, :doc_name, :file_size, :source_type, :source_url)
                    """
                )
                session.execute(stmt, {
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "chunk_type": c["chunk_type"],
                    "content": c["content"],
                    "caption": c.get("caption"),
                    "page_num": c["page_num"],
                    "heading_level": c.get("heading_level", 0),
                    "heading_text": c.get("heading_text", ""),
                    "sort_order": c.get("sort_order", 0),
                    "metadata": json.dumps(c.get("metadata", {})),
                    "creator": c.get("creator"),
                    "doc_type": c.get("metadata", {}).get("doc_type", ""),
                    "kb_id": c.get("kb_id", ""),
                    "doc_name": c.get("doc_name", ""),
                    "file_size": c.get("file_size", 0),
                    "source_type": c.get("source_type", "local_file"),
                    "source_url": c.get("source_url", ""),
                })
            session.commit()
            logger.info("mysql_insert_ok", count=len(chunks))
            self.last_error = ""
            return True
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("mysql_insert_failed", count=len(chunks), error=self.last_error)
            return False
        finally:
            session.close()

    @staticmethod
    def get_by_doc(
        doc_id: str,
        page: int = 1,
        size: int = 10,
        chunk_type: Optional[str] = None,
    ) -> tuple[int, list[DocChunk]]:
        """按 doc_id 分页查询 chunk。"""
        session = get_session()
        try:
            query = session.query(DocChunk).filter(DocChunk.doc_id == doc_id)
            if chunk_type:
                query = query.filter(DocChunk.chunk_type == chunk_type)

            total = query.count()
            items = query.order_by(DocChunk.sort_order, DocChunk.id).offset((page - 1) * size).limit(size).all()
            return total, items
        finally:
            session.close()

    @staticmethod
    def get_all_by_doc(doc_id: str) -> list[DocChunk]:
        """获取指定 doc_id 的全部 chunk。"""
        session = get_session()
        try:
            return (
                session.query(DocChunk)
                .filter(DocChunk.doc_id == doc_id)
                .order_by(DocChunk.sort_order, DocChunk.id)
                .all()
            )
        finally:
            session.close()

    def update_chunk(self, chunk_id: str, content: str) -> bool:
        """编辑切片内容。"""
        session = get_session()
        try:
            row = session.query(DocChunk).filter(DocChunk.chunk_id == chunk_id) \
                .update({"content": content, "gmt_modified": None})  # None triggers onupdate
            session.commit()
            logger.info("chunk_updated", chunk_id=chunk_id, row_count=row)
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("chunk_update_failed", chunk_id=chunk_id, error=self.last_error)
            return False
        finally:
            session.close()

    def delete_chunk(self, chunk_id: str) -> bool:
        """删除单个切片。"""
        session = get_session()
        try:
            row = session.query(DocChunk).filter(DocChunk.chunk_id == chunk_id).delete()
            session.commit()
            logger.info("chunk_deleted", chunk_id=chunk_id, row_count=row)
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("chunk_delete_failed", chunk_id=chunk_id, error=self.last_error)
            return False
        finally:
            session.close()

    def toggle_chunk(self, chunk_id: str, is_enabled: bool) -> bool:
        """启用/禁用单个切片。"""
        session = get_session()
        try:
            val = 1 if is_enabled else 0
            row = session.query(DocChunk).filter(DocChunk.chunk_id == chunk_id) \
                .update({"is_enabled": val})
            session.commit()
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("chunk_toggle_failed", chunk_id=chunk_id, error=self.last_error)
            return False
        finally:
            session.close()

    def batch_toggle(self, chunk_ids: list[str], is_enabled: bool) -> int:
        """批量启用/禁用切片。"""
        session = get_session()
        try:
            val = 1 if is_enabled else 0
            row = session.query(DocChunk).filter(DocChunk.chunk_id.in_(chunk_ids)) \
                .update({"is_enabled": val}, synchronize_session=False)
            session.commit()
            logger.info("chunk_batch_toggle", count=len(chunk_ids), enabled=is_enabled, affected=row)
            self.last_error = ""
            return row
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("chunk_batch_toggle_failed", error=self.last_error)
            return 0
        finally:
            session.close()

    def batch_delete(self, chunk_ids: list[str]) -> int:
        """批量删除切片。"""
        session = get_session()
        try:
            row = session.query(DocChunk).filter(DocChunk.chunk_id.in_(chunk_ids)) \
                .delete(synchronize_session=False)
            session.commit()
            logger.info("chunk_batch_delete", count=len(chunk_ids), affected=row)
            self.last_error = ""
            return row
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("chunk_batch_delete_failed", error=self.last_error)
            return 0
        finally:
            session.close()

    def delete_by_kb(self, kb_id: str) -> int:
        """删除指定知识库的所有切片。"""
        session = get_session()
        try:
            row = session.query(DocChunk).filter(DocChunk.kb_id == str(kb_id)) \
                .delete(synchronize_session=False)
            session.commit()
            logger.info("chunk_delete_by_kb", kb_id=kb_id, count=row)
            self.last_error = ""
            return row
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("chunk_delete_by_kb_failed", kb_id=kb_id)
            return 0
        finally:
            session.close()

    @staticmethod
    def get_by_chunk_id(chunk_id: str) -> Optional[DocChunk]:
        """根据 chunk_id 获取单个切片。"""
        session = get_session()
        try:
            return session.query(DocChunk).filter(DocChunk.chunk_id == chunk_id).first()
        finally:
            session.close()

    def toggle_all_by_doc(self, doc_id: str, is_enabled: bool) -> int:
        """全量启用/禁用某个文档的所有切片。"""
        session = get_session()
        try:
            val = 1 if is_enabled else 0
            row = session.query(DocChunk).filter(DocChunk.doc_id == doc_id) \
                .update({"is_enabled": val}, synchronize_session=False)
            session.commit()
            self.last_error = ""
            return row
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("chunk_toggle_all_failed", doc_id=doc_id, error=self.last_error)
            return 0
        finally:
            session.close()

    @staticmethod
    def list_documents(kb_id: str = "", page: int = 1, size: int = 20,
                       keyword: str = "", status_filter: str = "") -> tuple[int, list[dict]]:
        """按 doc_id 聚合查询文档列表。每个文档一条记录（取第一个 chunk 的元信息）。"""
        session = get_session()
        try:
            from sqlalchemy import func as sql_func
            q = session.query(
                DocChunk.doc_id,
                sql_func.min(DocChunk.doc_name).label("doc_name"),
                sql_func.min(DocChunk.doc_type).label("doc_type"),
                sql_func.min(DocChunk.source_type).label("source_type"),
                sql_func.min(DocChunk.source_url).label("source_url"),
                sql_func.min(DocChunk.file_size).label("file_size"),
                sql_func.count(DocChunk.id).label("chunk_count"),
                sql_func.max(DocChunk.gmt_modified).label("gmt_modified"),
                sql_func.min(DocChunk.is_enabled).label("is_enabled"),
                sql_func.min(DocChunk.status).label("doc_status"),
            ).group_by(DocChunk.doc_id)

            if kb_id:
                q = q.filter(DocChunk.kb_id == kb_id)
            if keyword:
                q = q.filter(DocChunk.doc_name.contains(keyword))

            subq = q.subquery()
            q2 = session.query(subq)
            if status_filter == "success":
                q2 = q2.filter(subq.c.doc_status == 1)
            elif status_filter == "failed":
                q2 = q2.filter(subq.c.doc_status == 0)
            elif status_filter == "disabled":
                q2 = q2.filter(subq.c.is_enabled == 0)

            total = q2.count()
            items = q2.order_by(subq.c.gmt_modified.desc()) \
                      .offset((page - 1) * size).limit(size).all()

            result = [
                {
                    "doc_id": row.doc_id,
                    "doc_name": row.doc_name or "",
                    "doc_type": row.doc_type or "",
                    "source_type": row.source_type or "local_file",
                    "source_url": row.source_url or "",
                    "file_size": row.file_size or 0,
                    "chunk_count": row.chunk_count,
                    "status": "success" if row.doc_status == 1 else "failed",
                    "is_enabled": bool(row.is_enabled),
                    "gmt_modified": row.gmt_modified.isoformat() if row.gmt_modified else "",
                }
                for row in items
            ]
            return total, result
        finally:
            session.close()

    @staticmethod
    def get_nested_table(parent_chunk_id: str, nested_index: int) -> Optional[DocChunk]:
        """
        获取嵌套表格 chunk。

        根据 parent_chunk_id 和 nested_index 查找嵌套表格 chunk。
        嵌套表格的 chunk_id 格式：{parent_chunk_id}_nested_{nested_index}

        参数：
            parent_chunk_id: 父表格 chunk ID
            nested_index: 嵌套表格索引

        返回：
            DocChunk 对象或 None
        """
        session = get_session()
        try:
            # 方式1：直接根据 chunk_id 格式查询
            nested_chunk_id = f"{parent_chunk_id}_nested_{nested_index}"
            chunk = session.query(DocChunk).filter(DocChunk.chunk_id == nested_chunk_id).first()
            if chunk:
                return chunk

            # 方式2：根据 metadata 中的 parent_chunk_id 和 nested_index 查询
            # 使用 JSON 查询（MySQL 5.7+）
            from sqlalchemy import cast, String
            chunk = session.query(DocChunk).filter(
                DocChunk.chunk_type == "nested_table",
                DocChunk.metadata.contains(f'"parent_chunk_id": "{parent_chunk_id}"'),
                DocChunk.metadata.contains(f'"nested_index": {nested_index}'),
            ).first()
            return chunk
        finally:
            session.close()

    @staticmethod
    def get_nested_tables_by_parent(parent_chunk_id: str) -> list[DocChunk]:
        """
        获取某个父表格的所有嵌套表格 chunks。

        参数：
            parent_chunk_id: 父表格 chunk ID

        返回：
            嵌套表格 chunk 列表
        """
        session = get_session()
        try:
            # 方式1：根据 chunk_id 前缀查询
            prefix = f"{parent_chunk_id}_nested_"
            chunks = session.query(DocChunk).filter(
                DocChunk.chunk_id.like(f"{prefix}%")
            ).all()
            if chunks:
                return chunks

            # 方式2：根据 metadata 查询
            chunks = session.query(DocChunk).filter(
                DocChunk.chunk_type == "nested_table",
                DocChunk.metadata.contains(f'"parent_chunk_id": "{parent_chunk_id}"'),
            ).all()
            return chunks
        finally:
            session.close()


# 单例
chunk_repo = ChunkRepository()
