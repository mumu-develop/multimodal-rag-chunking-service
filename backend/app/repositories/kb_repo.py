"""Repository: MySQL knowledge_base data access."""
from typing import Optional

import structlog

from app.models import KnowledgeBase, get_session

logger = structlog.get_logger()


class KnowledgeBaseRepository:
    """MySQL knowledge_bases table repository."""

    def __init__(self):
        self.last_error: str = ""

    def create(self, name: str, collection_name: str,
               owner: str, description: str = "", kb_type: str = "doc") -> Optional[KnowledgeBase]:
        session = get_session()
        try:
            record = KnowledgeBase(
                name=name,
                collection_name=collection_name,
                owner=owner,
                description=description,
                kb_type=kb_type,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.info("kb_created", id=record.id, name=name, kb_type=kb_type)
            self.last_error = ""
            return record
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("kb_create_failed", name=name, error=self.last_error)
            return None
        finally:
            session.close()

    @staticmethod
    def list(page: int = 1, size: int = 20, keyword: str = "",
             owner: str = "", kb_type: str = "") -> tuple[int, list[KnowledgeBase]]:
        session = get_session()
        try:
            q = session.query(KnowledgeBase)
            if keyword:
                q = q.filter(KnowledgeBase.name.contains(keyword))
            if owner:
                q = q.filter(KnowledgeBase.owner == owner)
            if kb_type:
                q = q.filter(KnowledgeBase.kb_type == kb_type)
            total = q.count()
            items = q.order_by(KnowledgeBase.gmt_create.desc()) \
                     .offset((page - 1) * size).limit(size).all()
            return total, items
        finally:
            session.close()

    @staticmethod
    def get_by_id(kb_id: int) -> Optional[KnowledgeBase]:
        session = get_session()
        try:
            return session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        finally:
            session.close()

    def update(self, kb_id: int, **kwargs) -> bool:
        session = get_session()
        try:
            row = session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).update(kwargs)
            session.commit()
            logger.info("kb_updated", id=kb_id, row_count=row)
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("kb_update_failed", id=kb_id, error=self.last_error)
            return False
        finally:
            session.close()

    def delete(self, kb_id: int) -> bool:
        session = get_session()
        try:
            row = session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).delete()
            session.commit()
            logger.info("kb_deleted", id=kb_id, row_count=row)
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("kb_delete_failed", id=kb_id, error=self.last_error)
            return False
        finally:
            session.close()


# 单例
kb_repo = KnowledgeBaseRepository()
