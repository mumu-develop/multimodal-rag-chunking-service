"""Repository: MySQL faq_items data access."""
from __future__ import annotations

import uuid
from typing import Optional

import structlog

from app.models import FAQItem, get_session

logger = structlog.get_logger()


class FAQRepository:
    """MySQL faq_items table repository."""

    def __init__(self):
        self.last_error: str = ""

    def create(
        self,
        kb_id: int,
        question: str,
        answer: str,
        category: str = "",
        keywords: str = "",
        source: str = "manual",
        sort_order: int = 0,
        creator: str = None,
    ) -> Optional[FAQItem]:
        """创建FAQ问答对。"""
        session = get_session()
        try:
            faq_id = uuid.uuid4().hex
            record = FAQItem(
                faq_id=faq_id,
                kb_id=kb_id,
                question=question,
                answer=answer,
                category=category,
                keywords=keywords,
                source=source,
                sort_order=sort_order,
                creator=creator,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.info("faq_created", faq_id=faq_id, kb_id=kb_id)
            self.last_error = ""
            return record
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("faq_create_failed", kb_id=kb_id, error=self.last_error)
            return None
        finally:
            session.close()

    @staticmethod
    def get_by_id(faq_id: str) -> Optional[FAQItem]:
        """根据 faq_id 获取FAQ。"""
        session = get_session()
        try:
            return session.query(FAQItem).filter(FAQItem.faq_id == faq_id).first()
        finally:
            session.close()

    @staticmethod
    def get_by_pk(id: int) -> Optional[FAQItem]:
        """根据主键 id 获取FAQ。"""
        session = get_session()
        try:
            return session.query(FAQItem).filter(FAQItem.id == id).first()
        finally:
            session.close()

    @staticmethod
    def list(
        kb_id: int,
        page: int = 1,
        size: int = 20,
        category: str = "",
        is_enabled: int = None,
    ) -> tuple[int, list[FAQItem]]:
        """分页查询FAQ列表。"""
        session = get_session()
        try:
            query = session.query(FAQItem).filter(FAQItem.kb_id == kb_id)
            if category:
                query = query.filter(FAQItem.category == category)
            if is_enabled is not None:
                query = query.filter(FAQItem.is_enabled == is_enabled)

            total = query.count()
            items = (
                query.order_by(FAQItem.sort_order, FAQItem.id.desc())
                .offset((page - 1) * size)
                .limit(size)
                .all()
            )
            return total, items
        finally:
            session.close()

    def update(self, faq_id: str, **kwargs) -> bool:
        """更新FAQ。"""
        session = get_session()
        try:
            row = session.query(FAQItem).filter(FAQItem.faq_id == faq_id).update(kwargs)
            session.commit()
            logger.info("faq_updated", faq_id=faq_id, row_count=row)
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("faq_update_failed", faq_id=faq_id, error=self.last_error)
            return False
        finally:
            session.close()

    def delete(self, faq_id: str) -> bool:
        """删除FAQ。"""
        session = get_session()
        try:
            row = session.query(FAQItem).filter(FAQItem.faq_id == faq_id).delete()
            session.commit()
            logger.info("faq_deleted", faq_id=faq_id, row_count=row)
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("faq_delete_failed", faq_id=faq_id, error=self.last_error)
            return False
        finally:
            session.close()

    def batch_delete(self, faq_ids: list[str]) -> int:
        """批量删除FAQ。"""
        if not faq_ids:
            return 0
        session = get_session()
        try:
            rows = session.query(FAQItem).filter(FAQItem.faq_id.in_(faq_ids)).delete(synchronize_session=False)
            session.commit()
            logger.info("faq_batch_deleted", count=rows)
            self.last_error = ""
            return rows
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("faq_batch_delete_failed", count=len(faq_ids), error=self.last_error)
            return 0
        finally:
            session.close()

    def batch_import(
        self,
        kb_id: int,
        items: list[dict],
        source: str = "import",
        creator: str = None,
    ) -> tuple[int, list[dict]]:
        """批量导入FAQ。

        返回：(成功数量, 结果列表)
        """
        if not items:
            return 0, []

        session = get_session()
        success = 0
        results = []

        try:
            for item in items:
                faq_id = uuid.uuid4().hex
                try:
                    record = FAQItem(
                        faq_id=faq_id,
                        kb_id=kb_id,
                        question=item["question"],
                        answer=item["answer"],
                        category=item.get("category", ""),
                        keywords=item.get("keywords", ""),
                        source=source,
                        creator=creator,
                    )
                    session.add(record)
                    success += 1
                    results.append({"faq_id": faq_id, "success": True})
                except Exception as e:
                    results.append({"question": item["question"][:50], "success": False, "error": str(e)})

            session.commit()
            logger.info("faq_batch_import_done", kb_id=kb_id, success=success, total=len(items))
            self.last_error = ""
            return success, results
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("faq_batch_import_failed", kb_id=kb_id, error=self.last_error)
            return 0, [{"success": False, "error": str(e)}]
        finally:
            session.close()

    @staticmethod
    def get_by_ids(faq_ids: list[str]) -> list[FAQItem]:
        """批量获取FAQ（检索后回查）。"""
        if not faq_ids:
            return []
        session = get_session()
        try:
            return session.query(FAQItem).filter(FAQItem.faq_id.in_(faq_ids)).all()
        finally:
            session.close()

    def increment_hit_count(self, faq_id: str) -> bool:
        """增加命中次数。"""
        session = get_session()
        try:
            session.query(FAQItem).filter(FAQItem.faq_id == faq_id).update(
                {FAQItem.hit_count: FAQItem.hit_count + 1}
            )
            session.commit()
            return True
        except Exception:
            session.rollback()
            logger.exception("faq_hit_count_failed", faq_id=faq_id)
            return False
        finally:
            session.close()

    @staticmethod
    def count_by_kb(kb_id: int) -> int:
        """统计知识库的FAQ数量。"""
        session = get_session()
        try:
            return session.query(FAQItem).filter(FAQItem.kb_id == kb_id, FAQItem.is_enabled == 1).count()
        finally:
            session.close()

    def delete_by_kb(self, kb_id: int) -> int:
        """删除知识库的所有FAQ。"""
        session = get_session()
        try:
            rows = session.query(FAQItem).filter(FAQItem.kb_id == kb_id).delete(synchronize_session=False)
            session.commit()
            logger.info("faq_deleted_by_kb", kb_id=kb_id, count=rows)
            self.last_error = ""
            return rows
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("faq_delete_by_kb_failed", kb_id=kb_id, error=self.last_error)
            return 0
        finally:
            session.close()


# 单例
faq_repo = FAQRepository()