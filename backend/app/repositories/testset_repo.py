"""Repository: MySQL test_set data access."""
from typing import Optional

import structlog

from app.models import TestSet, get_session

logger = structlog.get_logger()


class TestSetRepository:
    """MySQL test_set table repository."""

    def __init__(self):
        self.last_error: str = ""

    def batch_insert(self, items: list[dict]) -> bool:
        """批量插入测试集记录。

        Args:
            items: 每条记录包含 set_id, set_name, query, ground_truth, doc_id, tags 的字典列表。

        Returns:
            成功返回 True，失败返回 False。
        """
        session = get_session()
        try:
            for item in items:
                record = TestSet(
                    set_id=item["set_id"],
                    set_name=item["set_name"],
                    query=item["query"],
                    ground_truth=item["ground_truth"],
                    doc_id=item.get("doc_id", ""),
                    tags=item.get("tags", []),
                )
                session.add(record)
            session.commit()
            logger.info("test_set_insert_ok", count=len(items))
            self.last_error = ""
            return True
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("test_set_insert_failed", count=len(items), error=self.last_error)
            return False
        finally:
            session.close()

    @staticmethod
    def has_records_for(set_id: str, doc_id: str) -> bool:
        """检查 set_id + doc_id 是否已有记录（用于重试去重）。"""
        session = get_session()
        try:
            return (
                session.query(TestSet)
                .filter(TestSet.set_id == set_id, TestSet.doc_id == doc_id)
                .first()
                is not None
            )
        finally:
            session.close()

    @staticmethod
    def list_sets(
        page: int = 1,
        size: int = 20,
        doc_id_filter: Optional[str] = None,
        set_id_filter: Optional[str] = None,
        set_name_filter: Optional[str] = None,
    ) -> tuple[int, list[TestSet]]:
        """分页查询测试集记录。

        Returns:
            (total_count, [TestSet, ...])
        """
        session = get_session()
        try:
            q = session.query(TestSet)
            if doc_id_filter:
                q = q.filter(TestSet.doc_id == doc_id_filter)
            if set_id_filter:
                q = q.filter(TestSet.set_id == set_id_filter)
            if set_name_filter:
                q = q.filter(TestSet.set_name.contains(set_name_filter))

            total = q.count()
            items = q.order_by(TestSet.gmt_create.desc()).offset((page - 1) * size).limit(size).all()
            return total, items
        finally:
            session.close()

    @staticmethod
    def list_set_names() -> list[dict]:
        """查询所有测试集批次名称（去重，按创建时间倒序）。

        Returns:
            [{"set_id": ..., "set_name": ..., "gmt_create": ...}, ...]
        """
        from sqlalchemy import func as sql_func
        session = get_session()
        try:
            # 按 set_id 分组，取最早的 gmt_create
            results = session.query(
                TestSet.set_id,
                TestSet.set_name,
                sql_func.min(TestSet.gmt_create).label("gmt_create"),
            ).group_by(TestSet.set_id, TestSet.set_name).order_by(
                sql_func.min(TestSet.gmt_create).desc()
            ).all()

            return [
                {
                    "set_id": r.set_id,
                    "set_name": r.set_name,
                    "gmt_create": r.gmt_create.isoformat() if r.gmt_create else "",
                }
                for r in results
            ]
        finally:
            session.close()


# 单例
testset_repo = TestSetRepository()
