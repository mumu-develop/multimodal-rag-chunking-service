"""Repository: MySQL tasks data access."""
from typing import Optional

import structlog

from app.models import TaskStatus, get_session

logger = structlog.get_logger()


class TaskRepository:
    """MySQL tasks table repository."""

    def __init__(self):
        self.last_error: str = ""

    def upsert(self, task_id: str, **kwargs) -> None:
        """插入或更新任务记录。"""
        session = get_session()
        try:
            task = session.query(TaskStatus).filter(TaskStatus.task_id == task_id).first()
            if task:
                for k, v in kwargs.items():
                    setattr(task, k, v)
            else:
                task = TaskStatus(task_id=task_id, **kwargs)
                session.add(task)
            session.commit()
            self.last_error = ""
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("task_upsert_failed", task_id=task_id, error=self.last_error)
        finally:
            session.close()

    @staticmethod
    def get(task_id: str) -> Optional[TaskStatus]:
        """获取任务记录。"""
        session = get_session()
        try:
            return session.query(TaskStatus).filter(TaskStatus.task_id == task_id).first()
        finally:
            session.close()

    @staticmethod
    def list_all(
        page: int = 1,
        size: int = 20,
        status_filter: Optional[str] = None,
    ) -> tuple[int, list[TaskStatus]]:
        """查询所有任务列表（分页），可按状态过滤。

        Returns:
            (total_count, [TaskStatus, ...])
        """
        session = get_session()
        try:
            q = session.query(TaskStatus)
            if status_filter:
                q = q.filter(TaskStatus.status == status_filter)
            total = q.count()
            items = (
                q.order_by(TaskStatus.gmt_modified.desc())
                .offset((page - 1) * size)
                .limit(size)
                .all()
            )
            return total, items
        finally:
            session.close()

    @staticmethod
    def list_by_status(
        status: str = "completed",
        page: int = 1,
        size: int = 20,
        doc_id_filter: Optional[str] = None,
    ) -> tuple[int, list[TaskStatus]]:
        """按状态查询任务列表（分页）。

        Returns:
            (total_count, [TaskStatus, ...])
        """
        session = get_session()
        try:
            q = session.query(TaskStatus).filter(TaskStatus.status == status)
            if doc_id_filter:
                q = q.filter(TaskStatus.doc_id == doc_id_filter)
            total = q.count()
            items = (
                q.order_by(TaskStatus.gmt_modified.desc())
                .offset((page - 1) * size)
                .limit(size)
                .all()
            )
            return total, items
        finally:
            session.close()

    def update_pipeline_nodes(self, task_id: str, nodes: list[dict]) -> bool:
        """更新任务的流水线节点执行详情。"""
        session = get_session()
        try:
            row = session.query(TaskStatus).filter(TaskStatus.task_id == task_id) \
                .update({"pipeline_nodes": nodes})
            session.commit()
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("pipeline_nodes_update_failed", task_id=task_id, error=self.last_error)
            return False
        finally:
            session.close()

    def update_task_params(self, task_id: str, params: dict) -> bool:
        """更新任务参数快照。"""
        session = get_session()
        try:
            row = session.query(TaskStatus).filter(TaskStatus.task_id == task_id) \
                .update({"task_params": params})
            session.commit()
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("task_params_update_failed", task_id=task_id, error=self.last_error)
            return False
        finally:
            session.close()


# 单例
task_repo = TaskRepository()
