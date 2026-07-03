"""Service: Task status management."""
from typing import Optional

from app.repositories.task_repo import task_repo


class TaskService:
    """任务状态查询服务。"""

    def get_task(self, task_id: str) -> Optional[dict]:
        """获取任务状态。"""
        task = task_repo.get(task_id)
        if not task:
            return None

        return {
            "status": task.status,
            "progress": task.progress or 0,
            "chunk_count": task.chunk_count or 0,
            "stage": task.stage,
            "error_message": task.error_message,
            "doc_type": getattr(task, "doc_type", None),
        }

    def list_tasks(
        self,
        page: int = 1,
        size: int = 20,
        status_filter: Optional[str] = None,
    ) -> tuple[int, list[dict]]:
        """获取任务列表（分页）。"""
        total, items = task_repo.list_all(page, size, status_filter)
        result = [
            {
                "task_id": t.task_id,
                "doc_id": t.doc_id,
                "status": t.status,
                "progress": t.progress or 0,
                "chunk_count": t.chunk_count or 0,
                "stage": t.stage,
                "error_message": t.error_message,
                "doc_type": getattr(t, "doc_type", None),
                "gmt_create": str(t.gmt_create or ""),
                "gmt_modified": str(t.gmt_modified or ""),
            }
            for t in items
        ]
        return total, result


# 单例
task_service = TaskService()
