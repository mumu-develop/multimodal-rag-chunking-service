"""Task status router."""
import structlog
from fastapi import APIRouter, Request

from app.repositories.task_repo import task_repo
from app.schemas import (
    SyncedDocsListResponse,
    TaskDetailData,
    TaskDetailResponse,
    TaskListResponse,
    TaskStatusResponse,
)
from app.services.task_service import task_service
from app.repositories.task_repo import task_repo

logger = structlog.get_logger()

router = APIRouter()

_MAX_PAGE_SIZE = 100


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    request: Request,
    page: int = 1,
    size: int = 20,
    status: str | None = None,
):
    """查询任务列表（分页）。"""
    if page < 1:
        page = 1
    if size < 1:
        size = 20
    if size > _MAX_PAGE_SIZE:
        size = _MAX_PAGE_SIZE

    trace_id = _get_trace_id(request)
    try:
        total, items = task_service.list_tasks(page, size, status)
        return _ok(trace_id, data={"total": total, "page": page, "result": items})
    except Exception as exc:
        logger.exception("list_tasks_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Failed to list tasks", code="LIST_TASKS_ERROR")


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, request: Request):
    trace_id = _get_trace_id(request)
    task = task_service.get_task(task_id)
    if not task:
        return _err(trace_id=trace_id, message="Task not found", code="NOT_FOUND")
    return _ok(trace_id, data=task)


@router.get("/{task_id}/detail", response_model=TaskDetailResponse)
async def get_task_detail(task_id: str, request: Request):
    """增强版任务详情（含流水线节点 + 任务参数）。"""
    trace_id = _get_trace_id(request)
    task = task_repo.get(task_id)
    if not task:
        return _err(trace_id, "Task not found", "NOT_FOUND")

    return _ok(trace_id, data=TaskDetailData(
        task_id=task.task_id,
        doc_id=task.doc_id,
        doc_name=task.doc_name or "",
        doc_type=task.doc_type or "",
        status=task.status,
        progress=task.progress or 0,
        stage=task.stage,
        error_message=task.error_message,
        chunk_count=task.chunk_count or 0,
        pipeline_nodes=task.pipeline_nodes or [],
        task_params=task.task_params,
        gmt_create=task.gmt_create.isoformat() if task.gmt_create else "",
        gmt_modified=task.gmt_modified.isoformat() if task.gmt_modified else "",
    ))


@router.post("/{task_id}/retry")
async def retry_task(task_id: str, request: Request):
    """手动重试失败的任务。"""
    trace_id = _get_trace_id(request)

    # 获取原任务信息
    task = task_repo.get(task_id)
    if not task:
        return _err(trace_id, "Task not found", "NOT_FOUND")

    if task.status not in ("failed", "error"):
        return _err(trace_id, f"Only failed tasks can be retried (current: {task.status})", "INVALID_STATUS")

    # 获取原任务参数
    task_params = task.task_params or {}

    try:
        from app.tasks import parse_and_chunk_task

        # 重新提交任务，使用原参数
        file_path = task_params.get("file_path", "")
        doc_id = task.doc_id
        creator = task_params.get("creator", "")
        kb_id = task_params.get("kb_id", "")
        collection_name = task_params.get("collection_name", "")
        source_type = task_params.get("source_type", "local_file")
        source_url = task_params.get("source_url", "")
        doc_name = task_params.get("doc_name", "")
        file_size = task_params.get("file_size", 0)
        doc_type = task_params.get("doc_type", "")
        chunk_strategy = task_params.get("chunk_strategy", "smart")

        # 检查文件是否存在
        import os
        if file_path and not os.path.exists(file_path):
            return _err(trace_id, f"Source file not found: {file_path}", "FILE_NOT_FOUND")

        # 创建新任务
        new_task = parse_and_chunk_task.apply_async(
            args=[
                file_path, doc_id, creator, kb_id, collection_name,
                source_type, source_url, doc_name, file_size, doc_type,
                None, chunk_strategy
            ]
        )

        # 更新原任务状态为 retrying
        task_repo.upsert(task_id, status="retrying", error_message=None)

        logger.info("task_retried", old_task_id=task_id, new_task_id=new_task.id)

        return _ok(trace_id, data={
            "old_task_id": task_id,
            "new_task_id": new_task.id,
            "doc_id": doc_id,
            "message": "Task retry submitted"
        })

    except Exception as exc:
        logger.exception("retry_task_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id, f"Retry failed: {str(exc)}", "RETRY_ERROR")


@router.get("/synced-docs", response_model=SyncedDocsListResponse)
async def list_synced_docs(
    request: Request,
    page: int = 1,
    size: int = 20,
    doc_id: str = None,
):
    """查询同步成功的文档列表（分页）。"""
    if page < 1:
        page = 1
    if size < 1:
        size = 20
    if size > _MAX_PAGE_SIZE:
        size = _MAX_PAGE_SIZE

    trace_id = _get_trace_id(request)
    try:
        total, items = task_repo.list_by_status(status="completed", page=page, size=size, doc_id_filter=doc_id)
        result = [
            {
                "doc_id": t.doc_id,
                "task_id": t.task_id,
                "doc_type": t.doc_type or "",
                "chunk_count": t.chunk_count or 0,
                "gmt_create": str(t.gmt_create or ""),
                "gmt_modified": str(t.gmt_modified or ""),
            }
            for t in items
        ]
        return _ok(trace_id, data={"total": total, "page": page, "result": result})
    except Exception as exc:
        logger.exception("list_synced_docs_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Failed to list synced docs", code="LIST_DOCS_ERROR")


def _get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "")


def _ok(trace_id: str = "", data=None, code: str = "OK"):
    # 如果 data 是 Pydantic 模型，转换为 dict
    if data is not None and hasattr(data, 'model_dump'):
        data = data.model_dump()
    elif data is not None and hasattr(data, 'dict'):
        data = data.dict()
    return {"success": True, "code": code, "message": None, "trace_id": trace_id, "data": data}


def _err(trace_id: str = "", message: str = "Error", code: str = "ERROR"):
    return {"success": False, "code": code, "message": message, "trace_id": trace_id, "data": None}
