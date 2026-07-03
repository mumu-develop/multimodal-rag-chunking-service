"""Upload router."""
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.repositories.kb_repo import kb_repo
from app.schemas import BatchUploadResponse, BatchUploadResponseData, UploadResponse, UploadResponseData
from app.services.document_service import document_service

logger = structlog.get_logger()

router = APIRouter()


def get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "")


def _validate_kb(kb_id: str) -> str:
    """验证 KB 存在并返回 collection_name。"""
    if not kb_id:
        raise HTTPException(status_code=400, detail="kb_id is required")
    kb = kb_repo.get_by_id(int(kb_id))
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")
    return kb.collection_name


@router.post("", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    kb_id: str = Form(...),  # 必填
    source_type: str = Form(default="local_file"),
    source_url: str = Form(default=""),
    chunk_strategy: str = Form(default="smart", description="切分策略：smart/heading/semantic/recursive/fixed"),
    trace_id: str = Depends(get_trace_id),
):
    """上传文件到指定知识库（kb_id 必填）。

    Args:
        chunk_strategy: 切分策略选择
            - smart: 智能切分（标题+语义+递归）- 最高质量，默认
            - heading: 只按标题切分 - 保持文档结构
            - semantic: 只按语义切分 - 保持语义连贯
            - recursive: 只按递归切分 - 最简单快速
            - fixed: 固定长度切分 - 最基础方案
    """
    try:
        collection_name = _validate_kb(kb_id)
        config = {"chunk_strategy": chunk_strategy}
        data = document_service.upload(
            file, kb_id=kb_id, source_type=source_type, source_url=source_url,
            config=config,
        )
        if data.get("error"):
            return _err(trace_id=trace_id, message=data["error"], code="UPLOAD_ERROR")
        logger.info("upload_to_kb", kb_id=kb_id, collection=collection_name, doc_id=data.get("doc_id"), chunk_strategy=chunk_strategy)
        return _ok(trace_id, data=UploadResponseData(**data))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("upload_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Upload failed", code="UPLOAD_ERROR")


@router.post("/batch", response_model=BatchUploadResponse)
async def upload_batch(
    files: list[UploadFile] = File(...),
    kb_id: str = Form(...),  # 必填
    source_type: str = Form(default="local_file"),
    chunk_strategy: str = Form(default="smart", description="切分策略：smart/heading/semantic/recursive/fixed"),
    trace_id: str = Depends(get_trace_id),
):
    """批量上传文件到指定知识库（kb_id 必填）。

    Args:
        chunk_strategy: 切分策略选择（应用于所有文件）
    """
    try:
        collection_name = _validate_kb(kb_id)
        config = {"chunk_strategy": chunk_strategy}
        data = document_service.upload_batch(files, kb_id=kb_id, source_type=source_type, config=config)
        logger.info("batch_upload_to_kb", kb_id=kb_id, collection=collection_name,
                    total=data["total"], success=data["success"], failed=data["failed"], chunk_strategy=chunk_strategy)
        return _ok(trace_id, data=BatchUploadResponseData(**data))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("batch_upload_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Batch upload failed", code="BATCH_UPLOAD_ERROR")


@router.post("/url", response_model=UploadResponse)
async def upload_url(
    url: str = Form(...),
    kb_id: str = Form(...),  # 必填
    doc_name: str = Form(default=""),
    trace_id: str = Depends(get_trace_id),
):
    """从 URL 下载文档并提交处理（kb_id 必填）。"""
    try:
        collection_name = _validate_kb(kb_id)
        from app.tasks import enqueue_fetch_url
        task_id = enqueue_fetch_url(url, kb_id=kb_id, doc_name=doc_name)
        logger.info("url_upload_to_kb", kb_id=kb_id, collection=collection_name, task_id=task_id)
        return _ok(trace_id, data={"task_id": task_id, "doc_id": "", "doc_type": ""})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("url_upload_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="URL upload failed", code="URL_UPLOAD_ERROR")


def _ok(trace_id: str = "", data=None, code: str = "OK"):
    # 如果 data 是 Pydantic 模型，转换为 dict
    if data is not None and hasattr(data, 'model_dump'):
        data = data.model_dump()
    elif data is not None and hasattr(data, 'dict'):
        data = data.dict()
    return {"success": True, "code": code, "message": None, "trace_id": trace_id, "data": data}


def _err(trace_id: str = "", message: str = "Error", code: str = "ERROR"):
    return {"success": False, "code": code, "message": message, "trace_id": trace_id, "data": None}
