"""Search router: text search and image search."""
import json
import os

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.repositories.kb_repo import kb_repo
from app.schemas import SearchRequest, SearchResponse
from app.services.async_search_service import async_search_service

logger = structlog.get_logger()

router = APIRouter()

# 允许的查询图片目录白名单（防止路径遍历）
_ALLOWED_QUERY_DIRS = [
    os.path.realpath(os.path.expanduser("~")),  # 用户主目录
]


def _validate_kb(kb_id: str) -> str:
    """验证 KB 存在并返回 collection_name。"""
    if not kb_id:
        raise HTTPException(status_code=400, detail="kb_id is required")
    kb = kb_repo.get_by_id(int(kb_id))
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")
    return kb.collection_name


def _validate_image_path(path: str) -> str:
    """验证查询图片路径是否在允许范围内。"""
    real_path = os.path.realpath(os.path.expanduser(path))
    if not os.path.isfile(real_path):
        raise HTTPException(status_code=400, detail="Image file not found")
    # 检查是否为系统敏感路径
    for forbidden in ("/etc/", "/proc/", "/sys/", "/dev/"):
        if real_path.startswith(forbidden):
            raise HTTPException(status_code=403, detail="Access denied")
    return real_path


@router.post("", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request):
    """搜索指定知识库（kb_id 或 collection_name 必填）。"""
    trace_id = _get_trace_id(request)

    try:
        # 必须指定 kb_id 或 collection_name
        collection_name = body.collection_name
        if not collection_name:
            collection_name = _validate_kb(body.kb_id)

        is_image_query = body.is_image_query()
        if not is_image_query and not body.query:
            return _err(trace_id=trace_id, message="query or query_image is required", code="INVALID_REQUEST")

        safe_image_path = None
        if is_image_query:
            safe_image_path = _validate_image_path(body.query_image)

        hits = await async_search_service.search(
            query=body.query,
            query_image=safe_image_path,
            top_k=body.top_k,
            modal=body.modal,
            filters=body.filters,
            doc_ids=body.doc_ids,
            collection_name=collection_name,
            use_rerank=body.use_rerank,
            data_type=body.data_type,
        )
        logger.info("search_in_collection", collection=collection_name, kb_id=body.kb_id, query=body.query[:50], data_type=body.data_type)
        return _ok(trace_id, data={"hits": hits})

    except HTTPException:
        raise
    except ValueError as exc:
        # Milvus collection 不存在
        logger.error("search_collection_not_found", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message=str(exc), code="COLLECTION_NOT_FOUND")
    except Exception as exc:
        logger.exception("search_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Search failed", code="SEARCH_ERROR")


@router.post("/image", response_model=SearchResponse)
async def search_by_image(
    request: Request,
    image: UploadFile = File(...),
    query: str | None = Form(None),
    top_k: int = Form(10),
    modal: str = Form("both"),
    filters: str | None = Form(None),
    doc_ids: str | None = Form(None),
    kb_id: str = Form(""),  # 必填
    collection_name: str | None = Form(None),
    use_rerank: bool = Form(True),
    data_type: str = Form(""),  # 数据类型过滤
):
    """以图搜图 / 以文搜图 — 上传图片作为查询（kb_id 必填）。"""
    trace_id = _get_trace_id(request)

    try:
        # 必须指定 kb_id 或 collection_name
        if not collection_name:
            collection_name = _validate_kb(kb_id)

        parsed_doc_ids = None
        if doc_ids:
            parsed_doc_ids = [d.strip() for d in doc_ids.split(",") if d.strip()]

        parsed_filters = None
        if filters:
            try:
                parsed_filters = json.loads(filters)
            except json.JSONDecodeError:
                return _err(trace_id=trace_id, message="Invalid filters JSON", code="INVALID_REQUEST")

        hits = await async_search_service.search_by_image(
            image=image,
            query=query,
            top_k=top_k,
            modal=modal,
            filters=parsed_filters,
            doc_ids=parsed_doc_ids,
            collection_name=collection_name,
            use_rerank=use_rerank,
        )
        logger.info("image_search_in_collection", collection=collection_name, kb_id=kb_id, data_type=data_type)
        return _ok(trace_id, data={"hits": hits})

    except HTTPException:
        raise
    except ValueError as exc:
        # Milvus collection 不存在
        logger.error("search_collection_not_found", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message=str(exc), code="COLLECTION_NOT_FOUND")
    except Exception as exc:
        logger.exception("search_by_image_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Search failed", code="SEARCH_ERROR")


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
