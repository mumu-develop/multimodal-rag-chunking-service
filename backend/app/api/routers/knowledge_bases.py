"""Knowledge Base management router."""
import structlog
from fastapi import APIRouter, Request

from app.repositories.milvus_repo import milvus_repo
from app.repositories.chunk_repo import chunk_repo
from app.repositories.kb_repo import kb_repo
from app.repositories.document_repo import doc_repo
from app.repositories.faq_repo import faq_repo
from app.schemas import (
    BaseResponse,
    KnowledgeBaseCreate,
    KnowledgeBaseItem,
    KnowledgeBaseListData,
    KnowledgeBaseListResponse,
    KnowledgeBaseDetailResponse,
    KnowledgeBaseUpdate,
)

logger = structlog.get_logger()
router = APIRouter()


@router.post("", response_model=KnowledgeBaseDetailResponse)
async def create_kb(body: KnowledgeBaseCreate, request: Request):
    """创建知识库（MySQL + Milvus collection）。"""
    trace_id = _get_trace_id(request)

    # 1. MySQL 创建记录
    record = kb_repo.create(
        name=body.name,
        collection_name=body.collection_name,
        owner=body.owner,
        description=body.description,
        kb_type=body.kb_type,
    )
    if not record:
        msg = kb_repo.last_error or "创建知识库失败"
        return _err(trace_id, msg, "DB_ERROR")

    # 2. Milvus 创建 collection
    collection_name = record.collection_name
    try:
        milvus_ok = milvus_repo.create_collection_if_not_exists(collection_name)
        if not milvus_ok:
            # Milvus 失败时回滚 MySQL
            kb_repo.delete(record.id)
            return _err(trace_id, milvus_repo.last_error or "创建 Milvus collection 失败", "MILVUS_ERROR")
    except ValueError as e:
        # Collection 名称格式错误，回滚 MySQL
        kb_repo.delete(record.id)
        return _err(trace_id, str(e), "INVALID_COLLECTION_NAME")
    except Exception as e:
        # 其他异常，回滚 MySQL
        kb_repo.delete(record.id)
        return _err(trace_id, str(e), "MILVUS_ERROR")

    return _ok(trace_id, data=_kb_to_item(record))


@router.get("", response_model=KnowledgeBaseListResponse)
async def list_kbs(
    page: int = 1,
    size: int = 20,
    keyword: str = "",
    owner: str = "",
    kb_type: str = "",
    request: Request = None,
):
    """知识库列表（分页/搜索）。"""
    trace_id = _get_trace_id(request)
    total, items = kb_repo.list(page=page, size=size, keyword=keyword, owner=owner, kb_type=kb_type)
    return _ok(trace_id, data=KnowledgeBaseListData(
        total=total,
        page=page,
        result=[_kb_to_item(item) for item in items],
    ))


@router.get("/{kb_id}", response_model=KnowledgeBaseDetailResponse)
async def get_kb(kb_id: int, request: Request):
    """知识库详情。"""
    trace_id = _get_trace_id(request)
    record = kb_repo.get_by_id(kb_id)
    if not record:
        return _err(trace_id, "知识库不存在", "NOT_FOUND")
    return _ok(trace_id, data=_kb_to_item(record))


@router.put("/{kb_id}", response_model=BaseResponse)
async def update_kb(kb_id: int, body: KnowledgeBaseUpdate, request: Request):
    """编辑知识库。"""
    trace_id = _get_trace_id(request)
    updates = body.model_dump(exclude_none=True)
    if not updates:
        return _err(trace_id, "无更新字段", "BAD_REQUEST")
    ok = kb_repo.update(kb_id, **updates)
    if not ok:
        msg = kb_repo.last_error or "更新失败或知识库不存在"
        return _err(trace_id, msg, "DB_ERROR")
    return _ok(trace_id)


@router.delete("/{kb_id}", response_model=BaseResponse)
async def delete_kb(kb_id: int, request: Request):
    """删除知识库（MySQL documents + chunks + faqs + Milvus collection）。"""
    trace_id = _get_trace_id(request)

    # 1. 检查知识库是否存在
    record = kb_repo.get_by_id(kb_id)
    if not record:
        return _err(trace_id, "知识库不存在", "NOT_FOUND")

    collection_name = record.collection_name

    # 2. 删除 MySQL chunks
    deleted_chunks = chunk_repo.delete_by_kb(str(kb_id))

    # 3. 删除 MySQL faqs（如果是FAQ知识库）
    deleted_faqs = 0
    if record.kb_type in ("faq", "hybrid"):
        deleted_faqs = faq_repo.delete_by_kb(kb_id)

    # 4. 删除 Milvus collection
    milvus_repo.drop_collection_by_name(collection_name)

    # 5. 删除知识库记录
    ok = kb_repo.delete(kb_id)
    if not ok:
        return _err(trace_id, kb_repo.last_error or "删除失败", "DB_ERROR")

    return _ok(trace_id, data={
        "deleted_kb_id": kb_id,
        "deleted_chunks": deleted_chunks,
        "deleted_faqs": deleted_faqs,
        "dropped_collection": collection_name,
    })


@router.post("/{kb_id}/fix-collection", response_model=BaseResponse)
async def fix_kb_collection(kb_id: int, request: Request):
    """修复知识库的 Milvus collection（如果不存在则创建）。"""
    trace_id = _get_trace_id(request)

    # 1. 检查知识库是否存在
    record = kb_repo.get_by_id(kb_id)
    if not record:
        return _err(trace_id, "知识库不存在", "NOT_FOUND")

    collection_name = record.collection_name

    # 2. 检查 Milvus collection 是否存在
    client = milvus_repo._get_client()
    existing = client.list_collections()

    if collection_name in existing:
        return _ok(trace_id, data={
            "collection_name": collection_name,
            "status": "already_exists",
            "message": f"Collection '{collection_name}' 已存在",
        })

    # 3. 创建 collection
    ok = milvus_repo.create_collection_if_not_exists(collection_name)
    if not ok:
        return _err(trace_id, f"创建 Collection '{collection_name}' 失败", "MILVUS_ERROR")

    return _ok(trace_id, data={
        "collection_name": collection_name,
        "status": "created",
        "message": f"Collection '{collection_name}' 已创建",
    })


def _kb_to_item(record) -> KnowledgeBaseItem:
    # 获取文档数量
    doc_count = doc_repo.count_by_kb(record.id)
    # 获取FAQ数量（仅faq和hybrid类型）
    faq_count = 0
    if record.kb_type in ("faq", "hybrid"):
        faq_count = faq_repo.count_by_kb(record.id)
    return KnowledgeBaseItem(
        id=record.id,
        name=record.name,
        collection_name=record.collection_name,
        owner=record.owner,
        description=record.description or "",
        kb_type=record.kb_type or "doc",
        is_enabled=record.is_enabled,
        doc_count=doc_count,
        faq_count=faq_count,
        gmt_create=record.gmt_create.isoformat() if record.gmt_create else "",
        gmt_modified=record.gmt_modified.isoformat() if record.gmt_modified else "",
    )


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