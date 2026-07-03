"""Document management router."""
import structlog
from fastapi import APIRouter, Request

from app.config import settings
from app.repositories.chunk_repo import chunk_repo
from app.repositories.kb_repo import kb_repo
from app.repositories.document_repo import doc_repo
from app.schemas import (
    BaseResponse,
    DocumentBatchDeleteData,
    DocumentBatchDeleteRequest,
    DocumentItem,
    DocumentListData,
    DocumentListResponse,
    DocumentToggleRequest,
)

logger = structlog.get_logger()
router = APIRouter()


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    kb_id: int = 0,
    page: int = 1,
    size: int = 20,
    keyword: str = "",
    status: str = "",
    request: Request = None,
):
    """文档列表（从 documents 表查询，分页/搜索/状态过滤）。"""
    trace_id = _get_trace_id(request)

    if kb_id:
        # 查询指定知识库的文档（实时统计 chunk_count）
        total, items = doc_repo.list_by_kb(
            kb_id=kb_id, page=page, size=size,
            keyword=keyword, status=status,
        )
        # items 已经是 dict 格式，直接转换
        result = [_dict_to_item(item) for item in items]
    else:
        # 查询所有文档（从 chunks 表聚合，兜底）
        total, items = chunk_repo.list_documents(
            page=page, size=size,
            keyword=keyword, status_filter=status,
        )
        # 转换为 DocumentItem 格式
        result = [_chunk_doc_to_item(item) for item in items]

    return _ok(trace_id, data=DocumentListData(
        total=total, page=page, result=result,
    ))


@router.put("/{doc_id}/toggle", response_model=BaseResponse)
async def toggle_document(doc_id: str, body: DocumentToggleRequest, request: Request):
    """启用/禁用文档的所有切片。"""
    trace_id = _get_trace_id(request)
    row = chunk_repo.toggle_all_by_doc(doc_id, body.is_enabled)
    if row == 0:
        msg = chunk_repo.last_error or "文档不存在或无切片"
        return _err(trace_id, msg, "NOT_FOUND")
    return _ok(trace_id)


@router.delete("/{doc_id}", response_model=BaseResponse)
async def delete_document(doc_id: str, request: Request):
    """删除文档及所有切片（MySQL + Milvus 需同步清理）。"""
    trace_id = _get_trace_id(request)

    # 先检查 documents 表是否存在该文档
    doc_record = doc_repo.get_by_doc_id(doc_id)

    # 获取所有切片（可能为空）
    all_chunks = chunk_repo.get_all_by_doc(doc_id)

    # 如果 documents 表没有记录且 chunks 表也没有切片，才算文档不存在
    if not doc_record and not all_chunks:
        return _err(trace_id, "文档不存在", "NOT_FOUND")

    # 确定要删除的切片 ID
    chunk_ids = [c.chunk_id for c in all_chunks] if all_chunks else []

    # 确定 collection_name
    collection_name = settings.MILVUS_COLLECTION_NAME
    if doc_record and doc_record.kb_id:
        kb = kb_repo.get_by_id(doc_record.kb_id)
        if kb:
            collection_name = kb.collection_name
    elif all_chunks:
        first_chunk = all_chunks[0]
        if first_chunk.kb_id:
            kb = kb_repo.get_by_id(int(first_chunk.kb_id))
            if kb:
                collection_name = kb.collection_name

    # 从 Milvus 删除向量（如果有切片）
    if chunk_ids:
        try:
            from app.repositories.milvus_repo import milvus_repo
            milvus_repo.delete(chunk_ids, collection_name=collection_name)
        except Exception:
            logger.exception("milvus_doc_delete_failed", doc_id=doc_id, collection=collection_name)

    # 从 MySQL 删除切片（如果有）
    chunks_deleted = 0
    if chunk_ids:
        chunks_deleted = chunk_repo.batch_delete(chunk_ids)

    # 从 MySQL 删除 documents 表记录（新增：确保 documents 表也被删除）
    doc_deleted = False
    if doc_record:
        doc_deleted = doc_repo.delete(doc_id)

    logger.info(
        "document_deleted",
        doc_id=doc_id,
        chunks_deleted=chunks_deleted,
        doc_record_deleted=doc_deleted,
        collection=collection_name,
    )

    return _ok(trace_id, data={
        "deleted_chunks": chunks_deleted,
        "doc_record_deleted": doc_deleted,
    })


@router.post("/batch-delete", response_model=BaseResponse)
async def batch_delete_documents(body: DocumentBatchDeleteRequest, request: Request):
    """批量删除文档及所有切片（MySQL + Milvus 需同步清理）。"""
    trace_id = _get_trace_id(request)

    if not body.doc_ids:
        return _ok(trace_id, data=DocumentBatchDeleteData(total=0, success=0, failed=0, deleted_chunks=0))

    results = []
    total_deleted_chunks = 0
    success_count = 0
    failed_count = 0

    for doc_id in body.doc_ids:
        try:
            # 先检查 documents 表是否存在该文档
            doc_record = doc_repo.get_by_doc_id(doc_id)

            # 获取所有切片（可能为空）
            all_chunks = chunk_repo.get_all_by_doc(doc_id)

            # 如果 documents 表没有记录且 chunks 表也没有切片，才算文档不存在
            if not doc_record and not all_chunks:
                results.append({"doc_id": doc_id, "success": False, "error": "文档不存在", "deleted_chunks": 0})
                failed_count += 1
                continue

            chunk_ids = [c.chunk_id for c in all_chunks] if all_chunks else []

            # 确定 collection_name
            collection_name = settings.MILVUS_COLLECTION_NAME
            if doc_record and doc_record.kb_id:
                kb = kb_repo.get_by_id(doc_record.kb_id)
                if kb:
                    collection_name = kb.collection_name
            elif all_chunks:
                first_chunk = all_chunks[0]
                if first_chunk.kb_id:
                    kb = kb_repo.get_by_id(int(first_chunk.kb_id))
                    if kb:
                        collection_name = kb.collection_name

            # 从 Milvus 删除向量（如果有切片）
            if chunk_ids:
                try:
                    from app.repositories.milvus_repo import milvus_repo
                    milvus_repo.delete(chunk_ids, collection_name=collection_name)
                except Exception:
                    logger.exception("milvus_batch_doc_delete_failed", doc_id=doc_id, collection=collection_name)

            # 从 MySQL 删除切片（如果有）
            chunks_deleted = 0
            if chunk_ids:
                chunks_deleted = chunk_repo.batch_delete(chunk_ids)
                total_deleted_chunks += chunks_deleted

            # 从 MySQL 删除 documents 表记录
            doc_deleted = False
            if doc_record:
                doc_deleted = doc_repo.delete(doc_id)

            results.append({
                "doc_id": doc_id,
                "success": True,
                "error": None,
                "deleted_chunks": chunks_deleted,
                "doc_record_deleted": doc_deleted,
            })
            success_count += 1
            logger.info(
                "batch_doc_deleted",
                doc_id=doc_id,
                chunks_deleted=chunks_deleted,
                doc_record_deleted=doc_deleted,
            )
        except Exception as exc:
            results.append({"doc_id": doc_id, "success": False, "error": str(exc), "deleted_chunks": 0})
            failed_count += 1
            logger.exception("batch_doc_delete_error", doc_id=doc_id, error=str(exc))

    logger.info("batch_documents_deleted", total=len(body.doc_ids), success=success_count, failed=failed_count)
    return _ok(trace_id, data=DocumentBatchDeleteData(
        total=len(body.doc_ids),
        success=success_count,
        failed=failed_count,
        deleted_chunks=total_deleted_chunks,
        results=results,
    ))


@router.post("/{doc_id}/reprocess", response_model=BaseResponse)
async def reprocess_document(doc_id: str, request: Request):
    """重新处理文档（删除旧切片，重新解析/分块/向量化）。"""
    trace_id = _get_trace_id(request)
    from app.tasks import enqueue_reprocess_document
    task_id = enqueue_reprocess_document(doc_id)
    return _ok(trace_id, data={"task_id": task_id})


@router.post("/{doc_id}/rebuild-vectors", response_model=BaseResponse)
async def rebuild_vectors(doc_id: str, request: Request):
    """对文档所有切片重建向量索引（异步）。"""
    trace_id = _get_trace_id(request)
    from app.tasks import enqueue_rebuild_vectors
    task_id = enqueue_rebuild_vectors(doc_id)
    return _ok(trace_id, data={"task_id": task_id})


def _get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "")


def _dict_to_item(item: dict) -> DocumentItem:
    """将 dict 转换为 DocumentItem（用于 list_by_kb 返回的数据）。"""
    return DocumentItem(
        id=item.get("id", 0),
        doc_id=item.get("doc_id", ""),
        kb_id=item.get("kb_id", 0),
        doc_name=item.get("doc_name", ""),
        doc_type=item.get("doc_type", ""),
        file_size=item.get("file_size", 0),
        source_type=item.get("source_type", "local_file"),
        source_url=item.get("source_url", ""),
        status=item.get("status", "pending"),
        chunk_count=item.get("chunk_count", 0),
        is_enabled=item.get("is_enabled", True),
        config=item.get("config"),
        error_message=item.get("error_message"),
        creator=item.get("creator"),
        gmt_create=item.get("gmt_create", ""),
        gmt_modified=item.get("gmt_modified", ""),
    )


def _doc_to_item(record) -> DocumentItem:
    """将 Document 模型转换为 DocumentItem。"""
    return DocumentItem(
        id=record.id,
        doc_id=record.doc_id,
        kb_id=record.kb_id,
        doc_name=record.doc_name,
        doc_type=record.doc_type or "",
        file_size=record.file_size or 0,
        source_type=record.source_type or "local_file",
        source_url=record.source_url or "",
        status=record.status or "pending",
        chunk_count=record.chunk_count or 0,
        is_enabled=record.is_enabled,
        config=record.config,
        error_message=record.error_message,
        creator=record.creator,
        gmt_create=record.gmt_create.isoformat() if record.gmt_create else "",
        gmt_modified=record.gmt_modified.isoformat() if record.gmt_modified else "",
    )


def _chunk_doc_to_item(item: dict) -> DocumentItem:
    """将 chunk_repo.list_documents 返回的 dict 转换为 DocumentItem（兜底方案）。"""
    return DocumentItem(
        id=0,
        doc_id=item.get("doc_id", ""),
        kb_id=int(item.get("kb_id", 0)) if item.get("kb_id") else 0,
        doc_name=item.get("doc_name", ""),
        doc_type=item.get("doc_type", ""),
        file_size=item.get("file_size", 0),
        source_type=item.get("source_type", "local_file"),
        source_url=item.get("source_url", ""),
        status=item.get("status", "success"),
        chunk_count=item.get("chunk_count", 0),
        is_enabled=item.get("is_enabled", True),
        config=None,
        error_message=None,
        creator=item.get("creator"),
        gmt_create=item.get("gmt_create", ""),
        gmt_modified=item.get("gmt_modified", ""),
    )


def _ok(trace_id: str = "", data=None, code: str = "OK"):
    # 如果 data 是 Pydantic 模型，转换为 dict
    if data is not None and hasattr(data, 'model_dump'):
        data = data.model_dump()
    elif data is not None and hasattr(data, 'dict'):
        data = data.dict()
    return {"success": True, "code": code, "message": None, "trace_id": trace_id, "data": data}


def _err(trace_id: str = "", message: str = "Error", code: str = "ERROR"):
    return {"success": False, "code": code, "message": message, "trace_id": trace_id, "data": None}
