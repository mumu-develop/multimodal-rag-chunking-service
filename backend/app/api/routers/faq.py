"""FAQ management router."""
import structlog
from fastapi import APIRouter, Request

from app.repositories.faq_repo import faq_repo
from app.repositories.kb_repo import kb_repo
from app.repositories.async_milvus_repo import async_milvus_repo
from app.schemas import (
    BaseResponse,
    FAQCreate,
    FAQDetailResponse,
    FAQItem,
    FAQListData,
    FAQListResponse,
    FAQSearchRequest,
    FAQSearchResponse,
    FAQSearchResponseData,
    FAQSearchHitItem,
    FAQUpdate,
    FAQBatchImportRequest,
    FAQBatchImportData,
    FAQBatchImportResponse,
    FAQBatchDeleteRequest,
)
from app.search.async_hybrid_search import async_hybrid_searcher
from app.embedders.async_embedder import async_embedder

logger = structlog.get_logger()
router = APIRouter()


@router.post("", response_model=FAQDetailResponse)
async def create_faq(body: FAQCreate, request: Request):
    """创建FAQ问答对（MySQL + Milvus）。"""
    trace_id = _get_trace_id(request)

    # 1. 验证知识库存在
    kb = kb_repo.get_by_id(body.kb_id)
    if not kb:
        return _err(trace_id, "知识库不存在", "NOT_FOUND")

    # 2. 检查知识库类型（faq 或 hybrid）
    if kb.kb_type not in ("faq", "hybrid"):
        return _err(trace_id, "知识库类型不支持FAQ（需要 faq 或 hybrid）", "KB_TYPE_ERROR")

    # 3. 创建 MySQL 记录
    record = faq_repo.create(
        kb_id=body.kb_id,
        question=body.question,
        answer=body.answer,
        category=body.category,
        keywords=body.keywords,
        source=body.source,
        sort_order=body.sort_order,
        creator=body.creator,
    )
    if not record:
        msg = faq_repo.last_error or "创建FAQ失败"
        return _err(trace_id, msg, "DB_ERROR")

    # 4. 编码问题向量（异步）
    try:
        vector = await async_embedder.encode_query(body.question)
    except Exception as e:
        logger.exception("faq_encode_failed", faq_id=record.faq_id, error=str(e))
        # 编码失败也保留 MySQL 记录，仅标记未同步
        return _ok(trace_id, data=_faq_to_item(record, synced=False))

    # 5. 写入 Milvus（异步）
    milvus_ok = await async_milvus_repo.upsert_faq(
        {
            "faq_id": record.faq_id,
            "kb_id": str(body.kb_id),
            "question": body.question,
            "answer": body.answer,
            "category": body.category,
            "keywords": body.keywords,
            "source": body.source,
            "hit_count": 0,
            "vector": vector,
        },
        collection_name=kb.collection_name,
    )

    if not milvus_ok:
        logger.warning("faq_milvus_failed", faq_id=record.faq_id)
        return _ok(trace_id, data=_faq_to_item(record, synced=False))

    return _ok(trace_id, data=_faq_to_item(record))


@router.get("", response_model=FAQListResponse)
async def list_faqs(
    kb_id: int,
    page: int = 1,
    size: int = 20,
    category: str = "",
    request: Request = None,
):
    """FAQ列表（分页）。"""
    trace_id = _get_trace_id(request)
    total, items = faq_repo.list(kb_id, page=page, size=size, category=category)
    return _ok(trace_id, data=FAQListData(
        total=total,
        page=page,
        result=[_faq_to_item(item) for item in items],
    ))


@router.get("/{faq_id}", response_model=FAQDetailResponse)
async def get_faq(faq_id: str, request: Request):
    """FAQ详情。"""
    trace_id = _get_trace_id(request)
    record = faq_repo.get_by_id(faq_id)
    if not record:
        return _err(trace_id, "FAQ不存在", "NOT_FOUND")
    return _ok(trace_id, data=_faq_to_item(record))


@router.put("/{faq_id}", response_model=BaseResponse)
async def update_faq(faq_id: str, body: FAQUpdate, request: Request):
    """更新FAQ。"""
    trace_id = _get_trace_id(request)

    # 1. 检查 FAQ 存在
    record = faq_repo.get_by_id(faq_id)
    if not record:
        return _err(trace_id, "FAQ不存在", "NOT_FOUND")

    # 2. 更新 MySQL
    updates = body.model_dump(exclude_none=True)
    if not updates:
        return _err(trace_id, "无更新字段", "BAD_REQUEST")

    ok = faq_repo.update(faq_id, **updates)
    if not ok:
        msg = faq_repo.last_error or "更新失败"
        return _err(trace_id, msg, "DB_ERROR")

    # 3. 如果更新了问题，需要重新编码并更新 Milvus（异步）
    if "question" in updates:
        kb = kb_repo.get_by_id(record.kb_id)
        if kb:
            try:
                vector = await async_embedder.encode_query(updates["question"])
                # 更新 answer 也一并更新
                answer = updates.get("answer", record.answer)
                category = updates.get("category", record.category)
                keywords = updates.get("keywords", record.keywords)

                await async_milvus_repo.upsert_faq(
                    {
                        "faq_id": faq_id,
                        "kb_id": str(record.kb_id),
                        "question": updates["question"],
                        "answer": answer,
                        "category": category,
                        "keywords": keywords,
                        "source": record.source,
                        "hit_count": record.hit_count,
                        "vector": vector,
                    },
                    collection_name=kb.collection_name,
                )
            except Exception as e:
                logger.exception("faq_update_milvus_failed", faq_id=faq_id, error=str(e))

    return _ok(trace_id)


@router.delete("/{faq_id}", response_model=BaseResponse)
async def delete_faq(faq_id: str, request: Request):
    """删除FAQ（MySQL + Milvus）。"""
    trace_id = _get_trace_id(request)

    # 1. 检查 FAQ 存在
    record = faq_repo.get_by_id(faq_id)
    if not record:
        return _err(trace_id, "FAQ不存在", "NOT_FOUND")

    # 2. 获取知识库信息
    kb = kb_repo.get_by_id(record.kb_id)

    # 3. 删除 Milvus（异步）
    if kb:
        await async_milvus_repo.delete_faq([faq_id], collection_name=kb.collection_name)

    # 4. 删除 MySQL
    ok = faq_repo.delete(faq_id)
    if not ok:
        msg = faq_repo.last_error or "删除失败"
        return _err(trace_id, msg, "DB_ERROR")

    return _ok(trace_id, data={"faq_id": faq_id})


@router.post("/batch", response_model=FAQBatchImportResponse)
async def batch_import_faq(body: FAQBatchImportRequest, request: Request):
    """批量导入FAQ。"""
    trace_id = _get_trace_id(request)

    # 1. 验证知识库
    kb = kb_repo.get_by_id(body.kb_id)
    if not kb:
        return _err(trace_id, "知识库不存在", "NOT_FOUND")

    if kb.kb_type not in ("faq", "hybrid"):
        return _err(trace_id, "知识库类型不支持FAQ", "KB_TYPE_ERROR")

    # 2. 批量创建 MySQL 记录
    items_data = [item.model_dump() for item in body.items]
    success, results = faq_repo.batch_import(
        kb_id=body.kb_id,
        items=items_data,
        source="import",
        creator=body.creator,
    )

    # 3. 编码成功创建的 FAQ 向量并写入 Milvus（异步）
    if success > 0:
        # 获取成功创建的 FAQ
        successful_faqs = faq_repo.get_by_ids([r["faq_id"] for r in results if r.get("success")])

        # 批量异步编码
        faq_vectors = []
        for faq in successful_faqs:
            try:
                vector = await async_embedder.encode_query(faq.question)
                faq_vectors.append({
                    "faq_id": faq.faq_id,
                    "kb_id": str(body.kb_id),
                    "question": faq.question,
                    "answer": faq.answer,
                    "category": faq.category,
                    "keywords": faq.keywords,
                    "source": faq.source,
                    "hit_count": 0,
                    "vector": vector,
                })
            except Exception as e:
                logger.warning("faq_batch_encode_failed", faq_id=faq.faq_id, error=str(e))

        # 批量异步写入 Milvus
        if faq_vectors:
            await async_milvus_repo.batch_upsert_faq(faq_vectors, collection_name=kb.collection_name)

    return _ok(trace_id, data=FAQBatchImportData(
        total=len(body.items),
        success=success,
        failed=len(body.items) - success,
        results=results,
    ))


@router.post("/batch-delete", response_model=BaseResponse)
async def batch_delete_faq(body: FAQBatchDeleteRequest, request: Request):
    """批量删除FAQ。"""
    trace_id = _get_trace_id(request)

    if not body.faq_ids:
        return _err(trace_id, "faq_ids 不能为空", "BAD_REQUEST")

    # 获取 FAQ 记录以获取 kb_id
    faqs = faq_repo.get_by_ids(body.faq_ids)
    if not faqs:
        return _err(trace_id, "未找到任何FAQ", "NOT_FOUND")

    # 按知识库分组删除 Milvus
    kb_faq_map = {}
    for faq in faqs:
        kb_id = faq.kb_id
        if kb_id not in kb_faq_map:
            kb_faq_map[kb_id] = []
        kb_faq_map[kb_id].append(faq.faq_id)

    for kb_id, faq_ids in kb_faq_map.items():
        kb = kb_repo.get_by_id(kb_id)
        if kb:
            await async_milvus_repo.delete_faq(faq_ids, collection_name=kb.collection_name)

    # 批量删除 MySQL
    deleted = faq_repo.batch_delete(body.faq_ids)

    return _ok(trace_id, data={
        "requested": len(body.faq_ids),
        "deleted": deleted,
    })


@router.post("/search", response_model=FAQSearchResponse)
async def search_faq(body: FAQSearchRequest, request: Request):
    """FAQ检索。"""
    trace_id = _get_trace_id(request)

    # 1. 验证知识库
    kb = kb_repo.get_by_id(body.kb_id)
    if not kb:
        return _err(trace_id, "知识库不存在", "NOT_FOUND")

    # 2. 根据知识库类型选择检索模式（异步）
    if kb.kb_type == "faq":
        # 纯 FAQ 检索（异步）
        hits = await async_hybrid_searcher.search_faq(
            query=body.query,
            top_k=body.top_k,
            collection_name=kb.collection_name,
            use_rerank=body.use_rerank,
            category=body.category,
        )
    elif kb.kb_type == "hybrid":
        # 混合检索（FAQ + 文档）（异步）
        hits = await async_hybrid_searcher.search_hybrid(
            query=body.query,
            top_k=body.top_k,
            collection_name=kb.collection_name,
            use_rerank=body.use_rerank,
            faq_weight=0.6,
            category=body.category,
        )
    else:
        return _err(trace_id, "知识库类型不支持FAQ检索", "KB_TYPE_ERROR")

    # 3. 格式化结果
    hit_items = []
    for hit in hits:
        hit_items.append(FAQSearchHitItem(
            faq_id=hit["chunk_id"],
            kb_id=body.kb_id,
            question=hit["content"],
            answer=hit["summary"],
            category=hit["section"],
            keywords=hit["tags"],
            hit_count=hit["priority"],
            score=round(hit["score"], 4),
            rerank_score=round(hit.get("rerank_score", 0), 4) if hit.get("rerank_score") else None,
        ))

    # 4. 更新命中次数（仅 FAQ 结果）
    for hit in hits:
        if hit["data_type"] == "faq":
            faq_repo.increment_hit_count(hit["chunk_id"])

    return _ok(trace_id, data=FAQSearchResponseData(hits=hit_items))


def _faq_to_item(record, synced: bool = True) -> FAQItem:
    """转换 ORM 记录为 Schema Item。"""
    return FAQItem(
        id=record.id,
        faq_id=record.faq_id,
        kb_id=record.kb_id,
        question=record.question,
        answer=record.answer,
        category=record.category or "",
        keywords=record.keywords or "",
        source=record.source,
        hit_count=record.hit_count,
        sort_order=record.sort_order,
        is_enabled=record.is_enabled,
        creator=record.creator,
        gmt_create=record.gmt_create.isoformat() if record.gmt_create else "",
        gmt_modified=record.gmt_modified.isoformat() if record.gmt_modified else "",
    )


def _get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "")


def _ok(trace_id: str = "", data=None, code: str = "OK"):
    if data is not None and hasattr(data, 'model_dump'):
        data = data.model_dump()
    elif data is not None and hasattr(data, 'dict'):
        data = data.dict()
    return {"success": True, "code": code, "message": None, "trace_id": trace_id, "data": data}


def _err(trace_id: str = "", message: str = "Error", code: str = "ERROR"):
    return {"success": False, "code": code, "message": message, "trace_id": trace_id, "data": None}