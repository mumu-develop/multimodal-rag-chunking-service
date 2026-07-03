"""Chunks and document tree router."""
import structlog
from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.repositories.chunk_repo import chunk_repo
from app.repositories.kb_repo import kb_repo
from app.schemas import (
    BaseResponse,
    ChunkBatchDeleteRequest,
    ChunkBatchToggleRequest,
    ChunkUpdateRequest,
    ChunksListResponse,
    NestedTableResponse,
    NestedTablesListResponse,
    NestedTableData,
    ProductionDocTreeResponse,
)
from app.services.chunk_service import chunk_service

logger = structlog.get_logger()

router = APIRouter()

_MAX_PAGE_SIZE = 100


@router.get("/chunks", response_model=ChunksListResponse)
async def list_chunks(request: Request, doc_id: str, page: int = 1, size: int = 10, type: str | None = None):
    if page < 1:
        page = 1
    if size < 1:
        size = 10
    if size > _MAX_PAGE_SIZE:
        raise HTTPException(status_code=400, detail=f"Page size must be <= {_MAX_PAGE_SIZE}")

    trace_id = _get_trace_id(request)
    try:
        total, result = chunk_service.list_chunks(doc_id, page, size, type)
        return _ok(trace_id, data={"total": total, "page": page, "result": result})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("list_chunks_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Failed to list chunks", code="LIST_CHUNKS_ERROR")


@router.get("/docs/{doc_id}/tree", response_model=ProductionDocTreeResponse)
async def get_document_tree(doc_id: str, request: Request):
    """获取文档树（Paragraph + Chunks 双层结构）。"""
    trace_id = _get_trace_id(request)
    try:
        tree_data = chunk_service.get_production_doc_tree(doc_id)
        if not tree_data:
            return _err(trace_id=trace_id, message="Document not found or has no chunks", code="NOT_FOUND")
        return _ok(trace_id, data=tree_data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("get_doc_tree_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Failed to get document tree", code="DOC_TREE_ERROR")


# ============================================================
# 切片管理操作
# ============================================================

@router.put("/chunks/{chunk_id}", response_model=BaseResponse)
async def update_chunk(chunk_id: str, body: ChunkUpdateRequest, request: Request):
    """编辑切片内容。"""
    trace_id = _get_trace_id(request)
    ok = chunk_repo.update_chunk(chunk_id, body.content)
    if not ok:
        msg = chunk_repo.last_error or "切片不存在"
        return _err(trace_id, msg, "NOT_FOUND")
    return _ok(trace_id)


@router.delete("/chunks/{chunk_id}", response_model=BaseResponse)
async def delete_chunk(chunk_id: str, request: Request):
    """删除单个切片（MySQL + Milvus）。"""
    trace_id = _get_trace_id(request)

    # 1. 获取 chunk 的 kb_id
    chunk = chunk_repo.get_by_chunk_id(chunk_id)
    if not chunk:
        return _err(trace_id, "切片不存在", "NOT_FOUND")

    # 2. 获取 collection_name
    collection_name = settings.MILVUS_COLLECTION_NAME
    if chunk.kb_id:
        kb = kb_repo.get_by_id(int(chunk.kb_id))
        if kb:
            collection_name = kb.collection_name

    # 3. 删除 Milvus
    try:
        from app.repositories.milvus_repo import milvus_repo
        milvus_repo.delete([chunk_id], collection_name=collection_name)
    except Exception:
        logger.exception("milvus_chunk_delete_failed", chunk_id=chunk_id, collection=collection_name)

    # 4. 删除 MySQL
    ok = chunk_repo.delete_chunk(chunk_id)
    if not ok:
        msg = chunk_repo.last_error or "切片不存在"
        return _err(trace_id, msg, "NOT_FOUND")
    return _ok(trace_id)


@router.put("/chunks/{chunk_id}/toggle", response_model=BaseResponse)
async def toggle_chunk(chunk_id: str, is_enabled: bool, request: Request):
    """启用/禁用单个切片。"""
    trace_id = _get_trace_id(request)
    ok = chunk_repo.toggle_chunk(chunk_id, is_enabled)
    if not ok:
        msg = chunk_repo.last_error or "切片不存在"
        return _err(trace_id, msg, "NOT_FOUND")
    return _ok(trace_id)


@router.post("/chunks/batch-toggle", response_model=BaseResponse)
async def batch_toggle_chunks(body: ChunkBatchToggleRequest, request: Request):
    """批量启用/禁用切片。"""
    trace_id = _get_trace_id(request)
    row = chunk_repo.batch_toggle(body.chunk_ids, body.is_enabled)
    return _ok(trace_id, data={"affected": row})


@router.post("/chunks/batch-delete", response_model=BaseResponse)
async def batch_delete_chunks(body: ChunkBatchDeleteRequest, request: Request):
    """批量删除切片。"""
    trace_id = _get_trace_id(request)

    if not body.chunk_ids:
        return _ok(trace_id, data={"affected": 0})

    # 获取第一个 chunk 的 kb_id 来确定 collection（假设批量删除的 chunks 属于同一 KB）
    first_chunk = chunk_repo.get_by_chunk_id(body.chunk_ids[0])
    collection_name = settings.MILVUS_COLLECTION_NAME
    if first_chunk and first_chunk.kb_id:
        kb = kb_repo.get_by_id(int(first_chunk.kb_id))
        if kb:
            collection_name = kb.collection_name

    try:
        from app.repositories.milvus_repo import milvus_repo
        milvus_repo.delete(body.chunk_ids, collection_name=collection_name)
    except Exception:
        logger.exception("milvus_batch_delete_failed", collection=collection_name)

    row = chunk_repo.batch_delete(body.chunk_ids)
    return _ok(trace_id, data={"affected": row})


# ============================================================
# 嵌套表格查询
# ============================================================

@router.get("/chunks/{parent_chunk_id}/nested/{nested_index}", response_model=NestedTableResponse)
async def get_nested_table(parent_chunk_id: str, nested_index: int, request: Request):
    """
    获取单个嵌套表格。

    参数：
        parent_chunk_id: 父表格 chunk ID
        nested_index: 嵌套表格索引（从 HTML 中的 data-nested 属性获取）

    返回：
        嵌套表格的完整结构化数据（HTML、headers、rows、merges）
    """
    trace_id = _get_trace_id(request)
    try:
        import json

        chunk = chunk_repo.get_nested_table(parent_chunk_id, nested_index)
        if not chunk:
            return _err(trace_id, f"Nested table not found: parent={parent_chunk_id}, index={nested_index}", "NOT_FOUND")

        # 解析 content JSON
        try:
            table_data = json.loads(chunk.content)
        except (json.JSONDecodeError, TypeError):
            table_data = {}

        # 构建响应数据
        nested_data = NestedTableData(
            nested_index=nested_index,
            chunk_id=chunk.chunk_id,
            html=table_data.get("html", ""),
            row_count=table_data.get("row_count", 0),
            col_count=table_data.get("col_count", 0),
            headers=table_data.get("headers", []),
            rows=table_data.get("rows", []),
            merges=table_data.get("merges", []),
            parent_row=chunk.metadata.get("parent_row") if chunk.metadata else None,
            parent_col=chunk.metadata.get("parent_col") if chunk.metadata else None,
        )
        return _ok(trace_id, data=nested_data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("get_nested_table_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Failed to get nested table", code="NESTED_TABLE_ERROR")


@router.get("/chunks/{parent_chunk_id}/nested", response_model=NestedTablesListResponse)
async def get_nested_tables(parent_chunk_id: str, request: Request):
    """
    获取某个父表格的所有嵌套表格。

    参数：
        parent_chunk_id: 父表格 chunk ID

    返回：
        嵌套表格列表
    """
    trace_id = _get_trace_id(request)
    try:
        import json

        chunks = chunk_repo.get_nested_tables_by_parent(parent_chunk_id)
        if not chunks:
            return _ok(trace_id, data={
                "parent_chunk_id": parent_chunk_id,
                "nested_tables": [],
                "total": 0,
            })

        nested_tables = []
        for chunk in chunks:
            try:
                table_data = json.loads(chunk.content)
            except (json.JSONDecodeError, TypeError):
                table_data = {}

            nested_index = chunk.metadata.get("nested_index", 0) if chunk.metadata else 0
            nested_tables.append(NestedTableData(
                nested_index=nested_index,
                chunk_id=chunk.chunk_id,
                html=table_data.get("html", ""),
                row_count=table_data.get("row_count", 0),
                col_count=table_data.get("col_count", 0),
                headers=table_data.get("headers", []),
                rows=table_data.get("rows", []),
                merges=table_data.get("merges", []),
                parent_row=chunk.metadata.get("parent_row") if chunk.metadata else None,
                parent_col=chunk.metadata.get("parent_col") if chunk.metadata else None,
            ))

        return _ok(trace_id, data={
            "parent_chunk_id": parent_chunk_id,
            "nested_tables": nested_tables,
            "total": len(nested_tables),
        })
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("get_nested_tables_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Failed to get nested tables", code="NESTED_TABLES_ERROR")


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


# ============================================================
# 切分策略和插件信息 API
# ============================================================

@router.get("/chunking/strategies")
async def list_chunking_strategies(request: Request):
    """获取所有可用的切分策略。

    返回：
    - strategies: 策略列表，包含名称、描述、默认配置
    """
    trace_id = _get_trace_id(request)
    try:
        # 策略定义（硬编码，与 markdown_chunker 保持一致）
        strategies = [
            {
                "name": "smart",
                "description": "智能切分（标题 + 语义 + 递归）- 最高质量，默认推荐",
                "default_config": {
                    "chunk_token_limit": 1024,
                    "chunk_token_overlap": 100,
                },
                "recommended": True,
            },
            {
                "name": "heading",
                "description": "只按标题切分 - 保持文档原始结构",
                "default_config": {
                    "chunk_token_limit": 1024,
                },
                "recommended": False,
            },
            {
                "name": "semantic",
                "description": "只按语义切分 - 保持语义连贯性",
                "default_config": {
                    "chunk_token_limit": 1024,
                    "similarity_threshold": 0.5,
                },
                "recommended": False,
            },
            {
                "name": "recursive",
                "description": "只按递归切分 - 最简单快速",
                "default_config": {
                    "chunk_size": 800,
                },
                "recommended": False,
            },
            {
                "name": "fixed",
                "description": "固定长度切分 - 最基础方案",
                "default_config": {
                    "chunk_size": 800,
                },
                "recommended": False,
            },
        ]
        return _ok(trace_id, data={"strategies": strategies})
    except Exception as exc:
        logger.exception("list_strategies_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Failed to list strategies", code="STRATEGIES_ERROR")


@router.get("/chunking/plugins")
async def list_chunking_plugins(request: Request):
    """获取所有已加载的插件。

    返回：
    - plugins: 插件列表，包含名称、描述、优先级、状态
    """
    trace_id = _get_trace_id(request)
    try:
        from app.plugins import plugin_manager
        plugins = plugin_manager.list()
        return _ok(trace_id, data={"plugins": plugins})
    except Exception as exc:
        logger.exception("list_plugins_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Failed to list plugins", code="PLUGINS_ERROR")


@router.post("/chunking/plugins/{name}/enable")
async def enable_plugin(name: str, request: Request):
    """启用指定插件。

    参数：
    - name: 插件名称
    """
    trace_id = _get_trace_id(request)
    try:
        from app.plugins import plugin_manager
        success = plugin_manager.enable(name)
        if success:
            return _ok(trace_id, data={"name": name, "enabled": True})
        return _err(trace_id=trace_id, message=f"Plugin '{name}' not found", code="PLUGIN_NOT_FOUND")
    except Exception as exc:
        logger.exception("enable_plugin_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Failed to enable plugin", code="PLUGIN_ENABLE_ERROR")


@router.post("/chunking/plugins/{name}/disable")
async def disable_plugin(name: str, request: Request):
    """禁用指定插件。

    参数：
    - name: 插件名称
    """
    trace_id = _get_trace_id(request)
    try:
        from app.plugins import plugin_manager
        success = plugin_manager.disable(name)
        if success:
            return _ok(trace_id, data={"name": name, "enabled": False})
        return _err(trace_id=trace_id, message=f"Plugin '{name}' not found", code="PLUGIN_NOT_FOUND")
    except Exception as exc:
        logger.exception("disable_plugin_failed", trace_id=trace_id, error=str(exc))
        return _err(trace_id=trace_id, message="Failed to disable plugin", code="PLUGIN_DISABLE_ERROR")
