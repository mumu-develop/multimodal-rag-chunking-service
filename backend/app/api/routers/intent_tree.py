"""Intent Tree management router - 知识库分类树管理。

用于前端展示知识库的分类层级结构，用户直接选择知识库查询，无需意图分类。
"""
import structlog
from fastapi import APIRouter, Request

from app.repositories.intent_repo import intent_repo
from app.schemas import (
    BaseResponse,
    IntentNodeCreate,
    IntentNodeItem,
    IntentNodeUpdate,
    IntentTreeResponse,
)

logger = structlog.get_logger()
router = APIRouter()


@router.get("/tree", response_model=IntentTreeResponse)
async def get_intent_tree(request: Request):
    """获取整棵意图树（知识库分类结构）。"""
    trace_id = _get_trace_id(request)
    nodes = intent_repo.get_tree()
    return _ok(trace_id, data=[IntentNodeItem(**n) for n in nodes])


@router.post("/nodes", response_model=BaseResponse)
async def create_intent_node(body: IntentNodeCreate, request: Request):
    """创建意图树节点（分类节点）。"""
    trace_id = _get_trace_id(request)
    record = intent_repo.create_node(
        name=body.name,
        node_type=body.node_type,
        parent_id=body.parent_id,
        sort_order=body.sort_order,
        collection_name=body.collection_name,
        description=body.description,
        example_questions=body.example_questions,
    )
    if not record:
        msg = intent_repo.last_error or "创建节点失败"
        return _err(trace_id, msg, "DB_ERROR")
    return _ok(trace_id, data={"id": record.id})


@router.put("/nodes/{node_id}", response_model=BaseResponse)
async def update_intent_node(node_id: int, body: IntentNodeUpdate, request: Request):
    """编辑意图树节点。"""
    trace_id = _get_trace_id(request)
    updates = body.model_dump(exclude_none=True)
    if not updates:
        return _err(trace_id, "无更新字段", "BAD_REQUEST")
    ok = intent_repo.update_node(node_id, **updates)
    if not ok:
        msg = intent_repo.last_error or "更新失败或节点不存在"
        return _err(trace_id, msg, "DB_ERROR")
    return _ok(trace_id)


@router.delete("/nodes/{node_id}", response_model=BaseResponse)
async def delete_intent_node(node_id: int, request: Request):
    """删除意图树节点（需无子节点）。"""
    trace_id = _get_trace_id(request)
    ok = intent_repo.delete_node(node_id)
    if not ok:
        msg = intent_repo.last_error or "删除失败（节点不存在或存在子节点）"
        return _err(trace_id, msg, "DB_ERROR")
    return _ok(trace_id)


@router.put("/nodes/{node_id}/toggle", response_model=BaseResponse)
async def toggle_intent_node(node_id: int, is_enabled: bool = True, request: Request = None):
    """启用/禁用意图树节点。"""
    trace_id = _get_trace_id(request)
    ok = intent_repo.toggle_node(node_id, is_enabled)
    if not ok:
        msg = intent_repo.last_error or "操作失败或节点不存在"
        return _err(trace_id, msg, "DB_ERROR")
    return _ok(trace_id)


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