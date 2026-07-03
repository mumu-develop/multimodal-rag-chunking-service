"""User management router."""
import structlog
from fastapi import APIRouter, Request

from app.repositories.user_repo import user_repo
from app.schemas import (
    BaseResponse,
    UserCreate,
    UserItem,
    UserListData,
    UserListResponse,
    UserUpdate,
)

logger = structlog.get_logger()
router = APIRouter()


@router.get("", response_model=UserListResponse)
async def list_users(
    page: int = 1,
    size: int = 20,
    request: Request = None,
):
    """用户列表。"""
    trace_id = _get_trace_id(request)
    total, items = user_repo.list(page=page, size=size)
    return _ok(trace_id, data=UserListData(
        total=total,
        page=page,
        result=[_user_to_item(item) for item in items],
    ))


@router.post("", response_model=BaseResponse)
async def create_user(body: UserCreate, request: Request):
    """创建用户。"""
    trace_id = _get_trace_id(request)
    record = user_repo.create(username=body.username, password=body.password, role=body.role)
    if not record:
        msg = user_repo.last_error or "创建用户失败"
        return _err(trace_id, msg, "DB_ERROR")
    return _ok(trace_id, data={"id": record.id})


@router.put("/{user_id}", response_model=BaseResponse)
async def update_user(user_id: int, body: UserUpdate, request: Request):
    """编辑用户。"""
    trace_id = _get_trace_id(request)
    updates = body.model_dump(exclude_none=True)
    if not updates:
        return _err(trace_id, "无更新字段", "BAD_REQUEST")
    ok = user_repo.update(user_id, **updates)
    if not ok:
        msg = user_repo.last_error or "更新失败或用户不存在"
        return _err(trace_id, msg, "DB_ERROR")
    return _ok(trace_id)


@router.delete("/{user_id}", response_model=BaseResponse)
async def delete_user(user_id: int, request: Request):
    """删除用户。"""
    trace_id = _get_trace_id(request)
    ok = user_repo.delete(user_id)
    if not ok:
        msg = user_repo.last_error or "删除失败或用户不存在"
        return _err(trace_id, msg, "DB_ERROR")
    return _ok(trace_id)


def _user_to_item(record) -> UserItem:
    return UserItem(
        id=record.id,
        username=record.username,
        role=record.role,
        is_active=record.is_active,
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
