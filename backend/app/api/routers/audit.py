"""审计日志查询路由。"""
import structlog
from fastapi import APIRouter, Request
from datetime import date, timedelta

from app.repositories.audit_repo import audit_repo
from app.schemas import (
    BaseResponse,
    AuditLogItem,
    AuditLogListData,
    AuditLogListResponse,
    DailyActiveItem,
    DailyActiveTrendData,
    DailyActiveTrendResponse,
)

logger = structlog.get_logger()
router = APIRouter()


@router.get("/logs", response_model=AuditLogListResponse)
async def get_audit_logs(
    user_id: str = "",
    page: int = 1,
    size: int = 20,
    request: Request = None,
):
    """获取审计日志列表（可按用户过滤）。"""
    trace_id = _get_trace_id(request)

    if user_id:
        total, logs = audit_repo.get_user_access_logs(user_id, page=page, size=size)
    else:
        # 获取所有日志（需要添加新方法）
        total, logs = audit_repo.get_all_logs(page=page, size=size)

    items = [
        AuditLogItem(
            id=log["id"],
            user_id=log["user_id"],
            user_role=log.get("user_role", ""),
            path=log["path"],
            method=log["method"],
            status_code=log["status_code"],
            latency_ms=log["latency_ms"],
            params=log.get("params"),
            trace_id=log.get("trace_id", ""),
            gmt_create=log["gmt_create"],
        )
        for log in logs
    ]

    return _ok(trace_id, data=AuditLogListData(
        total=total,
        page=page,
        result=items,
    ))


@router.get("/daily-active", response_model=DailyActiveTrendResponse)
async def get_daily_active_trend(
    days: int = 7,
    request: Request = None,
):
    """获取日活趋势（最近 N 天）。"""
    trace_id = _get_trace_id(request)

    trend_data = audit_repo.get_daily_active_trend(days=days)

    # 计算总用户数
    total_users = sum(item["count"] for item in trend_data)

    items = [
        DailyActiveItem(date=item["date"], count=item["count"])
        for item in trend_data
    ]

    return _ok(trace_id, data=DailyActiveTrendData(
        trend=items,
        total_users=total_users,
    ))


@router.get("/daily-active/{date_str}", response_model=BaseResponse)
async def get_daily_active_count(
    date_str: str,
    request: Request = None,
):
    """获取指定日期的活跃用户数。"""
    trace_id = _get_trace_id(request)

    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        return _err(trace_id, "日期格式错误，请使用 YYYY-MM-DD 格式", "BAD_REQUEST")

    count = audit_repo.get_daily_active_count(target_date)

    return _ok(trace_id, data={"date": date_str, "count": count})


def _get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "")


def _ok(trace_id: str = "", data=None, code: str = "OK"):
    return {"success": True, "code": code, "message": None, "trace_id": trace_id, "data": data}


def _err(trace_id: str = "", message: str = "Error", code: str = "ERROR"):
    return {"success": False, "code": code, "message": message, "trace_id": trace_id, "data": None}