"""Dashboard router."""
import structlog
from fastapi import APIRouter, Query, Request

from app.repositories.session_repo import session_repo
from app.schemas import (
    AiPerformanceResponse,
    DashboardStatsResponse,
    QualityResponse,
    TrendResponse,
)

logger = structlog.get_logger()
router = APIRouter()


@router.get("/stats", response_model=DashboardStatsResponse)
async def dashboard_stats(
    window: str = Query(default="7d", description="时间窗口 24h|7d|30d"),
    request: Request = None,
):
    """核心指标卡片。"""
    trace_id = _get_trace_id(request)
    hours = _parse_window(window)
    stats = session_repo.stats(window_hours=hours)
    # 计算运营效率指标
    stats["per_user_sessions"] = round(stats["sessions"] / max(stats["active_users"], 1), 2)
    stats["per_session_messages"] = stats["session_depth"]
    return _ok(trace_id, data=stats)


@router.get("/ai-performance", response_model=AiPerformanceResponse)
async def ai_performance(
    window: str = Query(default="7d"),
    request: Request = None,
):
    """AI 性能指标。"""
    trace_id = _get_trace_id(request)
    hours = _parse_window(window)
    perf = session_repo.ai_performance(window_hours=hours)
    return _ok(trace_id, data=perf)


@router.get("/quality", response_model=QualityResponse)
async def quality_snapshot(
    days: int = Query(default=7, ge=1, le=30),
    request: Request = None,
):
    """质量快照（近N天错误率/无知识率/慢响应率）。"""
    trace_id = _get_trace_id(request)
    data = session_repo.quality_snapshot(days=days)
    return _ok(trace_id, data=data)


@router.get("/trends", response_model=TrendResponse)
async def trend_analysis(
    metric: str = Query(default="sessions", description="sessions|users"),
    days: int = Query(default=7, ge=1, le=30),
    request: Request = None,
):
    """趋势分析。"""
    trace_id = _get_trace_id(request)
    if metric == "users":
        data = session_repo.user_trend(days=days)
    else:
        data = session_repo.session_trend(days=days)
    return _ok(trace_id, data=data)


def _parse_window(window: str) -> int:
    mapping = {"24h": 24, "7d": 7 * 24, "30d": 30 * 24}
    return mapping.get(window, 7 * 24)


def _get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "")


def _ok(trace_id: str = "", data=None, code: str = "OK"):
    # 如果 data 是 Pydantic 模型，转换为 dict
    if data is not None and hasattr(data, 'model_dump'):
        data = data.model_dump()
    elif data is not None and hasattr(data, 'dict'):
        data = data.dict()
    return {"success": True, "code": code, "message": None, "trace_id": trace_id, "data": data}
