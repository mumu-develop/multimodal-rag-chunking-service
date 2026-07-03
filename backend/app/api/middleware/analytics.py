"""Analytics middleware — 审计日志 + 日活统计 + 请求耗时记录。"""
import time
import threading
import json
from datetime import datetime, date

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger()

# 需要采集消息数据的路径
ANALYTICS_PATHS = {
    "/api/v1/search": "search",
    "/api/v1/search/hybrid": "search",
}


class AnalyticsMiddleware(BaseHTTPMiddleware):
    """审计中间件：记录用户访问 + 日活统计 + 请求耗时。

    功能：
    1. 记录每个请求的用户ID、路径、参数、耗时
    2. 异步写入审计日志表（不阻塞响应）
    3. 日活统计（每日活跃用户数）
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.monotonic()

        # 获取请求参数（需要在调用 call_next 之前读取 body）
        request_params = await _get_request_params(request)

        response = await call_next(request)

        # 记录耗时
        latency_ms = int((time.monotonic() - start_time) * 1000)
        response.headers["X-Response-Time-Ms"] = str(latency_ms)

        # 获取用户信息（从 AuthMiddleware 设置的状态）
        user_id = getattr(request.state, "user_id", None) or request.headers.get("X-User-Id", "anonymous")
        user_role = getattr(request.state, "user_role", "guest")

        # 记录审计日志（所有请求）
        _log_audit_async(
            user_id=user_id,
            user_role=user_role,
            path=request.url.path,
            method=request.method,
            params=request_params,
            status_code=response.status_code,
            latency_ms=latency_ms,
            trace_id=getattr(request.state, "trace_id", ""),
        )

        # 日活统计（异步写入）
        _log_daily_active_async(user_id)

        # 对 search 路径采集会话/消息数据（异步写入，不阻塞）
        path = request.url.path
        if path in ANALYTICS_PATHS and response.status_code < 400:
            _collect_async(request, response, latency_ms)

        return response


async def _get_request_params(request: Request) -> dict:
    """获取请求参数（GET 参数 + POST body）。"""
    params = {}

    # GET 参数
    if request.query_params:
        params["query"] = dict(request.query_params)

    # POST body（需要小心处理，避免读取后影响后续处理）
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            # 读取 body 并缓存，以便后续路由可以再次读取
            body = await request.body()
            if body:
                try:
                    params["body"] = json.loads(body)
                except json.JSONDecodeError:
                    params["body"] = body.decode("utf-8")[:500]  # 非 JSON，截取前500字符
        except Exception:
            params["body"] = None

    return params


def _log_audit_async(
    user_id: str,
    user_role: str,
    path: str,
    method: str,
    params: dict,
    status_code: int,
    latency_ms: int,
    trace_id: str,
) -> None:
    """异步写入审计日志（后台线程）。"""
    def _write():
        try:
            from app.repositories.audit_repo import audit_repo
            audit_repo.log_access(
                user_id=user_id,
                user_role=user_role,
                path=path,
                method=method,
                params=params,
                status_code=status_code,
                latency_ms=latency_ms,
                trace_id=trace_id,
            )
        except Exception:
            logger.debug("audit_log_skipped", user_id=user_id, path=path)

    t = threading.Thread(target=_write, daemon=True)
    t.start()


def _log_daily_active_async(user_id: str) -> None:
    """异步记录日活（后台线程）。"""
    if not user_id or user_id == "anonymous":
        return

    def _write():
        try:
            from app.repositories.audit_repo import audit_repo
            audit_repo.log_daily_active(user_id=user_id, date=date.today())
        except Exception:
            logger.debug("daily_active_log_skipped", user_id=user_id)

    t = threading.Thread(target=_write, daemon=True)
    t.start()


def _collect_async(request: Request, response: Response, latency_ms: int) -> None:
    """异步采集会话/消息数据（后台线程写入）。"""
    trace_id = getattr(request.state, "trace_id", "")
    session_id = request.headers.get("X-Session-Id", trace_id)

    def _write():
        try:
            from app.repositories.session_repo import session_repo
            session_repo.add_message(
                session_id=session_id,
                query="",
                response="",
                latency_ms=latency_ms,
                is_success=True,
            )
        except Exception:
            logger.debug("analytics_collect_skipped", trace_id=trace_id)

    t = threading.Thread(target=_write, daemon=True)
    t.start()
