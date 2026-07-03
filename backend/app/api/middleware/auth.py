"""API Key / JWT authentication middleware."""
import uuid

import structlog
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.config import settings

logger = structlog.get_logger()

# 免认证路径
AUTH_EXEMPT_PREFIXES = ("/health", "/docs", "/openapi.json", "/redoc")


class AuthMiddleware(BaseHTTPMiddleware):
    """认证中间件 — 支持 API Key 和 JWT Token。

    认证后设置 request.state:
    - user_id: 用户标识
    - user_role: 角色（admin/user）
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        # 跳过免认证路径
        if any(path.startswith(p) for p in AUTH_EXEMPT_PREFIXES):
            return await call_next(request)

        # 未配置认证则跳过，默认设为 admin
        if not settings.auth_enabled:
            request.state.user_role = "admin"
            request.state.user_id = "default"
            return await call_next(request)

        # 检查请求头中的 API Key
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if api_key and api_key.startswith("Bearer "):
            api_key = api_key[7:]

        if not api_key or api_key not in settings.api_key_set:
            trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
            logger.warning("auth_rejected", path=path, trace_id=trace_id)
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "code": "UNAUTHORIZED",
                    "message": "Invalid or missing API key",
                    "trace_id": trace_id,
                    "data": None,
                },
            )

        # 认证通过，设置角色（API Key 认证默认 admin）
        request.state.user_role = "admin"
        request.state.user_id = "api_key_user"
        return await call_next(request)
