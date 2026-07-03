"""RBAC middleware — role-based access control for admin endpoints."""
import structlog
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.config import settings

logger = structlog.get_logger()

# 仅 admin 角色可访问的路径前缀
ADMIN_ONLY_PREFIXES = (
    "/api/v1/knowledge-bases",
    "/api/v1/users",
    "/api/v1/dashboard",
    "/api/v1/intent-tree",
)

# admin 只读路径（GET 请求全员可访问）
ADMIN_READONLY_PATHS = {
    "/api/v1/knowledge-bases",
    "/api/v1/intent-tree",
    "/api/v1/dashboard",
}


class RbacMiddleware(BaseHTTPMiddleware):
    """基于角色的访问控制中间件。

    管理类接口（knowledge-bases, users, dashboard, intent-tree）：
    - GET 请求：所有认证用户可访问
    - POST/PUT/DELETE：仅 admin 角色可访问
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        method = request.method

        # 只拦截管理类路径
        if not any(path.startswith(prefix) for prefix in ADMIN_ONLY_PREFIXES):
            return await call_next(request)

        # GET 请求在只读路径上放行
        if method == "GET":
            return await call_next(request)

        # 未启用认证则跳过 RBAC
        if not settings.auth_enabled:
            return await call_next(request)

        # 从 request.state 获取用户角色（由 AuthMiddleware 设置）
        user_role = getattr(request.state, "user_role", None)

        if user_role != "admin":
            trace_id = getattr(request.state, "trace_id", "")
            logger.warning(
                "rbac_denied",
                path=path,
                method=method,
                user_role=user_role,
                trace_id=trace_id,
            )
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "code": "FORBIDDEN",
                    "message": "Admin role required for this operation",
                    "trace_id": trace_id,
                    "data": None,
                },
            )

        return await call_next(request)
