"""Health check router."""
import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.repositories.milvus_repo import milvus_repo

logger = structlog.get_logger()

router = APIRouter()

# 共享 Redis 连接池
_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        import redis
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=3,
            socket_timeout=3,
            decode_responses=False,
        )
    return _redis_client


_CHECK_TIMEOUT = 5  # 每个检查的超时秒数


def _sanitize_error(msg: str) -> str:
    """去除敏感信息，只保留异常类型和简短描述。"""
    # 只保留异常类型（如 ConnectionRefusedError）
    if ":" in msg:
        return msg.split(":")[0].strip()
    return msg[:40]


@router.get("")
async def health_check():
    """深度健康检查：验证 MySQL、Milvus、Redis 连接。"""
    import concurrent.futures
    import threading

    results = {}

    def check_mysql():
        try:
            from app.repositories.task_repo import task_repo
            task_repo.get("__health_check__")
            return "ok"
        except Exception as exc:
            # 预期会报记录不存在，只要不是连接错误就算正常
            err = str(exc).lower()
            if any(kw in err for kw in ("connection", "timeout", "refused", "can't connect")):
                return f"error: {_sanitize_error(str(exc))}"
            return "ok"  # 记录不存在但连接正常

    def check_milvus():
        try:
            # 只验证 Milvus 连接可用，不创建 collection
            client = milvus_repo._get_client()
            client.list_collections()
            return "ok"
        except Exception as exc:
            return f"error: {_sanitize_error(str(exc))}"

    def check_redis():
        try:
            r = _get_redis()
            r.ping()
            return "ok"
        except Exception as exc:
            return f"error: {_sanitize_error(str(exc))}"

    checks = [
        ("mysql", check_mysql),
        ("milvus", check_milvus),
        ("redis", check_redis),
    ]

    # 并发执行所有检查，每个带超时
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {name: pool.submit(fn) for name, fn in checks}
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=_CHECK_TIMEOUT)
            except concurrent.futures.TimeoutError:
                results[name] = "error: timeout"

    all_ok = all(v == "ok" for v in results.values())
    status = "ok" if all_ok else "degraded"
    status_code = 200 if all_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={"status": status, "checks": results},
    )
