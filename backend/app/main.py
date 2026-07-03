import asyncio
import json
import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.middleware.analytics import AnalyticsMiddleware
from app.api.middleware.auth import AuthMiddleware
from app.api.middleware.rbac import RbacMiddleware
from app.api.middleware.trace import TraceMiddleware
from app.api.routers import (
    chunks, dashboard, documents, evaluation, health,
    intent_tree, knowledge_bases, search, tasks, upload, users, audit, sse,
)
from app.api.routers.faq import router as faq_router
from app.api.routers.sse import start_simulate_task
from app.config import settings
from app.repositories.milvus_repo import milvus_repo

# ============================================================
# 日志配置
# ============================================================
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
)
logger = structlog.get_logger()


# ============================================================
# 插件加载
# ============================================================

def _load_parser_plugins() -> None:
    """从 Python entry points 加载并注册插件解析器。"""
    try:
        from app.chunkers import register_parser
        logger.info("parser_registry_loaded")
    except ImportError as e:
        logger.error("parser_registry_import_failed", error=str(e))
        raise

    try:
        import importlib.metadata as metadata
    except ImportError:
        import importlib_metadata as metadata

    try:
        for ep in metadata.entry_points(group="rag.parsers"):
            try:
                parser_cls = ep.load()
                register_parser(ep.name, parser_cls)
                logger.info("parser_plugin_loaded", ext=ep.name, parser=parser_cls.__name__)
            except Exception:
                logger.exception("parser_plugin_load_failed", ext=ep.name)
    except Exception:
        # 无插件或 entry_points 不支持，静默跳过
        logger.debug("no_parser_plugins_found")


# ============================================================
# 生命周期：启动预热 + 关闭
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 创建目录
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.IMAGE_OUTPUT_DIR, exist_ok=True)

    logger.info("app_startup_begin", auth_enabled=settings.auth_enabled)

    # 执行数据库迁移
    try:
        from alembic.config import Config
        from alembic import command
        alembic_cfg = Config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "alembic.ini"))
        command.upgrade(alembic_cfg, "head")
        logger.info("db_migration_completed")
    except Exception:
        logger.exception("db_migration_failed")

    # 加载插件解析器（entry points）
    _load_parser_plugins()

    # 预热 Milvus 连接（只验证连接可用，不创建 collection）
    try:
        client = milvus_repo._get_client()
        logger.info("app_warmup_milvus", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT, collections=client.list_collections())
    except Exception as exc:
        logger.warning("app_warmup_milvus_failed", error=str(exc))

    # Embedding 和重排序改为首次请求时懒加载，跳过启动 warmup

    # 启动后台任务：日活模拟
    asyncio.create_task(start_simulate_task())
    logger.info("daily_active_task_started")

    logger.info("app_startup_done", upload_dir=settings.UPLOAD_DIR)
    yield
    # 关闭
    logger.info("app_stopped")


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title="Multimodal RAG Chunking Service", version="0.3.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 中间件（添加顺序决定包裹层级：后添加的在外层）
# 请求流向: Analytics → Rbac → Auth → Trace → CORS → App
app.add_middleware(TraceMiddleware)       # 最内层：注入 trace_id
app.add_middleware(AuthMiddleware)        # 认证 + 设置 user_role
app.add_middleware(RbacMiddleware)        # 角色权限校验
app.add_middleware(AnalyticsMiddleware)   # 最外层：耗时统计 + 数据采集

# 全局异常处理
from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    trace_id = getattr(request.state, "trace_id", "")
    logger.exception("unhandled_error", trace_id=trace_id, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "code": "INTERNAL_ERROR",
            "message": "Internal server error",
            "trace_id": trace_id,
            "data": None,
        },
    )


# ============================================================
# 路由注册（APIRouter 按模块拆分）
# ============================================================

# 公开路由（无认证）
app.include_router(health.router, prefix="/health", tags=["health"])

# API v1 路由（需认证）
app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["tasks"])
app.include_router(chunks.router, prefix="/api/v1", tags=["chunks"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(evaluation.router, prefix="/api/v1/evaluate", tags=["evaluate"])
# 新增：管理后台路由
app.include_router(knowledge_bases.router, prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(intent_tree.router, prefix="/api/v1/intent-tree", tags=["intent-tree"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(faq_router, prefix="/api/v1/faq", tags=["faq"])
app.include_router(audit.router, prefix="/api/v1/audit", tags=["audit"])
app.include_router(sse.router, prefix="/api/v1/sse", tags=["sse"])

# ============================================================
# 静态文件服务（图片访问）
# ============================================================
# 挂载 uploads 目录，使前端可以访问 OCR 生成的图片
# 访问方式：http://localhost:8000/uploads/images/xxx.png
uploads_dir = settings.UPLOAD_DIR
images_dir = settings.IMAGE_OUTPUT_DIR
if os.path.exists(uploads_dir):
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
    logger.info("static_files_mounted", path=uploads_dir)
if os.path.exists(images_dir):
    # 如果 images_dir 不是 uploads_dir 的子目录，单独挂载
    if not images_dir.startswith(uploads_dir):
        app.mount("/images", StaticFiles(directory=images_dir), name="images")
        logger.info("static_images_mounted", path=images_dir)


# ============================================================
# 向后兼容：保留旧模块级别导入
# 旧代码可能直接从 main.py 导入 app、_ok、_err 等
# ============================================================
def _ok(data=None, trace_id: str = "", message: str | None = None, code: str = "OK") -> dict:
    return {"success": True, "code": code, "message": message, "trace_id": trace_id, "data": data}


def _err(trace_id: str = "", message: str = "Error", code: str = "ERROR") -> dict:
    return {"success": False, "code": code, "message": message, "trace_id": trace_id, "data": None}
