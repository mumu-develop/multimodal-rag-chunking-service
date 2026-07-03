import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（backend/ 所在目录），确保路径不依赖工作目录
_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> str:
    """将相对路径转为基于 backend/ 目录的绝对路径。"""
    if os.path.isabs(p):
        return p
    return str(_BACKEND_DIR / p)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # MySQL 数据库
    DATABASE_URI: str = "mysql+pymysql://root:root@localhost:3306/rag_db"

    # Milvus
    MILVUS_LOCAL_DB: str = ""  # Milvus Lite 本地文件路径，如 "./milvus_data/milvus.db"；为空则使用 host:port
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: str = "19530"
    MILVUS_COLLECTION_NAME: str = "rag_chunks"

    # Redis（Celery 消息代理与后端）
    REDIS_URL: str = "redis://localhost:6379/0"

    # 文件上传目录
    UPLOAD_DIR: str = _resolve("./uploads")

    # VLM 服务（用于图像描述生成）
    VLM_API_URL: str = "http://localhost:8000/v1/chat/completions"

    # LLM（文本摘要，复用 VLM_API_URL 指向的 OpenAI 兼容 API）
    LLM_API_URL: str = ""
    LLM_TIMEOUT: int = 30
    LLM_MODEL_NAME: str = "qwen-plus"  # 意图分类、文本摘要使用的模型

    MAX_FILE_SIZE_MB: int = 50
    # 支持的文档类型
    ALLOWED_EXTENSIONS: set[str] = {"pdf", "docx", "pptx", "md", "xlsx", "xls"}

    # Celery
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND: str = ""

    # ============================================================
    # 文档解析
    # ============================================================
    PARSING_STRATEGY: str = "hi_res"  # hi_res | fast
    EXTRACT_IMAGES: bool = True
    IMAGE_OUTPUT_DIR: str = _resolve("./uploads/images")

    # ============================================================
    # 分块策略
    # ============================================================
    # 切片策略选项：
    # - smart: 智能切分（标题 + 语义 + 递归）- 最高质量，默认
    # - heading: 只按标题切分 - 保持文档结构
    # - semantic: 只按语义切分 - 保持语义连贯
    # - recursive: 只按递归切分 - 最简单快速
    # - fixed: 固定长度切分 - 最基础方案
    CHUNK_STRATEGY: str = "smart"  # smart | heading | semantic | recursive | fixed

    CHUNK_SIZE_TEXT: int = 800
    CHUNK_OVERLAP_TEXT: int = 150
    CHUNK_SIZE_TABLE: int = 1600

    # Token 级控制（借鉴 RAGFlow）
    CHUNK_TOKEN_LIMIT: int = 1024  # 嵌入模型 token 上限（原 512 调整为 1024）
    CHUNK_TOKEN_OVERLAP: int = 100  # token 级重叠（原 50 调整为 100）
    USE_TOKEN_COUNTING: bool = True  # 启用 token 计数（推荐）

    # 垃圾过滤（借鉴 RAGFlow）
    GARBAGE_FILTER_ENABLED: bool = True  # 启用垃圾文本过滤

    # ============================================================
    # Embedding（支持 DashScope 或本地 GPU 服务）
    # ============================================================
    EMBED_MODEL_NAME: str = "Qwen/Qwen3-VL-Embedding-2B"
    EMBED_BASE_URL: str = "http://127.0.0.1:6006/embedding/v1"  # Embedding API 端点
    EMBED_API_URL: str = ""  # 完整 Embedding URL（为空则使用 EMBED_BASE_URL + /embeddings）
    DASHSCOPE_API_KEY: str = ""  # API Key（本地服务可为空）
    UNIFIED_VECTOR_DIM: int = 2048  # 向量维度（根据模型调整：BGE-M3=1024, Qwen3-VL-Embedding-2B=2048）

    # ============================================================
    # 重排序（支持 DashScope 或本地 GPU 服务）
    # ============================================================
    RERANK_ENABLED: bool = True
    RERANK_API_URL: str = "http://127.0.0.1:6006/reranker/v1/rerank"  # Rerank API 端点
    RERANK_MODEL_NAME: str = "qwen3-vl-reranker-2b"  # Rerank 模型名称
    RERANK_TOP_N: int = 50

    # ============================================================
    # BM25
    # ============================================================
    BM25_ENABLED: bool = True
    HYBRID_SEARCH_ALPHA: float = 0.5  # 0=稀疏检索, 1=稠密检索

    # ============================================================
    # Milvus 批量配置
    # ============================================================
    MILVUS_BATCH_SIZE: int = 100

    # ============================================================
    # 身份认证
    # ============================================================
    API_KEYS: str = ""  # 逗号分隔，如 "key1,key2"，空表示不启用认证

    # ============================================================
    # RAGAS 评估 LLM
    # ============================================================
    EVAL_MODEL_NAME: str = "qwen3.6-plus"
    EVAL_API_KEY: str = ""
    EVAL_BASE_URL: str = "https://coding.dashscope.aliyuncs.com/v1"
    EVAL_MAX_WORKERS: int = 4

    # ============================================================
    # OCR 服务（PDF 图片化识别）
    # ============================================================
    OCR_API_URL: str = "http://localhost:6006/v1"  # OCR 服务地址（OpenAI 兼容格式）
    OCR_TIMEOUT: int = 60  # OCR 超时时间（秒）
    OCR_MODEL_NAME: str = "deepseek-ai/DeepSeek-OCR"  # OCR 模型名称（可通过 .env 覆盖）
    OCR_MAX_WORKERS: int = 4  # OCR 并发线程数
    OCR_MAX_TOKENS: int = 4096  # OCR 最大输出 token 数
    PDF_IMAGE_DPI: int = 200  # PDF 转图片 DPI（越高越清晰但更慢）

    def model_post_init(self, __context) -> None:
        # 将 MILVUS_LOCAL_DB 的相对路径转为绝对路径
        if self.MILVUS_LOCAL_DB and not os.path.isabs(self.MILVUS_LOCAL_DB):
            self.MILVUS_LOCAL_DB = _resolve(self.MILVUS_LOCAL_DB)
        # 如果未显式设置 LLM_API_URL，默认复用 VLM_API_URL
        if not self.LLM_API_URL:
            self.LLM_API_URL = self.VLM_API_URL
        # 如果未显式设置，默认 Celery URL 使用 REDIS_URL
        if not self.CELERY_BROKER_URL:
            self.CELERY_BROKER_URL = self.REDIS_URL
        if not self.CELERY_RESULT_BACKEND:
            self.CELERY_RESULT_BACKEND = self.REDIS_URL

    @property
    def api_key_set(self) -> set[str]:
        """将 API_KEYS 解析为集合，过滤空字符串。"""
        return {k.strip() for k in self.API_KEYS.split(",") if k.strip()}

    @property
    def auth_enabled(self) -> bool:
        return len(self.api_key_set) > 0


settings = Settings()