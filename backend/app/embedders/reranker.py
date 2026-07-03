"""重排序器：调用 Theta Qwen3-VL-Rerank API 对候选段落重排序。"""
import asyncio
import hashlib
import httpx
import structlog
from functools import lru_cache

from app.config import settings

logger = structlog.get_logger()

# Rerank 并发控制：限制同时进行的 rerank 调用数量
# DashScope rerank API 限制 10 QPS，设置 Semaphore 为 5 防止瞬时过载
_rerank_semaphore: asyncio.Semaphore | None = None


def _get_rerank_semaphore() -> asyncio.Semaphore:
    """获取 rerank 并发控制信号量（懒加载）。"""
    global _rerank_semaphore
    if _rerank_semaphore is None:
        # 并发限制为 5，避免瞬时请求超过 10 QPS
        _rerank_semaphore = asyncio.Semaphore(5)
    return _rerank_semaphore


class Reranker:
    """
    使用 Theta Qwen3-VL-Rerank API 的交叉编码器重排序器。
    对候选段落按查询相关性进行重排序，提升 top-k 准确率。
    """

    def __init__(self):
        self._client = None
        self._loaded = False

    def _load_client(self):
        """延迟初始化 HTTP 客户端。"""
        if self._loaded:
            return

        if not settings.RERANK_ENABLED:
            logger.info("reranker_disabled_by_config")
            self._loaded = False
            return

        # 创建 HTTP 客户端
        self._client = httpx.Client(timeout=30.0)
        self._loaded = True
        logger.info("reranker_client_init", model="qwen3-vl-rerank")

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        调用 Theta API 对候选字典重排序。
        每个候选必须包含 'content' 键。

        返回按重排序分数降序排列的候选列表。

        注意：此方法是同步版本，供 Celery 后台任务使用。
        """
        if not candidates or not settings.RERANK_ENABLED:
            return candidates

        self._load_client()
        if not self._loaded or self._client is None:
            logger.warning("reranker_not_available")
            return candidates

        # 构建请求（自动适配 DashScope 或本地 GPU 服务）
        documents = [c.get("content", "") for c in candidates]
        rerank_url = settings.RERANK_API_URL
        model_name = getattr(settings, "RERANK_MODEL_NAME", "qwen3-vl-rerank")

        # 判断是否使用 DashScope API
        is_dashscope = "dashscope.aliyuncs.com" in rerank_url

        if is_dashscope:
            # DashScope 格式
            payload = {
                "model": model_name,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": len(candidates),
                },
            }
        else:
            # 本地 GPU 服务格式
            payload = {
                "model": model_name,
                "query": query,
                "documents": documents,
            }

        headers = {
            "Content-Type": "application/json",
        }
        if settings.DASHSCOPE_API_KEY:
            headers["Authorization"] = f"Bearer {settings.DASHSCOPE_API_KEY}"

        try:
            response = self._client.post(rerank_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # 解析返回结果（兼容两种格式）
            if "results" in result:
                results = result.get("results", [])
            elif "output" in result:
                results = result.get("output", {}).get("results", [])
            else:
                logger.warning("rerank_unknown_format", result_keys=list(result.keys()))
                results = []

            for r in results:
                idx = r.get("index", 0)
                score = r.get("relevance_score", r.get("score", 0.0))
                if idx < len(candidates):
                    candidates[idx]["rerank_score"] = float(score)

            # 降序排列
            candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)

            logger.debug("rerank_completed",
                query_len=len(query),
                candidate_count=len(candidates),
                top_score=candidates[0].get("rerank_score", 0) if candidates else 0)

        except httpx.HTTPStatusError as e:
            # 429 限流错误：降级处理，返回原始结果
            if e.response.status_code == 429:
                logger.warning("rerank_rate_limited",
                    query_len=len(query),
                    candidate_count=len(candidates),
                    error="Rate limit exceeded, fallback to original order")
            else:
                logger.error("rerank_api_error",
                    status_code=e.response.status_code,
                    error=str(e))
        except Exception as e:
            logger.exception("reranking_failed",
                candidate_count=len(candidates),
                error=str(e))

        return candidates


# ============================================================
# 异步版本：供 FastAPI 搜索接口使用（带并发控制 + 缓存）
# ============================================================

@lru_cache(maxsize=1000)
def _get_cache_key(query: str, doc_hashes: tuple) -> str:
    """生成缓存 key（query + 文档内容 hash）。"""
    combined = query + "".join(doc_hashes)
    return hashlib.md5(combined.encode()).hexdigest()


# Rerank 结果缓存（LRU，最多缓存 1000 条）
_rerank_cache: dict = {}
_cache_max_size = 1000


async def async_rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    异步重排序（带并发控制 + 缓存 + 降级）。

    特性：
    1. Semaphore 并发控制：最多 5 个并发 rerank 调用
    2. LRU 缓存：热门查询结果缓存，减少 API 调用
    3. 降级策略：限流时直接返回原始结果，不影响用户体验

    参数：
        query: 查询文本
        candidates: 候选文档列表（每个包含 content 字段）

    返回：
        重排序后的候选列表（或原始列表，如果限流降级）
    """
    global _rerank_cache

    if not candidates or not settings.RERANK_ENABLED:
        return candidates

    # 1. 检查缓存
    doc_hashes = tuple(hashlib.md5(c.get("content", "").encode()).hexdigest()[:8] for c in candidates)
    cache_key = _get_cache_key(query, doc_hashes)

    if cache_key in _rerank_cache:
        logger.debug("rerank_cache_hit", cache_key=cache_key[:8])
        cached_scores = _rerank_cache[cache_key]
        for i, score in cached_scores.items():
            if i < len(candidates):
                candidates[i]["rerank_score"] = score
        candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
        return candidates

    # 2. Semaphore 并发控制
    semaphore = _get_rerank_semaphore()
    acquired = False

    try:
        # 尝试获取信号量（最多等待 2 秒）
        acquired = await asyncio.wait_for(semaphore.acquire(), timeout=2.0)
    except asyncio.TimeoutError:
        # 等待超时：降级，返回原始结果
        logger.warning("rerank_semaphore_timeout",
            query_len=len(query),
            candidate_count=len(candidates),
            fallback="Too many concurrent requests, skip rerank")
        return candidates

    if not acquired:
        return candidates

    try:
        # 3. 调用 rerank API
        reranker._load_client()
        if not reranker._loaded or reranker._client is None:
            return candidates

        documents = [c.get("content", "") for c in candidates]
        rerank_url = settings.RERANK_API_URL
        model_name = getattr(settings, "RERANK_MODEL_NAME", "qwen3-vl-rerank")

        # 判断是否使用 DashScope API
        is_dashscope = "dashscope.aliyuncs.com" in rerank_url

        if is_dashscope:
            # DashScope 格式
            payload = {
                "model": model_name,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": len(candidates),
                },
            }
        else:
            # 本地 GPU 服务格式
            payload = {
                "model": model_name,
                "query": query,
                "documents": documents,
            }

        headers = {
            "Content-Type": "application/json",
        }
        if settings.DASHSCOPE_API_KEY:
            headers["Authorization"] = f"Bearer {settings.DASHSCOPE_API_KEY}"

        # 使用 httpx.AsyncClient 发起请求
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(settings.RERANK_API_URL, json=payload, headers=headers)

            if response.status_code == 429:
                # 429 限流：降级处理
                logger.warning("rerank_rate_limited_async",
                    query_len=len(query),
                    candidate_count=len(candidates))
                return candidates

            response.raise_for_status()
            result = response.json()

        # 4. 解析结果（兼容多种格式）
        if "results" in result:
            results = result.get("results", [])
        elif "output" in result:
            results = result.get("output", {}).get("results", [])
        else:
            logger.warning("rerank_async_unknown_format", result_keys=list(result.keys()))
            results = []

        cached_scores = {}

        for r in results:
            idx = r.get("index", 0)
            score = r.get("relevance_score", r.get("score", 0.0))
            cached_scores[idx] = float(score)
            if idx < len(candidates):
                candidates[idx]["rerank_score"] = float(score)

        # 降序排列
        candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)

        # 5. 写入缓存（LRU 淘汰）
        if len(_rerank_cache) >= _cache_max_size:
            # 淘汰最早的一半缓存
            keys_to_remove = list(_rerank_cache.keys())[:_cache_max_size // 2]
            for k in keys_to_remove:
                _rerank_cache.pop(k, None)

        _rerank_cache[cache_key] = cached_scores

        logger.debug("rerank_async_completed",
            query_len=len(query),
            candidate_count=len(candidates),
            top_score=candidates[0].get("rerank_score", 0) if candidates else 0)

        return candidates

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning("rerank_rate_limited_async", fallback="skip rerank")
        else:
            logger.error("rerank_api_error_async", status_code=e.response.status_code)
        return candidates

    except Exception as e:
        logger.exception("rerank_async_failed", error=str(e))
        return candidates

    finally:
        if acquired:
            semaphore.release()


# 单例
reranker = Reranker()
