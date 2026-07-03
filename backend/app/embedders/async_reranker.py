"""异步重排序器：使用 httpx.AsyncClient 调用 DashScope Rerank API（带并发控制）。"""
import asyncio
import hashlib
import httpx
import structlog

from app.config import settings

logger = structlog.get_logger()

# Rerank 并发控制：DashScope 限制 10 QPS，设置 Semaphore 为 8 接近上限
_rerank_semaphore: asyncio.Semaphore | None = None

# Rerank 结果缓存（可选，用于相同 query+documents 的快速返回）
_rerank_cache: dict = {}
_cache_max_size = 500  # 缓存 500 条热门组合


def _get_rerank_semaphore() -> asyncio.Semaphore:
    """获取 rerank 并发控制信号量。"""
    global _rerank_semaphore
    if _rerank_semaphore is None:
        _rerank_semaphore = asyncio.Semaphore(8)  # 允许 8 个并发（接近 10 QPS）
    return _rerank_semaphore


def _get_cache_key(query: str, doc_hashes: tuple) -> str:
    """生成缓存 key。"""
    combined = query + "".join(doc_hashes)
    return hashlib.md5(combined.encode()).hexdigest()


class AsyncReranker:
    """
    异步重排序器（带并发控制 + 缓存 + 降级）。

    特性：
    1. Semaphore 并发控制：最多 5 个并发 rerank 调用
    2. LRU 缓存：热门查询结果缓存，减少 API 调用
    3. 降级策略：限流/超时时返回原始结果，不影响用户体验

    DashScope Rerank API 限制：10 QPS
    """

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """惰性初始化异步 HTTP 客户端（优化连接池）。"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=3.0),  # 快速失败
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=10,
                    keepalive_expiry=30.0,
                ),
            )
            logger.info("async_reranker_client_init", model="qwen3-vl-rerank")
        return self._client

    async def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        异步重排序（带并发控制 + 缓存 + 降级）。

        流程：
        1. 检查缓存，命中则直接返回
        2. 获取 Semaphore，最多等待 1 秒
        3. 调用 DashScope Rerank API
        4. 缓存结果
        5. 返回重排序后的候选列表

        参数：
            query: 查询文本
            candidates: 候选文档列表（每个包含 content 字段）

        返回：
            list[dict]: 重排序后的候选列表（或原始列表，如果降级）
        """
        global _rerank_cache

        if not candidates or not settings.RERANK_ENABLED:
            return candidates

        # 1. 检查缓存
        doc_hashes = tuple(
            hashlib.md5(c.get("content", "").encode()).hexdigest()[:8]
            for c in candidates
        )
        cache_key = _get_cache_key(query, doc_hashes)

        if cache_key in _rerank_cache:
            logger.debug("async_rerank_cache_hit", cache_key=cache_key[:8])
            cached_scores = _rerank_cache[cache_key]
            for idx, score in cached_scores.items():
                if idx < len(candidates):
                    candidates[idx]["rerank_score"] = score
            candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
            return candidates

        # 2. 获取 Semaphore（最多等待 2秒，更多请求能等到机会）
        semaphore = _get_rerank_semaphore()
        acquired = False

        try:
            acquired = await asyncio.wait_for(semaphore.acquire(), timeout=2.0)
        except asyncio.TimeoutError:
            # 等待超时：降级，返回原始结果
            logger.warning(
                "async_rerank_semaphore_timeout",
                query_len=len(query),
                candidate_count=len(candidates),
                fallback="Queue full after 2s, skip rerank",
            )
            return candidates

        if not acquired:
            return candidates

        try:
            # 3. 调用 API（自动适配 DashScope 或本地 GPU 服务）
            client = await self._get_client()

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

            response = await client.post(
                settings.RERANK_API_URL,
                json=payload,
                headers=headers,
            )

            # 429 限流：降级处理
            if response.status_code == 429:
                logger.warning(
                    "async_rerank_rate_limited",
                    query_len=len(query),
                    candidate_count=len(candidates),
                )
                return candidates

            response.raise_for_status()
            result = response.json()

            # 4. 解析结果（兼容多种格式）
            # 格式1: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
            # 格式2: {"output": {"results": [{"index": 0, "relevance_score": 0.95}, ...]}}
            if "results" in result:
                rerank_results = result.get("results", [])
            elif "output" in result:
                rerank_results = result.get("output", {}).get("results", [])
            else:
                logger.warning("async_rerank_unknown_format", result_keys=list(result.keys()))
                rerank_results = []

            cached_scores = {}

            for r in rerank_results:
                idx = r.get("index", 0)
                score = r.get("relevance_score", r.get("score", 0.0))
                cached_scores[idx] = float(score)
                if idx < len(candidates):
                    candidates[idx]["rerank_score"] = float(score)

            # 降序排列
            candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)

            # 5. 写入缓存（LRU 淘汰）
            if len(_rerank_cache) >= _cache_max_size:
                keys_to_remove = list(_rerank_cache.keys())[:_cache_max_size // 2]
                for k in keys_to_remove:
                    _rerank_cache.pop(k, None)

            _rerank_cache[cache_key] = cached_scores

            logger.debug(
                "async_rerank_completed",
                query_len=len(query),
                candidate_count=len(candidates),
                top_score=candidates[0].get("rerank_score", 0) if candidates else 0,
            )

            return candidates

        except httpx.HTTPStatusError as e:
            logger.error(
                "async_rerank_api_error",
                status_code=e.response.status_code,
                error=str(e),
            )
            return candidates

        except Exception as e:
            logger.exception("async_rerank_failed", error=str(e))
            return candidates

        finally:
            if acquired:
                semaphore.release()

    async def close(self) -> None:
        """关闭 HTTP 客户端（清理资源）。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# 单例
async_reranker = AsyncReranker()