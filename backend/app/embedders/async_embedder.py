"""
异步统一多模态嵌入器模块（原生 HTTP 异步调用）

提供文本和图片的异步向量编码能力，使用 httpx.AsyncClient 直接调用 DashScope API。

模型：qwen3-vl-embedding
输出维度：2560（统一向量）

特点：
- 原生异步 HTTP 调用，不使用 run_in_executor
- 支持 httpx.AsyncClient 高并发
- 支持纯文本、纯图片、文本+图片联合编码
- 内置并发控制（Semaphore）和重试机制
"""
import asyncio
import base64
import time
from pathlib import Path

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger()

# Embedding 并发控制：DashScope API 有 QPS 限制
_embedding_semaphore: asyncio.Semaphore | None = None


def _get_embedding_semaphore() -> asyncio.Semaphore:
    """获取 Embedding 并发控制信号量。"""
    global _embedding_semaphore
    if _embedding_semaphore is None:
        _embedding_semaphore = asyncio.Semaphore(10)  # 允许 10 个并发
    return _embedding_semaphore


def image_to_base64_data_uri(image_path: str) -> str | None:
    """将图片转为 data URI 格式（同步版本，用于预处理）。"""
    import urllib.request

    try:
        if image_path.startswith(("http://", "https://")):
            with urllib.request.urlopen(image_path, timeout=30) as resp:
                data = resp.read()
        else:
            data = Path(image_path).read_bytes()

        ext = Path(image_path).suffix.lower().lstrip(".")
        mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp", "bmp": "bmp"}
        mime = f"image/{mime_map.get(ext, 'jpeg')}"

        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        logger.exception("async_image_to_base64_failed", path=image_path)
        return None


class AsyncUnifiedEmbedder:
    """
    异步统一多模态嵌入器（原生 HTTP 异步调用）。

    使用 httpx.AsyncClient 直接调用 DashScope API，不使用 run_in_executor。

    API 格式：
    - 纯文本：OpenAI 兼容格式（更快）
    - 多模态：DashScope 原生格式

    支持的编码方式：
    - encode_query: 单文本查询编码（异步）
    - encode_image_query: 单图片查询编码（异步）
    - encode_multimodal: 文本+图片联合编码（异步）
    - encode_texts: 批量文本编码（异步）
    """

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """惰性初始化异步 HTTP 客户端（优化连接池）。"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=3.0),  # 快速失败
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                ),
            )
            logger.info("async_embedder_client_init", model=settings.EMBED_MODEL_NAME)
        return self._client

    async def _call_with_retry(
        self,
        url: str,
        payload: dict,
        headers: dict,
        max_retries: int = 3,
    ) -> dict:
        """
        带重试的异步 HTTP 调用。

        参数：
            url: API URL
            payload: 请求体
            headers: 请求头
            max_retries: 最大重试次数

        返回：
            dict: API 响应 JSON
        """
        client = await self._get_client()

        for attempt in range(max_retries):
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.warning(
                    "async_embedding_api_error",
                    status_code=e.response.status_code,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

            except httpx.TimeoutException as e:
                logger.warning(
                    "async_embedding_timeout",
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

            except Exception as e:
                logger.exception("async_embedding_error", attempt=attempt + 1, error=str(e))
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError("Async embedding API call failed after retries")

    async def encode_query(self, query: str) -> list[float]:
        """
        异步编码单个文本查询（带并发控制）。

        使用 DashScope 多模态 Embedding API。

        参数：
            query: 用户查询文本

        返回：
            list[float]: 向量
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # 截断超长文本
        text = query[:20000] if len(query) > 20000 else query

        # 获取 Semaphore（并发控制）
        semaphore = _get_embedding_semaphore()
        acquired = await semaphore.acquire()

        try:
            # 判断是否使用 DashScope API（根据 URL 判断）
            is_dashscope = "dashscope.aliyuncs.com" in settings.EMBED_BASE_URL

            if is_dashscope:
                # DashScope 多模态 Embedding API 格式
                # 使用配置中的 base_url 构建 API 地址
                embed_url = f"{settings.EMBED_BASE_URL.rstrip('/')}/services/embeddings/multimodal-embedding/multimodal-embedding"
                payload = {
                    "model": settings.EMBED_MODEL_NAME,
                    "input": {
                        "contents": [{"text": text}],
                    },
                }
                headers = {
                    "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
                    "Content-Type": "application/json",
                }
                result = await self._call_with_retry(embed_url, payload, headers)

                # 解析 DashScope 响应格式
                embeddings = result.get("output", {}).get("embeddings", [])
                if not embeddings:
                    raise RuntimeError("No embedding returned from API")
                embedding = embeddings[0]["embedding"]
            else:
                # OpenAI 兼容格式（本地 GPU 服务）
                embed_url = settings.EMBED_API_URL or f"{settings.EMBED_BASE_URL.rstrip('/')}/embeddings"
                payload = {
                    "input": [text],  # 数组格式
                }
                headers = {
                    "Content-Type": "application/json",
                }
                if settings.DASHSCOPE_API_KEY:
                    headers["Authorization"] = f"Bearer {settings.DASHSCOPE_API_KEY}"

                result = await self._call_with_retry(embed_url, payload, headers)

                # 解析 OpenAI 兼容格式响应
                data = result.get("data", [])
                if not data:
                    raise RuntimeError("No embedding returned from API")
                embedding = data[0].get("embedding", [])

            if not embedding:
                raise RuntimeError("No embedding in response")

            logger.debug("async_encode_query_done", query_len=len(query), dim=len(embedding))
            return embedding

        finally:
            semaphore.release()

    async def encode_image_query(self, image_path: str) -> list[float]:
        """
        异步编码单张图片作为查询向量。

        参数：
            image_path: 图片路径（本地路径或 HTTP URL）

        返回：
            list[float]: 向量
        """
        # 判断是否使用 DashScope API
        is_dashscope = "dashscope.aliyuncs.com" in settings.EMBED_BASE_URL

        if not is_dashscope:
            # 本地 GPU 服务不支持图片，返回零向量
            logger.warning(
                "async_encode_image_query_not_supported",
                message="OpenAI-compatible embedding API does not support images. Use multimodal service.",
                image_path=image_path,
            )
            return [0.0] * settings.UNIFIED_VECTOR_DIM

        # DashScope 多模态 Embedding API
        # 预处理图片为 data URI 或使用 URL
        if image_path.startswith(("http://", "https://")):
            image_ref = image_path
        else:
            image_ref = image_to_base64_data_uri(image_path)
            if image_ref is None:
                raise RuntimeError(f"Cannot read image: {image_path}")

        # 使用配置中的 base_url 构建 API 地址
        embed_url = f"{settings.EMBED_BASE_URL.rstrip('/')}/services/embeddings/multimodal-embedding/multimodal-embedding"
        payload = {
            "model": settings.EMBED_MODEL_NAME,
            "input": {
                "contents": [{"image": image_ref}],
            },
        }
        headers = {
            "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        }

        result = await self._call_with_retry(embed_url, payload, headers)

        # 解析响应
        embeddings = result.get("output", {}).get("embeddings", [])
        if not embeddings:
            raise RuntimeError("No embedding returned from API")

        embedding = embeddings[0]["embedding"]
        logger.debug("async_encode_image_done", dim=len(embedding))
        return embedding

    async def encode_multimodal(self, text: str, image_path: str) -> list[float]:
        """
        异步联合编码文本和图片为单个多模态向量。

        参数：
            text: 文本内容
            image_path: 图片路径

        返回：
            list[float]: 融合向量
        """
        # 判断是否使用 DashScope API
        is_dashscope = "dashscope.aliyuncs.com" in settings.EMBED_BASE_URL

        if not is_dashscope:
            # 本地 GPU 服务不支持多模态，仅编码文本
            logger.warning(
                "async_encode_multimodal_partial",
                message="OpenAI-compatible embedding API does not support images. Encoding text only.",
                text_len=len(text),
            )
            return await self.encode_query(text)

        # DashScope 多模态 Embedding API
        # 预处理图片
        if image_path.startswith(("http://", "https://")):
            image_ref = image_path
        else:
            image_ref = image_to_base64_data_uri(image_path)
            if image_ref is None:
                raise RuntimeError(f"Cannot read image: {image_path}")

        # 截断超长文本
        truncated_text = text[:20000] if len(text) > 20000 else text

        # 使用配置中的 base_url 构建 API 地址
        embed_url = f"{settings.EMBED_BASE_URL.rstrip('/')}/services/embeddings/multimodal-embedding/multimodal-embedding"
        payload = {
            "model": settings.EMBED_MODEL_NAME,
            "input": {
                "contents": [
                    {"text": truncated_text},
                    {"image": image_ref},
                ],
            },
        }
        headers = {
            "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        }

        result = await self._call_with_retry(embed_url, payload, headers)

        # 解析响应
        embeddings = result.get("output", {}).get("embeddings", [])
        if not embeddings:
            raise RuntimeError("No embedding returned from API")

        embedding = embeddings[0]["embedding"]
        logger.debug("async_encode_multimodal_done", text_len=len(text), dim=len(embedding))
        return embedding

    async def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """
        异步批量编码文本列表。

        使用 DashScope 原生 API 并发调用。

        参数：
            texts: 文本字符串列表

        返回：
            list[list[float]]: 向量列表
        """
        if not texts:
            return []

        # DashScope 原生 API 不支持批量输入，需要并发调用
        # 最多 10 个并发（DashScope QPS 限制）
        tasks = [self.encode_query(t) for t in texts]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        results = []
        for i, emb in enumerate(embeddings):
            if isinstance(emb, Exception):
                logger.error("async_encode_texts_failed", index=i, error=str(emb))
                results.append([0.0] * settings.UNIFIED_VECTOR_DIM)
            else:
                results.append(emb)

        logger.debug("async_encode_texts_done", count=len(texts))
        return results

    async def encode_images(self, image_paths: list[str]) -> list[list[float]]:
        """
        异步批量编码图片列表。

        使用并发调用多模态 API。

        参数：
            image_paths: 图片路径列表

        返回：
            list[list[float]]: 向量列表
        """
        if not image_paths:
            return []

        # 并发编码每张图片
        tasks = [self.encode_image_query(path) for path in image_paths]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        results = []
        for i, emb in enumerate(embeddings):
            if isinstance(emb, Exception):
                logger.error("async_encode_image_batch_failed", index=i, error=str(emb))
                # 使用空向量作为占位（或跳过）
                results.append([0.0] * settings.UNIFIED_VECTOR_DIM)
            else:
                results.append(emb)

        logger.debug("async_encode_images_done", count=len(image_paths))
        return results

    async def encode_multimodal_batch(
        self,
        items: list[tuple[str, str | None]],
    ) -> list[list[float]]:
        """
        异步批量联合编码文本+图片。

        参数：
            items: 元组列表，每个元素为 (text, image_path)
                   - image_path 为 None 表示纯文本
                   - image_path 有值表示文本+图片

        返回：
            list[list[float]]: 向量列表
        """
        if not items:
            return []

        # 分组处理：纯文本 vs 多模态
        text_only_indices = []
        text_only_texts = []
        multimodal_indices = []
        multimodal_items = []

        for i, (text, image_path) in enumerate(items):
            if image_path:
                multimodal_indices.append(i)
                multimodal_items.append((text, image_path))
            else:
                text_only_indices.append(i)
                text_only_texts.append(text)

        # 并发处理两组
        text_embeddings = []
        multimodal_embeddings = []

        if text_only_texts:
            text_embeddings = await self.encode_texts(text_only_texts)

        if multimodal_items:
            tasks = [self.encode_multimodal(text, path) for text, path in multimodal_items]
            multimodal_embeddings = await asyncio.gather(*tasks, return_exceptions=True)
            # 处理异常
            multimodal_embeddings = [
                emb if not isinstance(emb, Exception) else [0.0] * settings.UNIFIED_VECTOR_DIM
                for emb in multimodal_embeddings
            ]

        # 合并结果（按原顺序）
        results = [None] * len(items)
        for i, idx in enumerate(text_only_indices):
            results[idx] = text_embeddings[i]
        for i, idx in enumerate(multimodal_indices):
            results[idx] = multimodal_embeddings[i]

        logger.debug(
            "async_encode_multimodal_batch_done",
            total=len(items),
            text_only=len(text_only_indices),
            multimodal=len(multimodal_indices),
        )
        return results

    async def close(self) -> None:
        """关闭 HTTP 客户端（清理资源）。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# 单例模式：全局共享一个 AsyncUnifiedEmbedder 实例
async_embedder = AsyncUnifiedEmbedder()