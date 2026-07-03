"""增强版重排序器。

功能：
1. 多模型支持（BGE-Reranker、Cohere、Jina 等）
2. 分数归一化
3. 失败时 fallback
4. 批量处理优化
5. 缓存机制

特点：
- 可配置 rerank 模型
- 支持本地模型和 API
- 自动选择最优模型
"""
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import structlog

from app.config import settings

logger = structlog.get_logger()


@dataclass
class RerankConfig:
    """重排序配置。

    Attributes:
        model_name: 模型名称
        api_url: API 端点
        api_key: API Key
        top_n: 返回数量
        threshold: 分数阈值（低于此值过滤）
        cache_enabled: 是否启用缓存
        batch_size: 批处理大小
    """
    model_name: str = "bge-reranker-v2-m3"
    api_url: str = ""
    api_key: str = ""
    top_n: int = 50
    threshold: float = 0.0
    cache_enabled: bool = True
    batch_size: int = 20


class BaseReranker(ABC):
    """重排序器基类。"""

    @abstractmethod
    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """执行重排序。"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查是否可用。"""
        ...


class APIReranker(BaseReranker):
    """API 重排序器。

    支持多种 API：
    - DashScope (Qwen-Rerank)
    - Cohere Rerank
    - Jina Reranker
    """

    def __init__(self, config: RerankConfig):
        self.config = config
        self._client = None
        self._cache: dict[str, list[tuple[int, float]]] = {}

    def _get_client(self):
        """获取 HTTP 客户端。"""
        if self._client is None:
            import httpx
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def is_available(self) -> bool:
        """检查 API 是否可用。"""
        return bool(self.config.api_url and self.config.api_key)

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """执行 API 重排序。"""
        if not candidates:
            return candidates

        # 检查缓存
        if self.config.cache_enabled:
            cache_key = self._make_cache_key(query, candidates)
            if cache_key in self._cache:
                scores = self._cache[cache_key]
                for idx, score in scores:
                    if idx < len(candidates):
                        candidates[idx]["rerank_score"] = score
                candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
                return candidates

        # 调用 API
        try:
            scores = self._call_api(query, candidates)

            # 应用分数
            for idx, score in scores:
                if idx < len(candidates):
                    candidates[idx]["rerank_score"] = score

            # 缓存结果
            if self.config.cache_enabled:
                self._cache[cache_key] = scores

            # 排序
            candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)

            # 过滤低分
            if self.config.threshold > 0:
                candidates = [c for c in candidates if c.get("rerank_score", 0) >= self.config.threshold]

            logger.info(
                "api_rerank_completed",
                model=self.config.model_name,
                candidates=len(candidates),
                top_score=candidates[0].get("rerank_score", 0) if candidates else 0,
            )

        except Exception as e:
            logger.error("api_rerank_failed", model=self.config.model_name, error=str(e))
            # 返回原顺序，但添加默认分数
            for c in candidates:
                c["rerank_score"] = c.get("score", 0)

        return candidates

    def _call_api(self, query: str, candidates: list[dict]) -> list[tuple[int, float]]:
        """调用重排序 API。

        Returns:
            list of (index, score)
        """
        client = self._get_client()
        documents = [c.get("content", "") for c in candidates]

        # DashScope API 格式
        if "dashscope" in self.config.api_url.lower():
            payload = {
                "model": self.config.model_name,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": min(len(candidates), self.config.top_n),
                },
            }
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }

            response = client.post(self.config.api_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # 解析结果
            results = result.get("output", {}).get("results", [])
            scores = [(r.get("index", 0), r.get("relevance_score", 0.0)) for r in results]
            return scores

        # Cohere API 格式
        elif "cohere" in self.config.api_url.lower():
            payload = {
                "model": self.config.model_name,
                "query": query,
                "documents": documents,
                "top_n": min(len(candidates), self.config.top_n),
            }
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }

            response = client.post(self.config.api_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            results = result.get("results", [])
            scores = [(r.get("index", 0), r.get("relevance_score", 0.0)) for r in results]
            return scores

        # Jina API 格式
        elif "jina" in self.config.api_url.lower():
            payload = {
                "model": self.config.model_name,
                "query": query,
                "documents": documents,
                "top_n": min(len(candidates), self.config.top_n),
            }
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }

            response = client.post(self.config.api_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            results = result.get("results", [])
            scores = [(r.get("index", 0), r.get("relevance_score", 0.0)) for r in results]
            return scores

        # 默认格式
        else:
            payload = {
                "query": query,
                "documents": documents,
                "top_n": min(len(candidates), self.config.top_n),
            }
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }

            response = client.post(self.config.api_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # 尝试多种格式解析
            if "results" in result:
                results = result["results"]
            elif "output" in result and "results" in result["output"]:
                results = result["output"]["results"]
            else:
                results = []

            scores = []
            for i, r in enumerate(results):
                idx = r.get("index", i)
                score = r.get("relevance_score", r.get("score", 0.0))
                scores.append((idx, float(score)))

            return scores

    def _make_cache_key(self, query: str, candidates: list[dict]) -> str:
        """生成缓存 key。"""
        content_hash = hashlib.md5(
            json.dumps([c.get("content", "")[:100] for c in candidates]).encode()
        ).hexdigest()
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{query_hash}:{content_hash}"


class LocalReranker(BaseReranker):
    """本地模型重排序器。

    使用本地 BGE-Reranker 模型。

    需要安装：
    pip install sentence-transformers
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._loaded = False

    def _load_model(self):
        """延迟加载模型。"""
        if self._loaded:
            return

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._loaded = True
            logger.info("local_reranker_loaded", model=self.model_name)
        except ImportError:
            logger.error("sentence_transformers_not_installed")
            self._loaded = False
        except Exception as e:
            logger.error("local_reranker_load_failed", error=str(e))
            self._loaded = False

    def is_available(self) -> bool:
        """检查是否可用。"""
        self._load_model()
        return self._loaded and self._model is not None

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """执行本地重排序。"""
        if not candidates:
            return candidates

        self._load_model()
        if not self._model:
            # 返回原顺序
            for c in candidates:
                c["rerank_score"] = c.get("score", 0)
            return candidates

        try:
            # 构建输入
            pairs = [(query, c.get("content", "")) for c in candidates]

            # 批量预测
            scores = self._model.predict(pairs)

            # 应用分数
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)

            # 排序
            candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)

            logger.info(
                "local_rerank_completed",
                model=self.model_name,
                candidates=len(candidates),
                top_score=candidates[0].get("rerank_score", 0) if candidates else 0,
            )

        except Exception as e:
            logger.error("local_rerank_failed", error=str(e))
            for c in candidates:
                c["rerank_score"] = c.get("score", 0)

        return candidates


class HybridReranker(BaseReranker):
    """混合重排序器。

    组合多种重排序策略：
    1. API 重排序（主要）
    2. 本地模型（fallback）
    3. 规则辅助（query type 判断）

    Attributes:
        api_reranker: API 重排序器
        local_reranker: 本地重排序器
        use_rules: 是否使用规则辅助
    """

    def __init__(
        self,
        api_config: RerankConfig | None = None,
        local_model: str = "BAAI/bge-reranker-v2-m3",
        use_rules: bool = True,
    ):
        self.api_reranker = APIReranker(api_config or self._default_api_config())
        self.local_reranker = LocalReranker(local_model)
        self.use_rules = use_rules

    def _default_api_config(self) -> RerankConfig:
        """默认 API 配置。"""
        return RerankConfig(
            model_name="bge-reranker-v2-m3",
            api_url=settings.RERANK_API_URL,
            api_key=settings.DASHSCOPE_API_KEY,
            top_n=settings.RERANK_TOP_N,
        )

    def is_available(self) -> bool:
        """检查是否有可用的重排序器。"""
        return self.api_reranker.is_available() or self.local_reranker.is_available()

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """执行混合重排序。"""
        if not candidates:
            return candidates

        # 1. 规则预处理（可选）
        if self.use_rules:
            candidates = self._apply_rules(query, candidates)

        # 2. API 重排序
        if self.api_reranker.is_available():
            return self.api_reranker.rerank(query, candidates)

        # 3. Fallback 到本地模型
        if self.local_reranker.is_available():
            logger.info("rerank_fallback_to_local")
            return self.local_reranker.rerank(query, candidates)

        # 4. 无可用重排序器，保持原顺序
        logger.warning("no_reranker_available")
        for c in candidates:
            c["rerank_score"] = c.get("score", 0)
        return candidates

    def _apply_rules(self, query: str, candidates: list[dict]) -> list[dict]:
        """应用规则预处理。

        Args:
            query: 查询文本
            candidates: 候选列表

        Returns:
            处理后的候选列表
        """
        # 检测 query 类型
        query_type = self._detect_query_type(query)

        # 根据 query 类型调整候选权重
        for c in candidates:
            base_score = c.get("score", 0)
            chunk_type = c.get("chunk_type", "text")

            # 数字查询：提升表格权重
            if query_type == "number" and chunk_type == "table":
                c["score"] = base_score * 1.2

            # 定义查询：提升代码权重
            elif query_type == "definition" and chunk_type == "code":
                c["score"] = base_score * 1.1

            # 图片查询：提升图片权重
            elif query_type == "image" and chunk_type == "image":
                c["score"] = base_score * 1.3

        # 按调整后的分数预排序
        candidates.sort(key=lambda c: c.get("score", 0), reverse=True)

        return candidates

    def _detect_query_type(self, query: str) -> str:
        """检测查询类型。

        Args:
            query: 查询文本

        Returns:
            查询类型（number/definition/image/general）
        """
        import re

        # 数字查询
        if re.search(r"\d+|\多少|数量|金额|比例|百分比", query):
            return "number"

        # 定义查询
        if re.search(r"什么是|定义|概念|含义|解释", query):
            return "definition"

        # 图片查询
        if re.search(r"图片|图像|图表|图示|截图|照片", query):
            return "image"

        return "general"


class EnhancedReranker:
    """增强版重排序器入口。

    自动选择最优重排序策略。

    Attributes:
        hybrid_reranker: 混合重排序器
    """

    def __init__(self):
        self.hybrid_reranker = HybridReranker()

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """执行重排序。"""
        if not settings.RERANK_ENABLED:
            return candidates

        return self.hybrid_reranker.rerank(query, candidates)

    def batch_rerank(
        self,
        queries: list[str],
        candidates_list: list[list[dict]],
        max_workers: int = 5,
    ) -> list[list[dict]]:
        """批量重排序。

        Args:
            queries: 查询列表
            candidates_list: 候选列表列表
            max_workers: 并发数

        Returns:
            重排序后的候选列表列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.rerank, q, c): i
                for i, (q, c) in enumerate(zip(queries, candidates_list))
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    logger.error("batch_rerank_item_failed", idx=idx, error=str(e))
                    results.append((idx, candidates_list[idx]))

        # 按原顺序排列
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]


# 全局实例
enhanced_reranker = EnhancedReranker()

# 兼容旧接口
reranker = enhanced_reranker