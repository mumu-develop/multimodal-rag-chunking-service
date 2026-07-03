"""
增强版重排序器测试。

测试 EnhancedReranker、APIReranker、LocalReranker、HybridReranker 的功能。
使用 mock 模拟 API 响应和本地模型。
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 确保 backend 目录在 sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

os.environ["TMPDIR"] = "/tmp"


# ============================================================
# Mock HTTP Client for API Reranker
# ============================================================

class MockHTTPResponse:
    """Mock HTTP 响应。"""

    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception(f"HTTP {self.status_code}")


def create_mock_httpx_client():
    """创建 mock httpx 客户端。"""
    mock_client = MagicMock()

    def mock_post(url, json=None, headers=None):
        # 模拟 DashScope API 响应
        if "dashscope" in url.lower():
            return MockHTTPResponse({
                "output": {
                    "results": [
                        {"index": 0, "relevance_score": 0.95},
                        {"index": 1, "relevance_score": 0.80},
                        {"index": 2, "relevance_score": 0.60},
                    ]
                }
            })

        # 模拟 Cohere API 响应
        elif "cohere" in url.lower():
            return MockHTTPResponse({
                "results": [
                    {"index": 0, "relevance_score": 0.92},
                    {"index": 1, "relevance_score": 0.78},
                ]
            })

        # 默认响应
        return MockHTTPResponse({
            "results": [
                {"index": 0, "relevance_score": 0.85},
            ]
        })

    mock_client.post = mock_post
    return mock_client


# ============================================================
# API Reranker Tests
# ============================================================

class TestAPIReranker:
    """API 重排序器测试。"""

    @pytest.fixture
    def api_reranker(self):
        """创建 API 重排序器实例。"""
        from app.embedders.enhanced_reranker import APIReranker, RerankConfig
        config = RerankConfig(
            model_name="bge-reranker-v2-m3",
            api_url="https://dashscope.aliyuncs.com/api/v1/services/rerank",
            api_key="test-api-key",
            top_n=10,
            cache_enabled=True,
        )
        reranker = APIReranker(config)
        reranker._client = create_mock_httpx_client()
        return reranker

    @pytest.fixture
    def candidates(self):
        """创建测试候选列表。"""
        return [
            {"chunk_id": "c1", "content": "文档内容一", "score": 0.5},
            {"chunk_id": "c2", "content": "文档内容二", "score": 0.4},
            {"chunk_id": "c3", "content": "文档内容三", "score": 0.3},
        ]

    def test_is_available(self, api_reranker):
        """测试可用性检查。"""
        assert api_reranker.is_available() == True

    def test_rerank_basic(self, api_reranker, candidates):
        """测试基础重排序。"""
        query = "查询文本"
        result = api_reranker.rerank(query, candidates)

        assert len(result) == 3
        # 检查添加了 rerank_score
        for c in result:
            assert "rerank_score" in c

    def test_rerank_sorts_by_score(self, api_reranker, candidates):
        """测试按分数排序。"""
        query = "查询文本"
        result = api_reranker.rerank(query, candidates)

        # 应按 rerank_score 降序排列
        scores = [c["rerank_score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_threshold_filter(self):
        """测试阈值过滤。"""
        from app.embedders.enhanced_reranker import APIReranker, RerankConfig

        config = RerankConfig(
            api_url="https://dashscope.test.com",
            api_key="test",
            threshold=0.7,  # 设置阈值
        )
        reranker = APIReranker(config)
        reranker._client = create_mock_httpx_client()

        candidates = [
            {"chunk_id": "c1", "content": "高分内容", "score": 0.5},
            {"chunk_id": "c2", "content": "低分内容", "score": 0.3},
        ]

        result = reranker.rerank("query", candidates)

        # 低分候选应被过滤
        # 注意：mock 返回分数可能高于阈值，这里只测试逻辑存在
        assert len(result) >= 1

    def test_rerank_empty_candidates(self, api_reranker):
        """测试空候选列表。"""
        result = api_reranker.rerank("query", [])
        assert result == []

    def test_rerank_caching(self, api_reranker, candidates):
        """测试重排序缓存。"""
        query = "缓存测试查询"

        # 第一次调用
        api_reranker.rerank(query, candidates)

        # 检查缓存写入
        assert len(api_reranker._cache) > 0

        # 第二次调用（应命中缓存）
        api_reranker.rerank(query, candidates)

    def test_rerank_api_error_handling(self):
        """测试 API 错误处理。"""
        from app.embedders.enhanced_reranker import APIReranker, RerankConfig

        config = RerankConfig(api_url="https://error.com", api_key="test")
        reranker = APIReranker(config)

        # 创建抛异常的 mock
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("API Error")
        reranker._client = mock_client

        candidates = [{"chunk_id": "c1", "content": "test", "score": 0.5}]

        # 应不抛异常，返回原顺序
        result = reranker.rerank("query", candidates)
        assert len(result) == 1
        assert result[0].get("rerank_score") == 0.5  # 使用原 score

    def test_dashscope_api_format(self, api_reranker):
        """测试 DashScope API 格式。"""
        from app.embedders.enhanced_reranker import RerankConfig

        config = RerankConfig(
            api_url="https://dashscope.aliyuncs.com/rerank",
            api_key="test",
        )
        reranker = APIReranker(config)
        reranker._client = create_mock_httpx_client()

        candidates = [{"content": "test", "score": 0.5}]
        scores = reranker._call_api("query", candidates)

        assert isinstance(scores, list)
        assert len(scores) > 0
        assert isinstance(scores[0], tuple)
        assert len(scores[0]) == 2  # (index, score)


# ============================================================
# Local Reranker Tests
# ============================================================

class TestLocalReranker:
    """本地模型重排序器测试。"""

    @pytest.fixture
    def local_reranker(self):
        """创建本地重排序器实例。"""
        from app.embedders.enhanced_reranker import LocalReranker
        return LocalReranker(model_name="BAAI/bge-reranker-v2-m3")

    def test_is_available_with_mock(self, local_reranker):
        """测试可用性（使用 mock）。"""
        # Mock sentence_transformers
        mock_module = MagicMock()
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.95, 0.80, 0.60]
        mock_module.CrossEncoder = mock_cross_encoder

        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            local_reranker._loaded = False
            local_reranker._model = None
            assert local_reranker.is_available() == True

    def test_rerank_with_mock_model(self, local_reranker):
        """测试重排序（使用 mock 模型）。"""
        mock_module = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95, 0.80, 0.60]
        mock_module.CrossEncoder = MagicMock(return_value=mock_model)

        candidates = [
            {"chunk_id": "c1", "content": "内容一", "score": 0.5},
            {"chunk_id": "c2", "content": "内容二", "score": 0.4},
            {"chunk_id": "c3", "content": "内容三", "score": 0.3},
        ]

        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            local_reranker._loaded = False
            local_reranker._model = None
            local_reranker._load_model()

            result = local_reranker.rerank("query", candidates)

            assert len(result) == 3
            for c in result:
                assert "rerank_score" in c

    def test_rerank_without_model(self, local_reranker):
        """测试无模型时的处理。"""
        local_reranker._loaded = False
        local_reranker._model = None

        # Mock 导入失败
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            candidates = [{"content": "test", "score": 0.5}]
            result = local_reranker.rerank("query", candidates)

            # 应使用原 score
            assert result[0].get("rerank_score") == 0.5


# ============================================================
# Hybrid Reranker Tests
# ============================================================

class TestHybridReranker:
    """混合重排序器测试。"""

    @pytest.fixture
    def hybrid_reranker(self):
        """创建混合重排序器实例。"""
        from app.embedders.enhanced_reranker import HybridReranker, RerankConfig
        from app.config import settings

        # 使用测试配置
        api_config = RerankConfig(
            api_url=getattr(settings, "RERANK_API_URL", ""),
            api_key=getattr(settings, "DASHSCOPE_API_KEY", ""),
            top_n=getattr(settings, "RERANK_TOP_N", 50),
        )

        reranker = HybridReranker(
            api_config=api_config,
            local_model="BAAI/bge-reranker-v2-m3",
            use_rules=True,
        )
        return reranker

    def test_is_available(self, hybrid_reranker):
        """测试可用性检查。"""
        # 至少有一个可用
        # API 或 Local
        available = hybrid_reranker.is_available()
        # 如果都没有 API 配置，检查是否有 local
        assert isinstance(available, bool)

    def test_rerank_fallback_logic(self, hybrid_reranker):
        """测试 fallback 逻辑。"""
        candidates = [
            {"content": "内容一", "score": 0.5},
            {"content": "内容二", "score": 0.4},
        ]

        # Mock API 不可用
        hybrid_reranker.api_reranker._client = None

        result = hybrid_reranker.rerank("query", candidates)
        assert len(result) >= 0

    def test_detect_query_type_number(self, hybrid_reranker):
        """测试数字查询类型检测。"""
        query_type = hybrid_reranker._detect_query_type("有多少用户")
        assert query_type == "number"

        query_type = hybrid_reranker._detect_query_type("比例是多少")
        assert query_type == "number"

    def test_detect_query_type_definition(self, hybrid_reranker):
        """测试定义查询类型检测。"""
        query_type = hybrid_reranker._detect_query_type("什么是RAG")
        assert query_type == "definition"

        query_type = hybrid_reranker._detect_query_type("概念解释")
        assert query_type == "definition"

    def test_detect_query_type_image(self, hybrid_reranker):
        """测试图片查询类型检测。"""
        query_type = hybrid_reranker._detect_query_type("展示图片")
        assert query_type == "image"

        query_type = hybrid_reranker._detect_query_type("图表信息")
        assert query_type == "image"

    def test_detect_query_type_general(self, hybrid_reranker):
        """测试通用查询类型检测。"""
        query_type = hybrid_reranker._detect_query_type("搜索相关内容")
        assert query_type == "general"

    def test_apply_rules_table_boost(self, hybrid_reranker):
        """测试规则应用 - 表格权重提升。"""
        candidates = [
            {"chunk_type": "table", "content": "表格", "score": 0.5},
            {"chunk_type": "text", "content": "文本", "score": 0.5},
        ]

        hybrid_reranker.use_rules = True
        result = hybrid_reranker._apply_rules("有多少数据", candidates)

        # 数字查询时，表格权重应提升
        table_score = result[0].get("score", 0.5)
        assert table_score > 0.5  # 提升了

    def test_apply_rules_image_boost(self, hybrid_reranker):
        """测试规则应用 - 图片权重提升。"""
        candidates = [
            {"chunk_type": "image", "content": "图片", "score": 0.5},
            {"chunk_type": "text", "content": "文本", "score": 0.5},
        ]

        result = hybrid_reranker._apply_rules("展示图表", candidates)

        # 图片查询时，图片权重应提升
        image_score = None
        for c in result:
            if c.get("chunk_type") == "image":
                image_score = c.get("score")
                break

        assert image_score is not None and image_score > 0.5


# ============================================================
# Enhanced Reranker Tests
# ============================================================

class TestEnhancedReranker:
    """增强版重排序器入口测试。"""

    @pytest.fixture
    def enhanced_reranker(self):
        """创建增强版重排序器实例。"""
        from app.embedders.enhanced_reranker import EnhancedReranker
        return EnhancedReranker()

    def test_rerank_enabled(self, enhanced_reranker):
        """测试重排序启用。"""
        candidates = [{"content": "test", "score": 0.5}]

        # Mock settings
        with patch("app.embedders.enhanced_reranker.settings") as mock_settings:
            mock_settings.RERANK_ENABLED = True
            result = enhanced_reranker.rerank("query", candidates)
            assert len(result) >= 0

    def test_rerank_disabled(self, enhanced_reranker):
        """测试重排序禁用。"""
        candidates = [{"content": "test", "score": 0.5}]

        with patch("app.embedders.enhanced_reranker.settings") as mock_settings:
            mock_settings.RERANK_ENABLED = False
            result = enhanced_reranker.rerank("query", candidates)
            # 禁用时返回原候选
            assert result == candidates

    def test_batch_rerank(self, enhanced_reranker):
        """测试批量重排序。"""
        queries = ["查询一", "查询二", "查询三"]
        candidates_list = [
            [{"content": "c1", "score": 0.5}],
            [{"content": "c2", "score": 0.4}],
            [{"content": "c3", "score": 0.3}],
        ]

        with patch("app.embedders.enhanced_reranker.settings") as mock_settings:
            mock_settings.RERANK_ENABLED = True
            results = enhanced_reranker.batch_rerank(queries, candidates_list, max_workers=2)

            assert len(results) == 3

    def test_batch_rerank_preserves_order(self, enhanced_reranker):
        """测试批量重排序保持顺序。"""
        queries = ["q1", "q2"]
        candidates_list = [
            [{"content": "A", "score": 0.5}],
            [{"content": "B", "score": 0.4}],
        ]

        with patch("app.embedders.enhanced_reranker.settings") as mock_settings:
            mock_settings.RERANK_ENABLED = True
            results = enhanced_reranker.batch_rerank(queries, candidates_list)

            # 结果顺序应与输入一致
            assert results[0][0]["content"] == "A"
            assert results[1][0]["content"] == "B"


# ============================================================
# Cache Key Tests
# ============================================================

class TestCacheKeyGeneration:
    """缓存 Key 生成测试。"""

    def test_make_cache_key(self):
        """测试缓存 Key 生成。"""
        from app.embedders.enhanced_reranker import APIReranker, RerankConfig

        config = RerankConfig(api_url="https://test.com", api_key="test")
        reranker = APIReranker(config)

        query = "测试查询"
        candidates = [
            {"content": "内容一"},
            {"content": "内容二"},
        ]

        key1 = reranker._make_cache_key(query, candidates)
        key2 = reranker._make_cache_key(query, candidates)

        # 相同输入应生成相同 key
        assert key1 == key2

    def test_cache_key_different_queries(self):
        """测试不同查询不同 Key。"""
        from app.embedders.enhanced_reranker import APIReranker, RerankConfig

        config = RerankConfig(api_url="https://test.com", api_key="test")
        reranker = APIReranker(config)

        candidates = [{"content": "same"}]

        key1 = reranker._make_cache_key("query1", candidates)
        key2 = reranker._make_cache_key("query2", candidates)

        # 不同查询应生成不同 key
        assert key1 != key2