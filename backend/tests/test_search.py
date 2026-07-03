"""
HybridSearcher 单元测试。

测试搜索过滤表达式构建、MySQL数据填充、边界情况等。
"""
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["TMPDIR"] = "/tmp"


class TestBuildExpr:
    """测试 _build_expr 过滤表达式构建。"""

    def _get_searcher(self):
        """Get a HybridSearcher instance without needing Milvus."""
        from app.search.hybrid_search import HybridSearcher
        return HybridSearcher()

    def test_empty_filters(self):
        """No filters should return empty string."""
        searcher = self._get_searcher()
        assert searcher._build_expr(None, None) == ""

    def test_single_doc_id(self):
        """Single doc_id should produce equality filter."""
        searcher = self._get_searcher()
        expr = searcher._build_expr(None, ["doc_123"])
        assert expr == 'doc_id == "doc_123"'

    def test_multiple_doc_ids(self):
        """Multiple doc_ids should produce IN filter."""
        searcher = self._get_searcher()
        expr = searcher._build_expr(None, ["doc_1", "doc_2", "doc_3"])
        assert 'doc_id in ["doc_1", "doc_2", "doc_3"]' == expr

    def test_string_filter(self):
        """String filter should produce equality on metadata."""
        searcher = self._get_searcher()
        expr = searcher._build_expr({"chunk_type": "text"}, None)
        assert expr == 'metadata["chunk_type"] == "text"'

    def test_list_filter(self):
        """List filter should produce IN on metadata."""
        searcher = self._get_searcher()
        expr = searcher._build_expr({"doc_type": ["pdf", "md"]}, None)
        assert expr == 'metadata["doc_type"] in ["pdf", "md"]'

    def test_numeric_filter(self):
        """Numeric filter should produce equality without quotes."""
        searcher = self._get_searcher()
        expr = searcher._build_expr({"page_num": 1}, None)
        assert expr == 'metadata["page_num"] == 1'

    def test_combined_doc_id_and_filter(self):
        """Combined doc_ids and filters should be joined with 'and'."""
        searcher = self._get_searcher()
        expr = searcher._build_expr({"chunk_type": "text"}, ["doc_1"])
        assert 'doc_id == "doc_1"' in expr
        assert 'metadata["chunk_type"] == "text"' in expr
        assert " and " in expr

    def test_multiple_filters(self):
        """Multiple filters should all be joined."""
        searcher = self._get_searcher()
        expr = searcher._build_expr(
            {"chunk_type": "text", "doc_type": "markdown"}, None
        )
        assert 'metadata["chunk_type"] == "text"' in expr
        assert 'metadata["doc_type"] == "markdown"' in expr


class TestEnrichWithMySQL:
    """测试 _enrich_with_mysql MySQL数据填充。"""

    def _get_searcher(self):
        from app.search.hybrid_search import HybridSearcher
        return HybridSearcher()

    def test_empty_results(self):
        """Empty results should return empty dict."""
        searcher = self._get_searcher()
        result = searcher._enrich_with_mysql([])
        assert result == {}

    def test_enrich_with_mysql_data(self):
        """Results should be enriched with MySQL chunk data."""
        searcher = self._get_searcher()

        # Mock Milvus results
        milvus_results = [
            {"chunk_id": "c1", "doc_id": "doc_1", "score": 0.9, "modal": "text"},
            {"chunk_id": "c2", "doc_id": "doc_1", "score": 0.8, "modal": "text"},
        ]

        # Mock MySQL data
        mock_chunk_c1 = MagicMock()
        mock_chunk_c1.chunk_id = "c1"
        mock_chunk_c1.doc_id = "doc_1"
        mock_chunk_c1.chunk_type = "text"
        mock_chunk_c1.content = "Test content 1"
        mock_chunk_c1.caption = None
        mock_chunk_c1.page_num = 1
        mock_chunk_c1.meta_data = {"doc_type": "markdown"}
        mock_chunk_c1.status = 1
        mock_chunk_c1.audit_status = 0
        mock_chunk_c1.creator = "test"
        mock_chunk_c1.sort_order = 0
        mock_chunk_c1.heading_level = 0
        mock_chunk_c1.heading_text = ""

        mock_chunk_c2 = MagicMock()
        mock_chunk_c2.chunk_id = "c2"
        mock_chunk_c2.doc_id = "doc_1"
        mock_chunk_c2.chunk_type = "text"
        mock_chunk_c2.content = "Test content 2"
        mock_chunk_c2.caption = None
        mock_chunk_c2.page_num = 1
        mock_chunk_c2.meta_data = {"doc_type": "markdown"}
        mock_chunk_c2.status = 1
        mock_chunk_c2.audit_status = 0
        mock_chunk_c2.creator = "test"
        mock_chunk_c2.sort_order = 1
        mock_chunk_c2.heading_level = 0
        mock_chunk_c2.heading_text = ""

        with patch("app.search.hybrid_search.get_chunks_by_doc") as mock_get:
            mock_get.return_value = (2, [mock_chunk_c1, mock_chunk_c2])
            result = searcher._enrich_with_mysql(milvus_results)

        assert "c1" in result
        assert "c2" in result
        assert result["c1"]["score"] == 0.9
        assert result["c1"]["content"] == "Test content 1"
        assert result["c1"]["modal"] == "text"
        assert result["c2"]["score"] == 0.8

    def test_enrich_missing_mysql_chunks(self):
        """Chunks not in MySQL should be excluded."""
        searcher = self._get_searcher()

        milvus_results = [
            {"chunk_id": "c1", "doc_id": "doc_1", "score": 0.9, "modal": "text"},
            {"chunk_id": "c99", "doc_id": "doc_1", "score": 0.5, "modal": "text"},
        ]

        mock_chunk_c1 = MagicMock()
        mock_chunk_c1.chunk_id = "c1"
        mock_chunk_c1.doc_id = "doc_1"
        mock_chunk_c1.chunk_type = "text"
        mock_chunk_c1.content = "Content 1"
        mock_chunk_c1.caption = None
        mock_chunk_c1.page_num = 1
        mock_chunk_c1.meta_data = {}
        mock_chunk_c1.status = 1
        mock_chunk_c1.audit_status = 0
        mock_chunk_c1.creator = ""
        mock_chunk_c1.sort_order = 0
        mock_chunk_c1.heading_level = 0
        mock_chunk_c1.heading_text = ""

        with patch("app.search.hybrid_search.get_chunks_by_doc") as mock_get:
            mock_get.return_value = (1, [mock_chunk_c1])  # Only c1 exists
            result = searcher._enrich_with_mysql(milvus_results)

        assert "c1" in result
        assert "c99" not in result


class TestSearchEdgeCases:
    """测试边界情况。"""

    def test_no_doc_ids_no_filters(self):
        """Search without doc_ids or filters should have empty expression."""
        from app.search.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        expr = searcher._build_expr(None, None)
        assert expr == ""

    def test_empty_doc_ids_list(self):
        """Empty doc_ids list should be treated as no filter."""
        from app.search.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        expr = searcher._build_expr(None, [])
        assert expr == ""

    def test_empty_filters_dict(self):
        """Empty filters dict should be treated as no filter."""
        from app.search.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        expr = searcher._build_expr({}, None)
        assert expr == ""
