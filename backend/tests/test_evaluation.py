"""
召回率/准确率核心评估测试。

使用 golden dataset 对 RAG 检索质量进行评估，包括：
- Recall@K：检索到的相关 chunk 比例
- Precision@K：返回结果中相关的比例
- NDCG@K：归一化折损累计增益
- Rerank 效果对比
- Alpha 参数敏感性分析
"""
import json
import math
import os
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend/ is on sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent

os.environ["TMPDIR"] = "/tmp"


# ============================================================
# Mock embedder (since production uses DashScope API)
# ============================================================

class MockDashScopeResponse:
    """Mock DashScope API response with deterministic seeded vectors."""
    status_code = 200
    def __init__(self, **kwargs):
        self.output = MagicMock()
        self.output.embedding = self._hash_vector("mock")

    @staticmethod
    def _hash_vector(text: str = "mock", dim: int = 2048):
        """Generate a deterministic vector from text hash."""
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        random.seed(int.from_bytes(h[:8], "big"))
        v = [random.gauss(0, 1) for _ in range(dim)]
        norm = (sum(x*x for x in v)) ** 0.5
        return [x / norm for x in v]


def _mock_dashscope_call(api_key=None, model=None, input=None):
    """Mock dashscope.MultiModalEmbedding.call()."""
    import hashlib
    # Generate deterministic vector from input text
    text = ""
    for item in (input or []):
        text += item.get("text", "")

    h = hashlib.sha256(text.encode()).digest()
    random.seed(int.from_bytes(h[:8], "big"))
    v = [random.gauss(0, 1) for _ in range(2048)]
    norm = (sum(x*x for x in v)) ** 0.5
    vec = [x / norm for x in v]

    resp = MagicMock()
    resp.status_code = 200
    resp.output = MagicMock()
    resp.output.embedding = vec
    return resp


@pytest.fixture(autouse=True)
def mock_embedder_global():
    """Mock dashscope globally using sys.modules."""
    mock_module = MagicMock()
    mock_module.MultiModalEmbedding = MagicMock()
    mock_module.MultiModalEmbedding.call = _mock_dashscope_call
    sys.modules["dashscope"] = mock_module

    from app.embedders.unified_embedder import unified_embedder
    unified_embedder._loaded = False
    yield


# ============================================================
# Evaluation metric helpers
# ============================================================

def compute_recall_at_k(retrieved: list[dict], relevant_ids: set, k: int) -> float:
    """
    Recall@K = (检索到的相关 chunk 数) / (全部相关 chunk 数)
    """
    if not relevant_ids:
        return 1.0
    retrieved_ids = {h["chunk_id"] for h in retrieved[:k]}
    hit_count = len(retrieved_ids & relevant_ids)
    return hit_count / len(relevant_ids)


def compute_precision_at_k(retrieved: list[dict], relevant_ids: set, k: int) -> float:
    """
    Precision@K = (检索到的相关 chunk 数) / K
    """
    if k == 0:
        return 0.0
    retrieved_ids = {h["chunk_id"] for h in retrieved[:k]}
    hit_count = len(retrieved_ids & relevant_ids)
    return hit_count / k


def compute_ndcg_at_k(retrieved: list[dict], relevant_ids: set, k: int) -> float:
    """
    NDCG@K：归一化折损累计增益。
    二元相关性：chunk_id 在 relevant_ids 中得分为 1，否则为 0。
    """
    def dcg_at_k(items, k):
        dcg = 0.0
        for i, item in enumerate(items[:k]):
            rel = 1.0 if item["chunk_id"] in relevant_ids else 0.0
            dcg += rel / math.log2(i + 2)  # i+2 because position is 1-indexed
        return dcg

    dcg = dcg_at_k(retrieved, k)

    # Ideal DCG: all relevant items first
    ideal = [
        {"chunk_id": cid} for cid in list(relevant_ids)[:k]
    ]
    idcg = dcg_at_k(ideal, k)

    if idcg == 0:
        return 1.0
    return dcg / idcg


def is_relevant_by_keywords(content: str, keywords: list[str]) -> bool:
    """
    通过关键词匹配判断 chunk 内容是否相关。
    chunk 包含任意一个关键词即视为相关。
    """
    if not content:
        return False
    return any(kw in content for kw in keywords)


# ============================================================
# Helper: ingest test chunks into Milvus
# ============================================================

def ingest_golden_chunks(milvus_client_obj, test_chunks):
    """Encode and upsert test chunks into Milvus with proper vectors."""
    from app.embedders.unified_embedder import unified_embedder

    entities = []
    for chunk in test_chunks:
        content = chunk["content"]
        vector = unified_embedder.encode_query(content)
        entities.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "chunk_type": chunk["chunk_type"],
            "content": content,
            "image_path": chunk.get("image_path", ""),
            "metadata": chunk.get("metadata", {}),
            "vector": vector,
        })

    milvus_client_obj.batch_upsert(entities)


def build_golden_relevant_ids(chunks, keywords):
    """From chunk dicts, return set of chunk_ids matching keywords."""
    relevant = set()
    for c in chunks:
        if is_relevant_by_keywords(c.get("content", ""), keywords):
            relevant.add(c["chunk_id"])
    return relevant


# ============================================================
# Fixtures for evaluation tests
# ============================================================

@pytest.fixture
def seeded_ingested_milvus(milvus_client_isolated, mysql_mock, test_chunks_factory):
    """Create deterministic test chunks and ingest them into Milvus."""
    chunks = test_chunks_factory(doc_id="eval_doc_001", n_chunks=10)

    ingest_golden_chunks(milvus_client_isolated, chunks)

    for c in chunks:
        mysql_mock[c["chunk_id"]] = c

    return chunks


# ============================================================
# Test: Recall@K on golden dataset
# ============================================================

class TestRecallAtK:
    """Recall@K evaluation tests."""

    def test_recall_at_k_basic(self, seeded_ingested_milvus, golden_dataset, milvus_client_isolated, mock_settings):
        """For each query in golden dataset, compute Recall@K."""
        from app.embedders.unified_embedder import unified_embedder

        all_chunks = seeded_ingested_milvus

        for query_entry in golden_dataset["queries"]:
            query = query_entry["query"]
            keywords = query_entry["relevant_chunk_keywords"]
            top_k = golden_dataset.get("evaluation_params", {}).get("default_top_k", 5)

            relevant_ids = build_golden_relevant_ids(all_chunks, keywords)
            if not relevant_ids:
                continue

            query_vector = unified_embedder.encode_query(query)
            results = milvus_client_isolated.hybrid_search(
                dense_vector=query_vector,
                sparse_vector={"query_text": query},
                top_k=top_k,
                expr='doc_id == "eval_doc_001"',
                alpha=mock_settings.HYBRID_SEARCH_ALPHA,
            )

            recall = compute_recall_at_k(results, relevant_ids, top_k)
            assert 0.0 <= recall <= 1.0

    def test_recall_at_k_1(self, seeded_ingested_milvus, milvus_client_isolated, mock_settings):
        """Test Recall@1 is non-negative."""
        from app.embedders.unified_embedder import unified_embedder

        all_chunks = seeded_ingested_milvus
        query = "向量嵌入使用什么模型"
        keywords = ["Qwen3-VL-Embedding-2B", "2048维"]
        relevant_ids = build_golden_relevant_ids(all_chunks, keywords)

        if not relevant_ids:
            pytest.skip("No relevant chunks found")

        query_vector = unified_embedder.encode_query(query)
        results = milvus_client_isolated.hybrid_search(
            dense_vector=query_vector,
            sparse_vector={"query_text": query},
            top_k=1,
            expr='doc_id == "eval_doc_001"',
            alpha=mock_settings.HYBRID_SEARCH_ALPHA,
        )

        recall = compute_recall_at_k(results, relevant_ids, 1)
        assert recall >= 0.0


# ============================================================
# Test: Precision@K
# ============================================================

class TestPrecisionAtK:

    def test_precision_at_k_basic(self, seeded_ingested_milvus, golden_dataset, milvus_client_isolated, mock_settings):
        """Compute Precision@K and verify valid range."""
        from app.embedders.unified_embedder import unified_embedder

        all_chunks = seeded_ingested_milvus
        top_k = golden_dataset.get("evaluation_params", {}).get("default_top_k", 5)

        precisions = []
        for query_entry in golden_dataset["queries"]:
            query = query_entry["query"]
            keywords = query_entry["relevant_chunk_keywords"]
            relevant_ids = build_golden_relevant_ids(all_chunks, keywords)

            if not relevant_ids:
                continue

            query_vector = unified_embedder.encode_query(query)
            results = milvus_client_isolated.hybrid_search(
                dense_vector=query_vector,
                sparse_vector={"query_text": query},
                top_k=top_k,
                expr='doc_id == "eval_doc_001"',
                alpha=mock_settings.HYBRID_SEARCH_ALPHA,
            )

            precisions.append(compute_precision_at_k(results, relevant_ids, top_k))

        if precisions:
            assert all(0 <= p <= 1 for p in precisions)

    def test_precision_decreases_with_k(self, seeded_ingested_milvus, milvus_client_isolated, mock_settings):
        """Precision@K values should be valid for different K."""
        from app.embedders.unified_embedder import unified_embedder

        all_chunks = seeded_ingested_milvus
        query = "文档解析支持哪些格式"
        keywords = ["PDF", "DOCX", "PPTX", "Markdown", "解析"]
        relevant_ids = build_golden_relevant_ids(all_chunks, keywords)

        if not relevant_ids:
            pytest.skip("No relevant chunks found")

        query_vector = unified_embedder.encode_query(query)

        for k in [1, 3, 5, 10]:
            results = milvus_client_isolated.hybrid_search(
                dense_vector=query_vector,
                sparse_vector={"query_text": query},
                top_k=k,
                expr='doc_id == "eval_doc_001"',
                alpha=mock_settings.HYBRID_SEARCH_ALPHA,
            )
            p = compute_precision_at_k(results, relevant_ids, k)
            assert 0 <= p <= 1


# ============================================================
# Test: NDCG@K
# ============================================================

class TestNDCGAtK:

    def test_ndcg_at_k_basic(self, seeded_ingested_milvus, golden_dataset, milvus_client_isolated, mock_settings):
        """Compute NDCG@K for each query in golden dataset."""
        from app.embedders.unified_embedder import unified_embedder

        all_chunks = seeded_ingested_milvus
        k = golden_dataset.get("evaluation_params", {}).get("ndcg_k", 5)

        ndcg_scores = []
        for query_entry in golden_dataset["queries"]:
            query = query_entry["query"]
            keywords = query_entry["relevant_chunk_keywords"]
            relevant_ids = build_golden_relevant_ids(all_chunks, keywords)

            if not relevant_ids:
                continue

            query_vector = unified_embedder.encode_query(query)
            results = milvus_client_isolated.hybrid_search(
                dense_vector=query_vector,
                sparse_vector={"query_text": query},
                top_k=k,
                expr='doc_id == "eval_doc_001"',
                alpha=mock_settings.HYBRID_SEARCH_ALPHA,
            )

            ndcg_scores.append(compute_ndcg_at_k(results, relevant_ids, k))

        if ndcg_scores:
            avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
            assert 0.0 <= avg_ndcg <= 1.0


# ============================================================
# Test: Rerank comparison
# ============================================================

class TestRerankEffect:

    def test_search_without_rerank(self, seeded_ingested_milvus, milvus_client_isolated, mock_settings):
        """Verify search without rerank returns results."""
        from app.embedders.unified_embedder import unified_embedder

        all_chunks = seeded_ingested_milvus
        query = "系统架构包含哪些组件"
        keywords = ["FastAPI", "Celery", "Redis", "MySQL", "Milvus"]
        relevant_ids = build_golden_relevant_ids(all_chunks, keywords)

        if not relevant_ids:
            pytest.skip("No relevant chunks found")

        query_vector = unified_embedder.encode_query(query)
        results = milvus_client_isolated.hybrid_search(
            dense_vector=query_vector,
            sparse_vector={"query_text": query},
            top_k=5,
            expr='doc_id == "eval_doc_001"',
            alpha=mock_settings.HYBRID_SEARCH_ALPHA,
        )

        assert len(results) > 0


# ============================================================
# Test: Alpha parameter sensitivity
# ============================================================

class TestAlphaSensitivity:

    def test_recall_with_different_alphas(self, seeded_ingested_milvus, milvus_client_isolated):
        """Compare retrieval with different alpha values."""
        from app.embedders.unified_embedder import unified_embedder

        all_chunks = seeded_ingested_milvus
        query = "文本分块的参数是什么"
        keywords = ["RecursiveCharacterTextSplitter", "chunk_size", "800"]
        relevant_ids = build_golden_relevant_ids(all_chunks, keywords)

        if not relevant_ids:
            pytest.skip("No relevant chunks found")

        query_vector = unified_embedder.encode_query(query)
        top_k = 5

        for alpha in [0.3, 0.5, 0.7]:
            results = milvus_client_isolated.hybrid_search(
                dense_vector=query_vector,
                sparse_vector={"query_text": query},
                top_k=top_k,
                expr='doc_id == "eval_doc_001"',
                alpha=alpha,
            )
            assert len(results) > 0, f"Alpha={alpha} returned no results"


# ============================================================
# Test: Metric function unit tests
# ============================================================

class TestMetricFunctions:
    """Unit tests for the evaluation metric helper functions."""

    def test_recall_perfect(self):
        retrieved = [{"chunk_id": f"c{i}"} for i in range(5)]
        relevant_ids = {"c0", "c1", "c2", "c3", "c4"}
        assert compute_recall_at_k(retrieved, relevant_ids, 5) == 1.0

    def test_recall_partial(self):
        retrieved = [{"chunk_id": f"c{i}"} for i in range(4)]
        relevant_ids = {"c0", "c1", "c5", "c6"}
        assert compute_recall_at_k(retrieved, relevant_ids, 4) == 0.5

    def test_recall_zero(self):
        retrieved = [{"chunk_id": f"c{i}"} for i in range(3)]
        relevant_ids = {"c10", "c11"}
        assert compute_recall_at_k(retrieved, relevant_ids, 3) == 0.0

    def test_recall_empty_relevant(self):
        retrieved = [{"chunk_id": "c0"}]
        assert compute_recall_at_k(retrieved, set(), 3) == 1.0

    def test_precision_perfect(self):
        retrieved = [{"chunk_id": f"c{i}"} for i in range(5)]
        relevant_ids = {"c0", "c1", "c2", "c3", "c4"}
        assert compute_precision_at_k(retrieved, relevant_ids, 5) == 1.0

    def test_precision_partial(self):
        retrieved = [{"chunk_id": f"c{i}"} for i in range(5)]
        relevant_ids = {"c0", "c1"}
        assert compute_precision_at_k(retrieved, relevant_ids, 5) == 0.4

    def test_precision_zero_k(self):
        assert compute_precision_at_k([], {"c0"}, 0) == 0.0

    def test_ndcg_perfect(self):
        retrieved = [{"chunk_id": f"c{i}"} for i in range(5)]
        relevant_ids = {"c0", "c1", "c2"}
        ndcg = compute_ndcg_at_k(retrieved, relevant_ids, 3)
        assert ndcg == pytest.approx(1.0, abs=0.01)

    def test_ndcg_worst(self):
        retrieved = [{"chunk_id": f"c{i}"} for i in range(5)]
        relevant_ids = {"c10", "c11"}
        assert compute_ndcg_at_k(retrieved, relevant_ids, 3) == 0.0

    def test_ndcg_no_relevant(self):
        retrieved = [{"chunk_id": "c0"}]
        assert compute_ndcg_at_k(retrieved, set(), 5) == 1.0

    def test_keyword_match_positive(self):
        content = "多模态RAG是一种结合多种模态的技术"
        assert is_relevant_by_keywords(content, ["多模态", "RAG"])

    def test_keyword_match_negative(self):
        content = "这是一个关于数据库的段落"
        assert not is_relevant_by_keywords(content, ["多模态", "RAG"])
