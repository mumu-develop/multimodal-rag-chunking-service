"""
Milvus 客户端单元测试。

测试集合创建、schema验证、batch_upsert、search的round-trip等。
"""
import os
import random
import sys
from unittest.mock import MagicMock

import pytest

os.environ["TMPDIR"] = "/tmp"


def _make_mock_vector(dim=2048):
    """Generate a normalized mock vector."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = (sum(x*x for x in v)) ** 0.5
    return [x / norm for x in v]


class MockDashScopeResponse:
    """Mock DashScope API response with deterministic seeded vectors."""
    status_code = 200
    def __init__(self, **kwargs):
        self.output = MagicMock()
        self.output.embedding = _make_mock_vector()


def _mock_dashscope_call(api_key=None, model=None, input=None):
    """Mock dashscope.MultiModalEmbedding.call()."""
    resp = MagicMock()
    resp.status_code = 200
    resp.output = MagicMock()
    resp.output.embedding = _make_mock_vector()
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


class TestMilvusCollection:
    """测试集合创建和 schema 验证。"""

    def test_connect_creates_collection(self, milvus_client_isolated):
        """connect() should create the collection if it doesn't exist."""
        client = milvus_client_isolated
        client._get_client()
        name = milvus_client_isolated._collection_name
        assert name in client._get_client().list_collections()

    def test_collection_reused_on_second_connect(self, milvus_client_isolated):
        """Second connect() should reuse existing collection without error."""
        milvus_client_isolated.connect()
        milvus_client_isolated.connect()

    def test_drop_collection(self, milvus_client_isolated):
        """drop_collection should remove the collection."""
        client = milvus_client_isolated
        client.connect()
        client.drop_collection()
        assert client._collection_name not in client._get_client().list_collections()


class TestBatchUpsert:
    """测试批量 upsert 操作。"""

    def test_upsert_empty_list(self, milvus_client_isolated):
        """Empty upsert should return True without error."""
        assert milvus_client_isolated.batch_upsert([]) is True

    def test_upsert_single_chunk(self, milvus_client_isolated):
        """Single chunk upsert should succeed."""
        from app.embedders.unified_embedder import unified_embedder

        vector = unified_embedder.encode_query("test content")
        entities = [
            {
                "chunk_id": "test_chunk_001",
                "doc_id": "test_doc_001",
                "chunk_type": "text",
                "content": "test content",
                "image_path": "",
                "metadata": {"doc_type": "markdown"},
                "vector": vector,
            }
        ]
        assert milvus_client_isolated.batch_upsert(entities) is True

    def test_upsert_multiple_chunks(self, milvus_client_isolated):
        """Multiple chunks upsert should succeed."""
        from app.embedders.unified_embedder import unified_embedder

        entities = []
        for i in range(5):
            vector = unified_embedder.encode_query(f"content {i}")
            entities.append({
                "chunk_id": f"upsert_chunk_{i:03d}",
                "doc_id": "upsert_doc_001",
                "chunk_type": "text",
                "content": f"test content number {i}",
                "image_path": "",
                "metadata": {"doc_type": "markdown", "index": i},
                "vector": vector,
            })

        assert milvus_client_isolated.batch_upsert(entities) is True

    def test_upsert_and_search_round_trip(self, milvus_client_isolated):
        """
        Round-trip test: upsert a chunk, then search for it.
        """
        from app.embedders.unified_embedder import unified_embedder

        content = "round trip test content for search"
        vector = unified_embedder.encode_query(content)
        chunk_id = "round_trip_001"

        entities = [
            {
                "chunk_id": chunk_id,
                "doc_id": "round_trip_doc",
                "chunk_type": "text",
                "content": content,
                "image_path": "",
                "metadata": {"doc_type": "markdown"},
                "vector": vector,
            }
        ]
        assert milvus_client_isolated.batch_upsert(entities) is True

        query_vector = unified_embedder.encode_query("round trip test")
        results = milvus_client_isolated.hybrid_search(
            dense_vector=query_vector,
            sparse_vector={"query_text": "round trip test"},
            top_k=5,
            expr='doc_id == "round_trip_doc"',
            alpha=0.5,
        )

        chunk_ids = [r["chunk_id"] for r in results]
        assert chunk_id in chunk_ids, (
            f"Upserted chunk {chunk_id} not found in search results: {chunk_ids}"
        )


class TestHybridSearch:
    """测试混合搜索操作。"""

    def test_search_empty_collection(self, milvus_client_isolated):
        """Search on empty collection should return empty results."""
        from app.embedders.unified_embedder import unified_embedder

        vector = unified_embedder.encode_query("anything")
        results = milvus_client_isolated.hybrid_search(
            dense_vector=vector,
            sparse_vector={"query_text": "anything"},
            top_k=5,
            alpha=0.5,
        )
        assert results == []

    def test_search_with_output_fields(self, milvus_client_isolated):
        """Search with custom output_fields should return specified fields."""
        from app.embedders.unified_embedder import unified_embedder

        content = "output fields test content"
        vector = unified_embedder.encode_query(content)
        entities = [
            {
                "chunk_id": "output_test_001",
                "doc_id": "output_doc",
                "chunk_type": "text",
                "content": content,
                "image_path": "",
                "metadata": {"key": "value"},
                "vector": vector,
            }
        ]
        milvus_client_isolated.batch_upsert(entities)

        query_vector = unified_embedder.encode_query("output fields")
        results = milvus_client_isolated.hybrid_search(
            dense_vector=query_vector,
            sparse_vector={"query_text": "output fields"},
            top_k=1,
            expr='doc_id == "output_doc"',
            alpha=0.5,
            output_fields=["chunk_id", "content"],
        )

        assert len(results) == 1
        assert results[0]["chunk_id"] == "output_test_001"

    def test_search_with_empty_expr(self, milvus_client_isolated):
        """Search with empty expression should not raise."""
        from app.embedders.unified_embedder import unified_embedder

        vector = unified_embedder.encode_query("no filter")
        results = milvus_client_isolated.hybrid_search(
            dense_vector=vector,
            sparse_vector=None,
            top_k=5,
            expr="",
            alpha=0.5,
        )
        assert isinstance(results, list)

    def test_search_dense_only(self, milvus_client_isolated):
        """Search with only dense vector should work."""
        from app.embedders.unified_embedder import unified_embedder

        content = "dense only test content"
        vector = unified_embedder.encode_query(content)
        entities = [
            {
                "chunk_id": "dense_only_001",
                "doc_id": "dense_doc",
                "chunk_type": "text",
                "content": content,
                "image_path": "",
                "metadata": {},
                "vector": vector,
            }
        ]
        milvus_client_isolated.batch_upsert(entities)

        query_vector = unified_embedder.encode_query("dense only")
        results = milvus_client_isolated.hybrid_search(
            dense_vector=query_vector,
            sparse_vector=None,
            top_k=5,
            expr='doc_id == "dense_doc"',
            alpha=0.5,
        )

        chunk_ids = [r["chunk_id"] for r in results]
        assert "dense_only_001" in chunk_ids


class TestMilvusClientSchema:
    """测试 MilvusClient schema 验证逻辑。"""

    def test_schema_matches_returns_true_for_correct_collection(self, milvus_client_isolated):
        """_schema_matches should return True for a correctly created collection."""
        client = milvus_client_isolated._get_client()
        name = milvus_client_isolated.connect()
        assert milvus_client_isolated._schema_matches(client, name) is True

    def test_schema_matches_returns_false_for_nonexistent(self, milvus_client_isolated):
        """_schema_matches should return False for a non-existent collection."""
        client = milvus_client_isolated._get_client()
        assert milvus_client_isolated._schema_matches(client, "nonexistent_collection") is False
