"""
Embedder 单元测试。

测试文本编码的向量维度、归一化、批量编码一致性等。
使用 mock 模拟 DashScope API 响应。
"""
import os
import random
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ["TMPDIR"] = "/tmp"


def _make_mock_vector(dim=2048):
    """Generate a normalized mock vector."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = (sum(x*x for x in v)) ** 0.5
    return [x / norm for x in v]


class MockDashScopeResponse:
    """Mock DashScope API response."""
    def __init__(self, input_list):
        self.status_code = 200
        self.output = MagicMock()
        self.output.embedding = _make_mock_vector()


@pytest.fixture(autouse=True)
def setup_mock():
    """Patch dashscope MultiModalEmbedding before any test runs."""
    # Create a mock module for dashscope
    mock_module = MagicMock()
    mock_multi_modal = MagicMock()
    mock_multi_modal.call = MockDashScopeResponse
    mock_module.MultiModalEmbedding = mock_multi_modal
    sys.modules["dashscope"] = mock_module

    # Reset embedder singleton
    from app.embedders.unified_embedder import unified_embedder
    unified_embedder._loaded = False

    yield

    # Cleanup
    unified_embedder._loaded = False


class TestUnifiedEmbedderMock:
    """测试 mock 模式下的嵌入器。"""

    def test_encode_query_dimension(self, setup_mock):
        """Encoded query vector should have correct dimension (2048)."""
        from app.embedders.unified_embedder import unified_embedder
        vector = unified_embedder.encode_query("test query")
        assert len(vector) == 2048

    def test_encode_texts_dimension(self, setup_mock):
        """Batch encoded text vectors should have correct dimensions."""
        from app.embedders.unified_embedder import unified_embedder
        vectors = unified_embedder.encode_texts(["text one", "text two", "text three"])
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 2048

    def test_encode_query_returns_single_vector(self, setup_mock):
        """encode_query should return a single vector, not a list of vectors."""
        from app.embedders.unified_embedder import unified_embedder
        vector = unified_embedder.encode_query("hello")
        assert isinstance(vector, list)
        assert isinstance(vector[0], float)

    def test_encode_texts_returns_list_of_vectors(self, setup_mock):
        """encode_texts should return a list of vectors."""
        from app.embedders.unified_embedder import unified_embedder
        vectors = unified_embedder.encode_texts(["hello", "world"])
        assert isinstance(vectors, list)
        assert isinstance(vectors[0], list)

    def test_empty_texts_returns_empty_list(self, setup_mock):
        """Empty text list should return empty list."""
        from app.embedders.unified_embedder import unified_embedder
        assert unified_embedder.encode_texts([]) == []

    def test_mock_vectors_are_normalized(self, setup_mock):
        """Mock vectors should be L2 normalized (norm ≈ 1.0)."""
        from app.embedders.unified_embedder import unified_embedder
        vector = unified_embedder.encode_query("test")
        norm = (sum(x*x for x in vector)) ** 0.5
        assert abs(norm - 1.0) < 0.001

    def test_mock_vectors_vary_between_calls(self, setup_mock):
        """Vectors should vary between different inputs (random)."""
        from app.embedders.unified_embedder import unified_embedder
        v1 = unified_embedder.encode_query("query one")
        v2 = unified_embedder.encode_query("query two")
        assert v1 != v2


class TestEmbedderAPI:
    """测试嵌入器 API 接口。"""

    def test_encode_images_returns_vectors(self, setup_mock, tmp_path):
        """encode_images should return vectors for image paths."""
        from app.embedders.unified_embedder import unified_embedder
        img_path = str(tmp_path / "test.png")
        with open(img_path, "w") as f:
            f.write("fake image")
        vectors = unified_embedder.encode_images([img_path])
        assert len(vectors) == 1
        assert len(vectors[0]) == 2048

    def test_encode_empty_images_returns_empty(self, setup_mock):
        """Empty image list should return empty list."""
        from app.embedders.unified_embedder import unified_embedder
        assert unified_embedder.encode_images([]) == []

    def test_encode_image_query_returns_vector(self, setup_mock, tmp_path):
        """encode_image_query should return a single vector."""
        from app.embedders.unified_embedder import unified_embedder
        img_path = str(tmp_path / "test.png")
        with open(img_path, "w") as f:
            f.write("fake image")
        vector = unified_embedder.encode_image_query(img_path)
        assert len(vector) == 2048
        assert isinstance(vector[0], float)


class TestConsistency:
    """测试批量编码与单条编码的一致性。"""

    def test_single_vs_batch_same_dimension(self, setup_mock):
        """
        Single query encoding via encode_query should produce same dimension
        as first element of encode_texts result.
        """
        from app.embedders.unified_embedder import unified_embedder
        query = "test consistency"
        single = unified_embedder.encode_query(query)
        batch = unified_embedder.encode_texts([query])

        assert len(single) == len(batch[0]) == 2048
        assert len(batch) == 1
