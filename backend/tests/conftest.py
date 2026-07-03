"""
Pytest fixtures for RAG evaluation tests.

Provides isolated Milvus Lite instances, MySQL mock, and test data.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend/ is on sys.path so `app.*` imports work
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

# Set TMPDIR to /tmp before importing app modules (Milvus Lite socket path limit)
os.environ["TMPDIR"] = "/tmp"

# ============================================================
# CRITICAL: Set test env vars BEFORE any app module is imported.
# This ensures the settings singleton is initialized for testing.
# ============================================================
_TEST_ENV_VARS = {
    "RERANK_ENABLED": "false",
    "HYBRID_SEARCH_ALPHA": "0.5",
    "BM25_ENABLED": "true",
    "UNIFIED_VECTOR_DIM": "2560",
    "MILVUS_COLLECTION_NAME": "test_rag_chunks",
    "DATABASE_URI": "mysql+pymysql://test:test@localhost:3306/test_db",
    "DASHSCOPE_API_KEY": "sk-test-key",
}
for key, val in _TEST_ENV_VARS.items():
    os.environ[key] = val

# Also unset MILVUS_LOCAL_DB to avoid Milvus Lite auto-init in conftest
os.environ.pop("MILVUS_LOCAL_DB", None)


# ============================================================
# Settings override fixture
# ============================================================

@pytest.fixture
def mock_settings(monkeypatch):
    """
    Provide the pre-configured test settings.
    The env vars were set at module load time so the settings
    singleton already has the correct test values.
    """
    from app.config import settings
    return settings


# ============================================================
# Milvus Lite availability check
# ============================================================

def _is_milvus_lite_available() -> bool:
    """Check if Milvus Lite can actually start."""
    try:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="milvus_check_") as tmpdir:
            db_path = os.path.join(tmpdir, "check.db")
            from pymilvus import MilvusClient
            client = MilvusClient(uri=db_path)
            client.list_collections()
            return True
    except Exception:
        return False


_MILVUS_AVAILABLE = _is_milvus_lite_available()


# ============================================================
# Milvus Lite isolated fixture
# ============================================================

@pytest.fixture
def milvus_temp_db():
    """Create a temporary Milvus Lite .db file path."""
    with tempfile.TemporaryDirectory(prefix="milvus_test_") as tmpdir:
        db_path = os.path.join(tmpdir, "test_milvus.db")
        yield db_path


@pytest.fixture
def milvus_client_isolated(mock_settings, milvus_temp_db, monkeypatch):
    """
    Create an isolated MilvusClient with a temporary .db file.
    Automatically cleans up the collection after the test.
    Skips the test if Milvus Lite is not available.
    """
    if not _MILVUS_AVAILABLE:
        pytest.skip("Milvus Lite is not available in this environment")

    # Reload settings to pick up new MILVUS_LOCAL_DB
    from app import config
    new_settings = config.Settings()
    monkeypatch.setattr(config, "settings", new_settings)

    # Patch settings on all consuming modules
    for mod_path in [
        "app.milvus_client",
        "app.embedders.unified_embedder",
        "app.embedders.reranker",
        "app.search.hybrid_search",
    ]:
        try:
            mod = sys.modules[mod_path]
            monkeypatch.setattr(mod, "settings", new_settings)
        except KeyError:
            pass  # Module not yet imported

    # Clear the singleton milvus_client module-level instance
    from app import milvus_client as milvus_module
    milvus_module.milvus_client._client = None

    from app.milvus_client import milvus_client

    # Create collection
    collection_name = milvus_client.connect()
    yield milvus_client

    # Cleanup: drop collection
    try:
        milvus_client.drop_collection()
    except Exception:
        pass
    # Reset singleton for next test
    milvus_client._client = None


# ============================================================
# MySQL mock fixture
# ============================================================

@pytest.fixture
def mysql_mock():
    """
    Mock MySQL storage functions.
    Returns a dict that tracks inserted chunks for verification.
    """
    storage_data = {}  # chunk_id -> chunk_dict

    def mock_insert_doc_chunks(chunks):
        for c in chunks:
            storage_data[c["chunk_id"]] = c
        return True

    class MockChunk:
        """Mock SQLAlchemy DocChunk object."""
        def __init__(self, data):
            self.chunk_id = data["chunk_id"]
            self.doc_id = data["doc_id"]
            self.chunk_type = data["chunk_type"]
            self.content = data["content"]
            self.caption = data.get("caption")
            self.page_num = data["page_num"]
            self.meta_data = data.get("metadata", {})
            self.status = data.get("status", 1)
            self.audit_status = data.get("audit_status", 0)
            self.creator = data.get("creator", "")
            self.sort_order = data.get("sort_order", 0)
            self.heading_level = data.get("heading_level", 0)
            self.heading_text = data.get("heading_text", "")

    def mock_get_chunks_by_doc(doc_id, page=1, size=10, chunk_type=None):
        items = [
            MockChunk(c) for c in storage_data.values()
            if c["doc_id"] == doc_id
        ]
        if chunk_type:
            items = [c for c in items if c.chunk_type == chunk_type]
        total = len(items)
        start = (page - 1) * size
        return total, items[start:start + size]

    with patch("app.storage.insert_doc_chunks", side_effect=mock_insert_doc_chunks):
        with patch("app.storage.get_chunks_by_doc", side_effect=mock_get_chunks_by_doc):
            yield storage_data


# ============================================================
# Test document fixture
# ============================================================

@pytest.fixture
def sample_doc_path():
    """Path to the controlled test document."""
    return str(_BACKEND_DIR / "tests" / "sample_documents" / "test_doc.md")


@pytest.fixture
def sample_doc_content(sample_doc_path):
    """Read the test document content."""
    with open(sample_doc_path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# Golden dataset fixture
# ============================================================

@pytest.fixture
def golden_dataset():
    """Load the golden evaluation dataset."""
    dataset_path = _BACKEND_DIR / "tests" / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Test chunk data factory
# ============================================================

@pytest.fixture
def test_chunks_factory():
    """
    Factory to create test chunk dictionaries for insertion into Milvus.
    Usage: chunks = test_chunks_factory(doc_id="doc1", n_chunks=5)
    """
    def _create_chunks(doc_id="test_doc_001", n_chunks=5, doc_type="markdown"):
        """Create n_chunks test dictionaries with realistic content."""
        sample_texts = [
            "多模态RAG（Retrieval-Augmented Generation）是一种结合多种模态（文本、图像、表格）的检索增强生成技术。"
            "它通过从外部知识库中检索相关内容，增强大语言模型的生成能力。",

            "文档解析是RAG系统的第一步，支持PDF、DOCX、PPTX、Markdown等多种格式。"
            "解析器提取文档的文本内容、图像、表格和结构信息。",

            "分块策略包括文本分块、表格分块和图像分块。"
            "文本使用RecursiveCharacterTextSplitter进行分块，参数为chunk_size=800, chunk_overlap=150。",

            "向量嵌入使用Qwen3-VL-Embedding-2B模型，支持文本和图像的统一嵌入。"
            "文本嵌入维度为2048，图像嵌入维度为2048。",

            "Milvus是一个开源向量数据库，支持HNSW、IVF_FLAT等多种索引类型。"
            "混合检索结合dense向量和sparse BM25，通过WeightedRanker融合结果。",

            "重排序使用bge-reranker-v2-m3交叉编码器模型，对候选结果进行精排。"
            "重排序可以显著提升top-k准确率。",

            "系统架构采用FastAPI作为Web框架，Celery+Redis实现异步任务队列。"
            "MySQL存储元数据，Milvus存储向量数据。",

            "混合搜索的alpha参数控制dense和sparse的权重。"
            "alpha=0.5表示等权重，alpha=1.0表示仅dense，alpha=0表示仅sparse。",

            "图片处理流程：从文档中提取图片，使用VLM生成caption，"
            "然后通过OpenCLIP或统一嵌入模型编码为向量。",

            "任务状态管理：上传文档后创建异步任务，状态包括pending、processing、completed、failed。"
            "进度通过Celery的update_state机制报告。",
        ]
        chunk_types = ["text", "text", "text", "text", "text", "text", "text", "text", "image", "text"]

        chunks = []
        for i in range(min(n_chunks, len(sample_texts))):
            chunk_id = f"chunk_{doc_id}_{i:03d}"
            content = sample_texts[i % len(sample_texts)]
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_type": chunk_types[i % len(chunk_types)],
                "content": content,
                "caption": "图片描述：测试图片" if chunk_types[i % len(chunk_types)] == "image" else None,
                "page_num": (i // 3) + 1,
                "metadata": {"doc_type": doc_type, "source": "test"},
                "image_path": "/test/image.png" if chunk_types[i % len(chunk_types)] == "image" else "",
                "vector": [0.0] * 2560,  # placeholder
            })
        return chunks

    return _create_chunks
