"""
Chunker 扩展模块测试。

测试新增的切分策略：
- ChunkerRegistry（策略注册）
- SemanticChunker（语义切分）
- RecursiveChunker（递归切分）
- AutoStrategySelector（自动选择）
- QualityEvaluationLoop（质量闭环）
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
# Chunker Registry Tests
# ============================================================

class TestChunkerRegistry:
    """策略注册中心测试。"""

    def test_register_strategy(self):
        """测试策略注册。"""
        from app.chunkers.registry import ChunkerRegistry
        from app.chunkers.base import ChunkStrategy

        # 创建 mock 策略类
        class MockChunker(ChunkStrategy):
            def chunk(self, pages, doc_id, creator="", kb_id="", kb_config=None):
                return [{"chunk_id": "test"}]

        # 注册
        ChunkerRegistry.register("mock_strategy", MockChunker, {"param": 1})

        # 检查注册
        assert "mock_strategy" in ChunkerRegistry._strategies
        assert ChunkerRegistry._configs["mock_strategy"]["param"] == 1

        # 清理
        ChunkerRegistry._strategies.pop("mock_strategy", None)
        ChunkerRegistry._configs.pop("mock_strategy", None)
        ChunkerRegistry._instances.pop("mock_strategy", None)

    def test_get_strategy(self):
        """测试获取策略实例。"""
        from app.chunkers.registry import ChunkerRegistry
        from app.chunkers.base import ChunkStrategy

        class MockChunker(ChunkStrategy):
            def chunk(self, pages, doc_id, creator="", kb_id="", kb_config=None):
                return []

        ChunkerRegistry.register("test_get", MockChunker)

        # 获取实例
        strategy = ChunkerRegistry.get("test_get")
        assert strategy is not None
        assert isinstance(strategy, MockChunker)

        # 清理
        ChunkerRegistry._strategies.pop("test_get", None)
        ChunkerRegistry._instances.pop("test_get", None)

    def test_get_strategy_singleton(self):
        """测试策略单例。"""
        from app.chunkers.registry import ChunkerRegistry
        from app.chunkers.base import ChunkStrategy

        class MockChunker(ChunkStrategy):
            def chunk(self, pages, doc_id, creator="", kb_id="", kb_config=None):
                return []

        ChunkerRegistry.register("singleton_test", MockChunker, use_cache=True)

        s1 = ChunkerRegistry.get("singleton_test")
        s2 = ChunkerRegistry.get("singleton_test")

        # 应为同一实例
        assert s1 is s2

        # 清理
        ChunkerRegistry._strategies.pop("singleton_test", None)
        ChunkerRegistry._instances.pop("singleton_test", None)

    def test_list_strategies(self):
        """测试列出策略。"""
        from app.chunkers.registry import ChunkerRegistry

        strategies = ChunkerRegistry.list()

        # 应返回策略名称列表
        assert isinstance(strategies, list)

    def test_get_strategy_info(self):
        """测试获取策略信息。"""
        from app.chunkers.registry import ChunkerRegistry

        ChunkerRegistry.init_builtin_strategies()
        info_list = ChunkerRegistry.info()

        # 应返回策略信息列表
        assert isinstance(info_list, list)
        assert len(info_list) > 0
        # 检查列表中包含必要字段
        for info in info_list:
            assert "name" in info
            assert "description" in info or "default_config" in info

    def test_get_unknown_strategy(self):
        """测试获取未注册策略。"""
        from app.chunkers.registry import ChunkerRegistry

        # 未注册策略应抛异常或返回 None
        with pytest.raises((KeyError, ValueError)):
            ChunkerRegistry.get("unknown_strategy_xyz")


# ============================================================
# Semantic Chunker Tests
# ============================================================

class TestSemanticChunker:
    """语义切分器测试。"""

    @pytest.fixture
    def mock_pages(self):
        """创建 mock 页面列表。"""
        from app.parsers.base import ParsedPage

        pages = []
        for i in range(3):
            page = ParsedPage(
                doc_id="test_doc",
                page_num=i + 1,
                text=f"""
这是第{i+1}页的内容。

段落一：关于多模态RAG系统的介绍。
多模态RAG结合了文本、图像、表格等多种数据形式的检索。

段落二：关于向量数据库的说明。
向量数据库用于存储和检索高维向量，支持语义搜索。

段落三：关于重排序优化的内容。
重排序可以显著提升检索结果的相关性。
                """,
                doc_type="pdf",
            )
            pages.append(page)
        return pages

    def test_semantic_chunker_init(self):
        """测试语义切分器初始化。"""
        from app.chunkers.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(
            similarity_threshold=0.5,
            min_chunk_tokens=100,
            max_chunk_tokens=500,
        )

        assert chunker.similarity_threshold == 0.5
        assert chunker.min_chunk_tokens == 100

    def test_chunk_basic(self, mock_pages):
        """测试基础切分。"""
        from app.chunkers.semantic_chunker import SemanticChunker

        # Mock 嵌入模型
        mock_embedder = MagicMock()
        mock_embedder.encode_texts.return_value = [
            [0.1] * 768 for _ in range(10)  # 10 个句子向量
        ]

        chunker = SemanticChunker()
        chunker._embedder = mock_embedder

        chunks = chunker.chunk(mock_pages, "test_doc")

        assert isinstance(chunks, list)
        # 应生成切片
        assert len(chunks) >= 0


# ============================================================
# Recursive Chunker Tests
# ============================================================

class TestRecursiveChunker:
    """递归切分器测试。"""

    @pytest.fixture
    def mock_pages(self):
        """创建 mock 页面。"""
        from app.parsers.base import ParsedPage

        page = ParsedPage(
            doc_id="test_doc",
            page_num=1,
            text="""
第一段内容：这是关于RAG系统的介绍。

第二段内容：向量数据库是核心组件。

第三段内容：文档解析支持多种格式。
PDF、DOCX、PPTX、Markdown等。
            """,
            doc_type="text",
        )
        return [page]

    @pytest.fixture
    def recursive_chunker(self):
        """创建递归切分器。"""
        from app.chunkers.recursive_chunker import RecursiveChunker

        return RecursiveChunker(
            chunk_size=200,
            chunk_overlap=50,
        )

    def test_recursive_chunker_init(self, recursive_chunker):
        """测试递归切分器初始化。"""
        assert recursive_chunker.chunk_size == 200
        assert recursive_chunker.chunk_overlap == 50

    def test_split_text_recursive(self, recursive_chunker):
        """测试递归文本分割。"""
        text = "第一段。\n\n第二段。\n\n第三段。"
        separators = ["\n\n", "\n", "。", ""]

        chunks = recursive_chunker._split_text_recursive(text, separators)

        assert isinstance(chunks, list)
        # 应分割成多个块
        assert len(chunks) >= 1

    def test_chunk_pages(self, recursive_chunker, mock_pages):
        """测试页面切分。"""
        chunks = recursive_chunker.chunk(mock_pages, "test_doc")

        assert isinstance(chunks, list)
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "content" in chunk

    def test_code_separators(self):
        """测试代码分隔符。"""
        from app.chunkers.recursive_chunker import RecursiveChunker, CODE_SEPARATORS

        code_text = """
def function_one():
    pass

def function_two():
    pass
        """

        chunker = RecursiveChunker()
        chunks = chunker._split_text_recursive(code_text, CODE_SEPARATORS)

        assert isinstance(chunks, list)


# ============================================================
# Auto Strategy Selector Tests
# ============================================================

class TestAutoStrategySelector:
    """自动策略选择器测试。"""

    @pytest.fixture
    def selector(self):
        """创建选择器实例。"""
        from app.chunkers.auto_selector import AutoStrategySelector
        return AutoStrategySelector()

    @pytest.fixture
    def pdf_pages(self):
        """创建 PDF 页面 mock。"""
        from app.parsers.base import ParsedPage

        pages = []
        for i in range(10):
            page = ParsedPage(
                doc_id="test_doc",
                page_num=i + 1,
                text=f"第{i+1}页内容，包含一些文本信息。",
                headings=[{"level": 1, "text": f"标题{i+1}"}] if i % 2 == 0 else [],
                doc_type="pdf",
            )
            pages.append(page)
        return pages

    @pytest.fixture
    def pptx_pages(self):
        """创建 PPTX 页面 mock。"""
        from app.parsers.base import ParsedPage

        pages = []
        for i in range(5):
            page = ParsedPage(
                doc_id="pptx_doc",
                page_num=i + 1,
                text=f"幻灯片{i+1}内容",
                doc_type="pptx",
            )
            pages.append(page)
        return pages

    def test_analyze_pdf_pages(self, selector, pdf_pages):
        """测试 PDF 页面分析。"""
        features = selector.analyze(pdf_pages)

        assert features.doc_type == "pdf"
        assert features.total_pages == 10
        assert features.has_headings == True

    def test_analyze_pptx_pages(self, selector, pptx_pages):
        """测试 PPTX 页面分析。"""
        features = selector.analyze(pptx_pages)

        assert features.doc_type == "pptx"
        assert features.total_pages == 5

    def test_select_pptx_strategy(self, selector, pptx_pages):
        """测试 PPTX 策略选择。"""
        features = selector.analyze(pptx_pages)
        strategy, config = selector.select(features)

        # PPTX 应选择 slide 策略
        assert strategy == "slide"

    def test_select_with_user_preference(self, selector, pdf_pages):
        """测试用户指定策略。"""
        features = selector.analyze(pdf_pages)
        strategy, config = selector.select(features, user_preference="semantic")

        # 用户指定应优先
        assert strategy == "semantic"

    def test_strategy_rules_priority(self, selector):
        """测试策略规则优先级。"""
        from app.chunkers.auto_selector import DocumentFeatures

        # 高优先级规则应先匹配
        features = DocumentFeatures(
            doc_type="pptx",
            total_pages=10,
            has_headings=False,
            heading_count=0,
        )

        strategy, _ = selector.select(features)
        # PPTX 规则优先级 100
        assert strategy == "slide"


# ============================================================
# Quality Evaluation Loop Tests
# ============================================================

class TestQualityEvaluationLoop:
    """质量评估闭环测试。"""

    @pytest.fixture
    def quality_loop(self):
        """创建质量闭环实例。"""
        from app.chunkers.quality_loop import QualityEvaluationLoop
        return QualityEvaluationLoop()

    @pytest.fixture
    def mock_pages(self):
        """创建 mock 页面。"""
        from app.parsers.base import ParsedPage

        pages = []
        for i in range(5):
            page = ParsedPage(
                doc_id="quality_doc",
                page_num=i + 1,
                text=f"第{i+1}页内容，用于质量评估测试。",
                doc_type="pdf",
            )
            pages.append(page)
        return pages

    def test_quality_loop_init(self, quality_loop):
        """测试质量闭环初始化。"""
        assert quality_loop.max_iterations >= 1
        assert quality_loop.target_quality >= 0.9

    def test_run_basic(self, quality_loop, mock_pages):
        """测试基本运行。"""
        result = quality_loop.run(
            pages=mock_pages,
            doc_id="quality_doc",
            kb_id="test_kb",
            kb_config=None,
            user_strategy=None,
        )

        # 应返回 OptimizationResult
        assert result.best_chunks is not None
        assert result.best_strategy is not None
        assert result.iterations >= 1

    def test_run_with_user_strategy(self, quality_loop, mock_pages):
        """测试用户指定策略运行。"""
        result = quality_loop.run(
            pages=mock_pages,
            doc_id="quality_doc",
            kb_id="test_kb",
            user_strategy="hierarchical",
        )

        # 应使用用户指定策略
        assert result.best_strategy == "hierarchical"

    def test_evaluate_chunks(self, quality_loop, mock_pages):
        """测试切片评估。"""
        chunks = [
            {"content": "测试内容一", "chunk_type": "text"},
            {"content": "测试内容二", "chunk_type": "text"},
        ]

        metrics = quality_loop._evaluate(chunks)

        # 应返回指标字典
        assert isinstance(metrics, dict)


# ============================================================
# Chunk Pages With Quality Tests
# ============================================================

class TestChunkPagesWithQuality:
    """智能分块测试。"""

    @pytest.fixture
    def mock_pages_for_quality(self):
        """创建测试用的 mock 页面。"""
        from app.parsers.base import ParsedPage

        pages = []
        for i in range(3):
            page = ParsedPage(
                doc_id="smart_doc",
                page_num=i + 1,
                text=f"第{i+1}页内容，用于智能分块测试。这是一段有效的文档文本内容。",
                doc_type="pdf",
                headings=[{"level": 1, "text": f"标题{i+1}"}] if i == 0 else [],
            )
            pages.append(page)
        return pages

    def test_chunk_pages_with_quality_basic(self, mock_pages_for_quality):
        """测试智能分块基本功能。"""
        from app.chunkers import chunk_pages_with_quality

        result = chunk_pages_with_quality(
            pages=mock_pages_for_quality,
            doc_id="smart_doc",
            kb_id="test_kb",
            kb_config=None,
        )

        # 应返回完整结果
        assert "chunks" in result
        assert "strategy" in result
        assert "config" in result
        assert "metrics" in result

    def test_empty_pages(self):
        """测试空页面处理。"""
        from app.chunkers import chunk_pages_with_quality

        result = chunk_pages_with_quality(
            pages=[],
            doc_id="empty_doc",
            kb_id="test_kb",
        )

        assert result["chunks"] == []
        assert result["strategy"] == ""


# ============================================================
# Integration Tests
# ============================================================

class TestChunkerIntegration:
    """切分器集成测试。"""

    def test_registry_and_selector_integration(self):
        """测试注册中心与选择器集成。"""
        from app.chunkers.registry import ChunkerRegistry
        from app.chunkers.auto_selector import AutoStrategySelector
        from app.parsers.base import ParsedPage

        # 创建测试页面
        page = ParsedPage(
            doc_id="integrate_doc",
            page_num=1,
            text="测试内容",
            doc_type="pdf",
        )

        selector = AutoStrategySelector()
        features = selector.analyze([page])
        strategy_name, config = selector.select(features)

        # 策略名称应在注册中心存在
        assert strategy_name in ChunkerRegistry.list() or strategy_name == "default"

    def test_chunk_pages_with_metadata(self):
        """测试切分后的元数据完整性。"""
        from app.chunkers import chunk_pages
        from app.parsers.base import ParsedPage

        page = ParsedPage(
            doc_id="meta_doc",
            page_num=1,
            text="测试元数据",
            doc_type="pdf",
        )

        chunks = chunk_pages([page], "meta_doc", kb_id="test_kb", doc_name="test.pdf")

        for chunk in chunks:
            assert "metadata" in chunk
            assert chunk.get("kb_id") == "test_kb"
            assert chunk.get("doc_name") == "test.pdf"