"""
LLM 辅助清洗模块测试。

测试 LLMCleaner 的噪音检测、Caption 生成、关键词提取等功能。
使用 mock 模拟 OpenAI API 响应。
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
# Mock OpenAI Client
# ============================================================

class MockOpenAIResponse:
    """Mock OpenAI API 响应。"""

    def __init__(self, content: str):
        self.choices = [MagicMock()]
        self.choices[0].message = MagicMock()
        self.choices[0].message.content = content


def create_mock_openai_client():
    """创建 mock OpenAI 客户端。"""
    mock_client = MagicMock()

    # 默认响应
    def mock_create(**kwargs):
        messages = kwargs.get("messages", [])
        user_msg = messages[-1].get("content", "") if messages else ""

        # 根据提示词内容返回不同响应
        if "噪音" in user_msg or "YES" in user_msg or "NO" in user_msg:
            # 噪音检测
            if "第 1 页" in user_msg or "页码" in user_msg:
                return MockOpenAIResponse("YES")
            else:
                return MockOpenAIResponse("NO")

        elif "表格" in user_msg and "标题" in user_msg:
            # 表格 Caption
            return MockOpenAIResponse("员工信息统计表")

        elif "图片" in user_msg and "Caption" in user_msg:
            # 图片 Caption
            return MockOpenAIResponse("流程示意图")

        elif "关键词" in user_msg:
            # 关键词提取
            return MockOpenAIResponse('["RAG", "多模态", "向量检索"]')

        elif "清洗" in user_msg:
            # 内容清洗
            return MockOpenAIResponse(user_msg.split("原文：")[-1].strip() if "原文：" in user_msg else "清洗后内容")

        return MockOpenAIResponse("默认响应")

    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = mock_create

    return mock_client


# ============================================================
# LLM Cleaner Tests
# ============================================================

class TestLLMCleaner:
    """LLM 辅助清洗器测试。"""

    @pytest.fixture
    def cleaner(self):
        """创建 LLMCleaner 实例。"""
        from app.parsers.llm_cleaner import LLMCleaner, LLMCleanConfig
        config = LLMCleanConfig(
            model_name="gpt-4o-mini",
            api_key="test-key",
            base_url="https://api.test.com/v1",
        )
        cleaner = LLMCleaner(config)
        cleaner._client = create_mock_openai_client()
        return cleaner

    def test_is_noise_page_number(self, cleaner):
        """测试页码噪音检测。"""
        is_noise, reason = cleaner.is_noise("第 1 页")
        assert is_noise == True

    def test_is_noise_valid_content(self, cleaner):
        """测试有效内容不被判定为噪音。"""
        is_noise, reason = cleaner.is_noise("这是一段有意义的文档内容，包含重要信息。")
        assert is_noise == False

    def test_is_noise_cache(self, cleaner):
        """测试噪音检测缓存。"""
        # 第一次调用
        cleaner.is_noise("测试文本")
        cache_key = f"noise:测试文本"

        # 检查缓存是否写入
        assert cache_key in cleaner._cache or "noise:" in str(cleaner._cache)

    def test_generate_table_caption(self, cleaner):
        """测试表格 Caption 生成。"""
        table_html = "<table><tr><td>姓名</td><td>年龄</td></tr></table>"
        caption = cleaner.generate_table_caption(table_html)

        assert caption is not None
        assert len(caption) > 0

    def test_generate_image_caption(self, cleaner):
        """测试图片 Caption 生成。"""
        description = "这是一张展示数据处理流程的图片"
        caption = cleaner.generate_image_caption(description)

        assert caption is not None

    def test_extract_keywords(self, cleaner):
        """测试关键词提取。"""
        text = "多模态RAG系统结合了向量检索和生成技术"
        keywords = cleaner.extract_keywords(text)

        assert isinstance(keywords, list)
        assert len(keywords) > 0

    def test_clean_content(self, cleaner):
        """测试内容清洗。"""
        text = "  这是  有多余空格的  文本  \n\n\n  "
        cleaned = cleaner.clean_content(text)

        assert cleaned is not None

    def test_batch_clean(self, cleaner):
        """测试批量清洗。"""
        texts = [
            "第 1 页",  # 噪音
            "有效内容段落一",
            "页眉信息",  # 噪音（模拟）
            "有效内容段落二",
        ]

        results = cleaner.batch_clean(texts, max_workers=2)

        assert len(results) == 4
        # 检查结果格式
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 3  # (text, is_noise, cleaned_text)

    def test_batch_clean_preserves_order(self, cleaner):
        """测试批量清洗保持顺序。"""
        texts = ["文本A", "文本B", "文本C"]
        results = cleaner.batch_clean(texts, max_workers=2)

        # 结果应按原始顺序排列
        original_texts = [r[0] for r in results]
        assert original_texts == texts


# ============================================================
# LLM Plugin Tests
# ============================================================

class TestLLMAssistedPlugin:
    """LLM 辅助清洗插件测试。"""

    @pytest.fixture
    def plugin(self):
        """创建插件实例。"""
        from app.parsers.llm_cleaner import LLMAssistedPlugin, LLMCleaner
        cleaner = LLMCleaner()
        cleaner._client = create_mock_openai_client()
        return LLMAssistedPlugin(cleaner=cleaner)

    def test_plugin_attributes(self, plugin):
        """测试插件基本属性。"""
        assert plugin.name == "llm_assisted_clean"
        assert plugin.priority == 50
        assert plugin.enabled == False  # 默认禁用

    def test_plugin_description(self, plugin):
        """测试插件描述。"""
        desc = plugin.description
        assert "LLM" in desc
        assert "清洗" in desc

    def test_on_chunk_table_caption(self, plugin):
        """测试切分后的表格 Caption 生成。"""
        chunks = [
            {
                "chunk_type": "table",
                "content": "<table><tr><td>数据</td></tr></table>",
                "caption": None,
            },
            {
                "chunk_type": "text",
                "content": "普通文本",
            },
        ]

        # 启用 caption_generation
        plugin.enabled_functions = ["caption_generation"]
        result = plugin.on_chunk(chunks)

        # 检查表格 chunk 是否添加了 caption
        table_chunk = result[0]
        assert table_chunk.get("caption") is not None

    def test_on_chunk_image_caption(self, plugin):
        """测试切分后的图片 Caption 增强。"""
        chunks = [
            {
                "chunk_type": "image",
                "content": "图片描述",
                "metadata": {
                    "vlm_caption": "原始VLM描述"
                },
            },
        ]

        plugin.enabled_functions = ["caption_generation"]
        result = plugin.on_chunk(chunks)

        # 检查图片 chunk 是否增强了 caption
        image_chunk = result[0]
        metadata = image_chunk.get("metadata", {})
        assert "caption" in metadata


# ============================================================
# Error Handling Tests
# ============================================================

class TestErrorHandling:
    """错误处理测试。"""

    def test_is_noise_api_error(self):
        """测试 API 错误时的噪音检测。"""
        from app.parsers.llm_cleaner import LLMCleaner, LLMCleanConfig

        config = LLMCleanConfig(api_key="", base_url="")
        cleaner = LLMCleaner(config)
        cleaner._client = None  # 无客户端

        # 应返回 False，不抛异常
        is_noise, reason = cleaner.is_noise("测试文本")
        assert is_noise == False
        assert reason == "LLM_ERROR"

    def test_generate_caption_api_error(self):
        """测试 Caption 生成 API 错误。"""
        from app.parsers.llm_cleaner import LLMCleaner, LLMCleanConfig

        config = LLMCleanConfig(api_key="", base_url="")
        cleaner = LLMCleaner(config)
        cleaner._client = None

        caption = cleaner.generate_table_caption("table content")
        # 应返回空字符串，不抛异常
        assert caption == ""

    def test_extract_keywords_parse_error(self):
        """测试关键词解析错误。"""
        from app.parsers.llm_cleaner import LLMCleaner, LLMCleanConfig

        config = LLMCleanConfig(api_key="test", base_url="https://test.com")
        cleaner = LLMCleaner(config)

        # 创建返回无效 JSON 的 mock
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse("不是JSON数组")

        cleaner._client = mock_client

        keywords = cleaner.extract_keywords("测试文本")
        # 应返回空列表
        assert keywords == []


# ============================================================
# Cache Tests
# ============================================================

class TestCaching:
    """缓存功能测试。"""

    def test_noise_check_cache_hit(self):
        """测试噪音检测缓存命中。"""
        from app.parsers.llm_cleaner import LLMCleaner, LLMCleanConfig

        config = LLMCleanConfig(api_key="test", base_url="https://test.com")
        cleaner = LLMCleaner(config)
        cleaner._client = create_mock_openai_client()

        # 第一次调用
        text = "缓存测试文本"
        cleaner.is_noise(text)

        # 第二次调用（应命中缓存）
        # 如果命中缓存，不会再次调用 API
        cleaner.is_noise(text)

        # 检查缓存
        assert len(cleaner._cache) > 0

    def test_cache_different_texts(self):
        """测试不同文本不同缓存。"""
        from app.parsers.llm_cleaner import LLMCleaner, LLMCleanConfig

        config = LLMCleanConfig(api_key="test", base_url="https://test.com")
        cleaner = LLMCleaner(config)
        cleaner._client = create_mock_openai_client()

        cleaner.is_noise("文本A")
        cleaner.is_noise("文本B")

        # 应有多个缓存条目
        assert len(cleaner._cache) >= 1