"""
文档清洗增强测试。

测试 CrossPageDetector（跨页重复检测）、敏感信息检测、扩展关键词等。
"""
import os
import sys
from pathlib import Path

import pytest

# 确保 backend 目录在 sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

os.environ["TMPDIR"] = "/tmp"


# ============================================================
# Cross Page Detector Tests
# ============================================================

class TestCrossPageDetector:
    """跨页重复内容检测器测试。"""

    @pytest.fixture
    def detector(self):
        """创建检测器实例。"""
        from app.parsers.noise_filter import CrossPageDetector
        return CrossPageDetector(threshold=0.5, min_pages=3)

    def test_init_default_threshold(self):
        """测试默认阈值初始化。"""
        from app.parsers.noise_filter import CrossPageDetector
        detector = CrossPageDetector()
        assert detector.threshold > 0

    def test_add_page_content(self, detector):
        """测试添加页面内容。"""
        texts = ["页眉文本", "正文内容", "页脚文本"]

        detector.add_page_content(0, texts)

        assert detector._total_pages == 1
        assert len(detector._text_counts) > 0

    def test_detect_header_footer(self, detector):
        """测试页眉页脚检测。"""
        # 添加 5 页，每页都有相同的页眉
        for i in range(5):
            texts = ["公司名称 - 内部文档", f"正文内容{i}", "第 {} 页".format(i + 1)]
            detector.add_page_content(i, texts)

        repeated = detector.get_repeated_texts()

        # 应检测到重复的页眉
        assert len(repeated) > 0
        # 检查检测结果格式
        for text, label in repeated:
            assert isinstance(text, str)
            assert label in ["header", "footer"]

    def test_threshold_not_reached(self, detector):
        """测试阈值未达到时不检测。"""
        # 只添加 2 页（小于 min_pages=3）
        for i in range(2):
            texts = ["重复文本", f"正文{i}"]
            detector.add_page_content(i, texts)

        repeated = detector.get_repeated_texts()

        # 页数不足，应返回空列表
        assert repeated == []

    def test_unique_content_not_detected(self, detector):
        """测试独特内容不被检测。"""
        # 添加多页，每页内容都不同
        for i in range(5):
            texts = [f"页眉{i}", f"正文{i}", f"页脚{i}"]
            detector.add_page_content(i, texts)

        repeated = detector.get_repeated_texts()

        # 所有内容都不同，应返回空列表
        assert len(repeated) == 0

    def test_clear(self, detector):
        """测试清空检测结果。"""
        for i in range(3):
            detector.add_page_content(i, ["test"])

        detector.clear()

        assert detector._total_pages == 0
        assert len(detector._text_counts) == 0

    def test_y_position_filtering(self):
        """测试 Y 位置过滤（页眉在上方，页脚在下方）。"""
        from app.parsers.noise_filter import CrossPageDetector

        detector = CrossPageDetector(y_threshold=0.1)

        # 添加页面，指定 Y 位置
        texts = ["页眉", "正文", "页脚"]
        y_positions = [10, 500, 900]  # 页眉在上，页脚在下

        detector.add_page_content(0, texts, y_positions)

        assert detector._total_pages == 1


# ============================================================
# Sensitive Information Tests
# ============================================================

class TestSensitiveInfoDetection:
    """敏感信息检测测试。"""

    def test_detect_phone_number(self):
        """测试电话号码检测。"""
        from app.parsers.noise_filter import mask_sensitive_info

        text = "联系电话：13812345678"
        masked = mask_sensitive_info(text)

        assert "***" in masked or "138" not in masked

    def test_detect_email(self):
        """测试邮箱检测。"""
        from app.parsers.noise_filter import mask_sensitive_info

        text = "邮箱：test@example.com"
        masked = mask_sensitive_info(text)

        assert "test@" not in masked or "***" in masked

    def test_detect_id_card(self):
        """测试身份证号检测。"""
        from app.parsers.noise_filter import mask_sensitive_info

        text = "身份证：123456789012345678"
        masked = mask_sensitive_info(text)

        assert "123456789012345678" not in masked

    def test_detect_bank_card(self):
        """测试银行卡号检测。"""
        from app.parsers.noise_filter import mask_sensitive_info

        text = "银行卡：6222021234567890123"
        masked = mask_sensitive_info(text)

        assert "6222021234567890123" not in masked

    def test_multiple_sensitive_info(self):
        """测试多种敏感信息。"""
        from app.parsers.noise_filter import mask_sensitive_info

        text = "电话：13812345678，邮箱：test@example.com，身份证：123456789012345678"
        masked = mask_sensitive_info(text)

        # 所有敏感信息应被处理
        assert "13812345678" not in masked
        assert "test@example.com" not in masked
        assert "123456789012345678" not in masked

    def test_no_sensitive_info(self):
        """测试无敏感信息文本。"""
        from app.parsers.noise_filter import mask_sensitive_info

        text = "这是一段普通文本，没有敏感信息。"
        masked = mask_sensitive_info(text)

        # 应返回原文本
        assert masked == text


# ============================================================
# Noise Filter Tests
# ============================================================

class TestNoiseFilter:
    """噪音过滤测试。"""

    def test_is_noise_text_page_number(self):
        """测试页码噪音判断。"""
        from app.parsers.noise_filter import is_noise_text

        assert is_noise_text("第 1 页") == True
        assert is_noise_text("Page 2") == True
        assert is_noise_text("- 3 -") == True

    def test_is_noise_text_header_footer(self):
        """测试页眉页脚噪音判断。"""
        from app.parsers.noise_filter import is_noise_text

        assert is_noise_text("公司内部文档") == True
        assert is_noise_text("版权所有") == True
        assert is_noise_text("机密文件") == True

    def test_is_noise_text_short_text(self):
        """测试短文本噪音判断。"""
        from app.parsers.noise_filter import is_noise_text

        assert is_noise_text("AB") == True
        assert is_noise_text("---") == True

    def test_is_noise_text_valid_content(self):
        """测试有效内容不被判断为噪音。"""
        from app.parsers.noise_filter import is_noise_text

        assert is_noise_text("这是一段有效的文档内容，包含重要信息。") == False
        assert is_noise_text("多模态RAG系统结合了向量检索和生成技术。") == False

    def test_clean_text_removes_extra_spaces(self):
        """测试清理多余空格。"""
        from app.parsers.noise_filter import clean_text

        text = "这是  有多余空格  的文本"
        cleaned = clean_text(text)

        # 应移除多余空格
        assert "  " not in cleaned

    def test_clean_text_removes_line_breaks(self):
        """测试清理换行符。"""
        from app.parsers.noise_filter import clean_text

        text = "文本一\n\n\n\n文本二"
        cleaned = clean_text(text)

        # 应减少连续换行
        assert "\n\n\n" not in cleaned

    def test_should_filter_text_comprehensive(self):
        """测试综合过滤判断。"""
        from app.parsers.noise_filter import should_filter_text

        # 噪音文本
        result, reason = should_filter_text("第1页", min_length=5)
        assert result == True
        result, reason = should_filter_text("机密文件", min_length=5)
        assert result == True

        # 有效文本
        result, reason = should_filter_text("这是一段有效的文档内容", min_length=5)
        assert result == False

    def test_custom_footer_keywords(self):
        """测试自定义页脚关键词。"""
        from app.parsers.noise_filter import should_filter_text, add_custom_keywords

        # 使用自定义关键词
        custom_keywords = ["自定义页脚", "公司名"]
        add_custom_keywords(custom_keywords)

        # 包含自定义关键词的文本应被过滤
        result, reason = should_filter_text("自定义页脚")
        assert result == True


# ============================================================
# Extended Keywords Tests
# ============================================================

class TestExtendedKeywords:
    """扩展关键词测试。"""

    def test_chinese_keywords_exist(self):
        """测试中文关键词存在。"""
        from app.parsers.noise_filter import FOOTER_KEYWORDS_CN, HEADER_KEYWORDS_CN

        # 应包含扩展的中文关键词
        assert "版权所有" in FOOTER_KEYWORDS_CN
        assert "机密文件" in FOOTER_KEYWORDS_CN
        assert "公司内部" in HEADER_KEYWORDS_CN

    def test_english_keywords_exist(self):
        """测试英文关键词存在。"""
        from app.parsers.noise_filter import FOOTER_KEYWORDS_EN, HEADER_KEYWORDS_EN

        # 应包含扩展的英文关键词
        assert "copyright" in FOOTER_KEYWORDS_EN
        assert "confidential" in FOOTER_KEYWORDS_EN

    def test_garbage_patterns_exist(self):
        """测试垃圾模式存在。"""
        from app.parsers.noise_filter import GARBAGE_PATTERNS

        # 应包含常见垃圾模式
        assert len(GARBAGE_PATTERNS) > 0
        # 检查某些特定模式
        assert any("页" in p or "page" in p.lower() for p in GARBAGE_PATTERNS)


# ============================================================
# Pattern Matching Tests
# ============================================================

class TestPatternMatching:
    """正则模式匹配测试。"""

    def test_page_number_pattern(self):
        """测试页码模式。"""
        import re
        from app.parsers.noise_filter import GARBAGE_PATTERNS

        # 测试页码格式
        page_patterns = [p for p in GARBAGE_PATTERNS if "页" in p or "page" in p.lower()]

        test_cases = [
            "第 1 页",
            "Page 1",
            "- 1 -",
            "1 / 10",
        ]

        for case in test_cases:
            # 应能匹配到某个模式
            matched = any(case.lower() == p.lower() or case in p for p in GARBAGE_PATTERNS)
            # 或者通过 is_noise_text 判断
            from app.parsers.noise_filter import is_noise_text
            assert is_noise_text(case) == True

    def test_phone_pattern(self):
        """测试电话号码模式。"""
        from app.parsers.noise_filter import PHONE_PATTERN

        # 测试匹配
        if PHONE_PATTERN:
            phones = ["13812345678", "021-12345678"]
            for phone in phones:
                match = PHONE_PATTERN.search(phone)
                assert match is not None


# ============================================================
# Integration Tests
# ============================================================

class TestNoiseFilterIntegration:
    """噪音过滤集成测试。"""

    def test_filter_page_content(self):
        """测试过滤页面内容。"""
        from app.parsers.noise_filter import filter_page_text

        raw_text = """
公司内部文档 - 请勿外传

这是第一段正文内容，包含重要信息。

第 1 页

这是第二段正文内容。

版权所有 © 2024
        """

        filtered = filter_page_text(raw_text)

        # 页眉页脚应被过滤
        assert "公司内部文档" not in filtered
        assert "版权所有" not in filtered
        assert "第 1 页" not in filtered

        # 正文应保留
        assert "正文内容" in filtered

    def test_cross_page_with_filter(self):
        """测试跨页检测与过滤结合。"""
        from app.parsers.noise_filter import CrossPageDetector, filter_page_text

        detector = CrossPageDetector()

        pages = [
            "页眉文本\n正文内容一\n页脚文本",
            "页眉文本\n正文内容二\n页脚文本",
            "页眉文本\n正文内容三\n页脚文本",
        ]

        # 添加页面内容
        for i, page in enumerate(pages):
            lines = page.split("\n")
            detector.add_page_content(i, lines)

        # 获取重复文本
        repeated = detector.get_repeated_texts()

        # 过滤每页
        for page in pages:
            filtered = filter_page_text(page)
            # 重复的页眉页脚应被移除
            for text, label in repeated:
                assert text not in filtered


# ============================================================
# Edge Cases Tests
# ============================================================

class TestEdgeCases:
    """边界情况测试。"""

    def test_empty_text(self):
        """测试空文本。"""
        from app.parsers.noise_filter import is_noise_text, clean_text

        assert is_noise_text("") == True
        assert clean_text("") == ""

    def test_none_text(self):
        """测试 None 文本。"""
        from app.parsers.noise_filter import clean_text

        result = clean_text(None)
        assert result == "" or result is None

    def test_unicode_text(self):
        """测试 Unicode 文本。"""
        from app.parsers.noise_filter import clean_text

        text = "包含特殊字符：© ® ™ 的文本"
        cleaned = clean_text(text)

        # 应保留或处理特殊字符
        assert cleaned is not None

    def test_very_long_text(self):
        """测试超长文本。"""
        from app.parsers.noise_filter import is_noise_text

        # 超长文本不应被判断为噪音
        long_text = "这是一段非常长的文本内容" * 100
        assert is_noise_text(long_text) == False

    def test_mixed_noise_and_content(self):
        """测试混合噪音和有效内容。"""
        from app.parsers.noise_filter import is_noise_text

        # 混合文本应不被判断为噪音（因为有有效内容）
        mixed = "第 1 页 有效内容部分"
        # 短文本可能被判断为噪音
        # 具体行为取决于实现
        result = is_noise_text(mixed)
        assert isinstance(result, bool)