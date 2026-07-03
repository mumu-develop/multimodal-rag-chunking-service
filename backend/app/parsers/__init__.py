"""Parsers 包：噪音过滤和基础类定义。

当前架构使用 Converter 流程，旧 Parser 已废弃。

保留模块：
- base.py: ParsedPage 等基础类定义（旧流程引用）
- noise_filter.py: 噪音过滤工具（被 Converter 和 MarkdownChunker 使用）

已删除模块（旧流程）：
- pdf_parser.py
- docx_parser.py
- pptx_parser.py
- markdown_parser.py
- excel_parser.py
- image_parser.py
- html_parser.py
- email_parser.py
- llm_cleaner.py
"""
from app.parsers.noise_filter import (
    clean_text,
    is_noise_text,
    should_filter_text,
)

__all__ = [
    "clean_text",
    "is_noise_text",
    "should_filter_text",
]