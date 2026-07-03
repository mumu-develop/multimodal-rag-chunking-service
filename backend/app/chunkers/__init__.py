"""Chunkers 包：切分质量评估。

当前架构使用 Converter + MarkdownChunker 流程，
旧流程（parse_document/chunk_pages）已废弃。

保留模块：
- base.py: 基础类定义
- chunk_quality_evaluator.py: 切分质量评估器（已集成到 tasks.py）

已删除模块（旧流程）：
- hierarchical_chunker.py
- slide_chunker.py
- semantic_chunker.py
- recursive_chunker.py
- table_chunker.py
- quality_loop.py（基于 ParsedPage 的旧版本）
- auto_selector.py（基于 ParsedPage 的旧版本）
- registry.py
"""

# 解析器注册表（用于插件扩展）
_PARSER_REGISTRY: dict = {}


def register_parser(name: str, parser_cls) -> None:
    """注册解析器类到全局注册表。

    Args:
        name: 解析器名称
        parser_cls: 解析器类
    """
    _PARSER_REGISTRY[name] = parser_cls


def get_parser(name: str):
    """获取注册的解析器类。

    Args:
        name: 解析器名称

    Returns:
        解析器类或 None
    """
    return _PARSER_REGISTRY.get(name)


from app.chunkers.chunk_quality_evaluator import (
    chunk_quality_evaluator,
    ChunkQualityEvaluator,
    ChunkQualityMetrics,
)

__all__ = [
    "chunk_quality_evaluator",
    "ChunkQualityEvaluator",
    "ChunkQualityMetrics",
    "register_parser",
    "get_parser",
]