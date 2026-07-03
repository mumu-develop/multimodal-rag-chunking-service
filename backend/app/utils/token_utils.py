"""Token 计数工具（借鉴 RAGFlow）。

使用 tiktoken 计算 token 数量，适配嵌入模型的 token 限制。
- 中文：约 1.5-2 字符/token
- 英文：约 3-4 字符/token
- 字符计数可能导致嵌入时超 token 限制

使用 cl100k_base 编码器（GPT-4/GPT-3.5-turbo 使用的编码），
对于其他嵌入模型也基本兼容。
"""
import functools

import tiktoken

from app.config import settings

# 延迟加载 encoder，避免启动时加载耗时
_encoder = None


def _get_encoder():
    """延迟加载 tiktoken encoder。"""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def num_tokens_from_string(text: str) -> int:
    """计算文本的 token 数量。

    Args:
        text: 输入文本

    Returns:
        token 数量
    """
    if not settings.USE_TOKEN_COUNTING:
        # fallback 到字符计数
        return len(text)

    if not text:
        return 0

    encoder = _get_encoder()
    return len(encoder.encode(text))


def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """按 token 数截断文本。

    Args:
        text: 输入文本
        max_tokens: 最大 token 数

    Returns:
        截断后的文本
    """
    if not settings.USE_TOKEN_COUNTING:
        # fallback 到字符截断
        return text[:max_tokens]

    encoder = _get_encoder()
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoder.decode(truncated_tokens)


def extract_tail_by_tokens(text: str, tail_tokens: int) -> str:
    """提取文本尾部指定 token 数的内容（用于 overlap）。

    Args:
        text: 输入文本
        tail_tokens: 需要提取的尾部 token 数

    Returns:
        尾部文本
    """
    if not settings.USE_TOKEN_COUNTING:
        # fallback 到字符提取
        return text[-tail_tokens * 2:] if len(text) > tail_tokens * 2 else text

    encoder = _get_encoder()
    tokens = encoder.encode(text)
    if len(tokens) <= tail_tokens:
        return text

    tail_encoded = tokens[-tail_tokens:]
    return encoder.decode(tail_encoded)


def total_token_count_from_chunks(chunks: list[dict]) -> int:
    """计算所有 chunk 的总 token 数。

    Args:
        chunks: chunk 列表

    Returns:
        总 token 数
    """
    total = 0
    for chunk in chunks:
        content = chunk.get("content", "")
        if content:
            total += num_tokens_from_string(content)
    return total