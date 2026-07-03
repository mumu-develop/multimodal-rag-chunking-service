"""Chunkers 基础模块。

保留 generate_chunk_id 函数供将来使用。
ChunkStrategy 和 select_strategy 已废弃（旧流程）。
"""
import hashlib


def generate_chunk_id(doc_id: str, page_num: int, chunk_type: str, content: str, sort_order: int = 0) -> str:
    """基于文档和分块内容生成确定性 chunk_id。

    使用 doc_id + page_num + chunk_type + sort_order + content 的 SHA-256 前 32 位。
    同一次文档解析即使 Celery 重试，只要解析结果相同，chunk_id 就相同，
    使得 MySQL INSERT IGNORE 和 Milvus upsert 都是幂等的。

    Args:
        doc_id: 文档ID
        page_num: 页码
        chunk_type: 切片类型（text/table/image/code）
        content: 切片内容
        sort_order: 排序序号

    Returns:
        32位 SHA-256 hash 字符串
    """
    raw = f"{doc_id}:{page_num}:{chunk_type}:{sort_order}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


__all__ = ["generate_chunk_id"]