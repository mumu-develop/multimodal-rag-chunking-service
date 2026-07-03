"""异步搜索服务：支持纯文本、纯图片、图文联合异步搜索。"""
import asyncio
import json
import os
import uuid
from pathlib import Path

import structlog
from fastapi import UploadFile

from app.config import settings
from app.search.async_hybrid_search import async_hybrid_searcher
from app.repositories.async_milvus_repo import async_milvus_repo

logger = structlog.get_logger()


class AsyncSearchService:
    """异步搜索服务：支持纯文本、纯图片、图文联合搜索。"""

    async def search(
        self,
        query: str = "",
        query_image: str | None = None,
        top_k: int = 10,
        modal: str = "text",
        filters: dict | None = None,
        doc_ids: list[str] | None = None,
        collection_name: str = None,
        use_rerank: bool = True,
        data_type: str = "",
    ) -> list[dict]:
        """异步执行搜索，collection_name 必须已指定。"""
        if not collection_name:
            raise ValueError("collection_name is required")

        hits = await async_hybrid_searcher.search(
            query=query,
            query_image=query_image,
            top_k=top_k,
            modal=modal,
            filters=filters,
            doc_ids=doc_ids,
            collection_name=collection_name,
            use_rerank=use_rerank,
            data_type=data_type,
        )
        return self._format_hits(hits)

    async def search_by_image(
        self,
        image: UploadFile,
        query: str | None = None,
        top_k: int = 10,
        modal: str = "both",
        filters: dict | None = None,
        doc_ids: list[str] | None = None,
        collection_name: str = None,
        use_rerank: bool = True,
    ) -> list[dict]:
        """异步上传图片进行搜索，collection_name 必须已指定。"""
        if not collection_name:
            raise ValueError("collection_name is required")

        # 保存上传的图片到临时路径（异步文件写入）
        tmp_dir = os.path.join(settings.UPLOAD_DIR, "query_images")
        os.makedirs(tmp_dir, exist_ok=True)

        ext = Path(image.filename).suffix or ".jpg"
        tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{ext}")

        # 异步读取并写入文件
        content = await image.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        hits = await async_hybrid_searcher.search(
            query=query or "",
            query_image=tmp_path,
            top_k=top_k,
            modal=modal,
            filters=filters,
            doc_ids=doc_ids,
            collection_name=collection_name,
            use_rerank=use_rerank,
        )
        return self._format_hits(hits, default_modal="image")

    @staticmethod
    def _format_hits(hits: list[dict], default_modal: str = "text") -> list[dict]:
        """格式化搜索结果（同步操作，无需 async）。"""
        from app.repositories.chunk_repo import chunk_repo

        result_hits = []
        for hit in hits:
            metadata = hit.get("metadata", {}) or {}
            doc_name = metadata.get("doc_name", "")
            heading_text = metadata.get("heading_text", "")
            heading_level = metadata.get("heading_level", 0)

            chunk_obj = {
                "chunk_id": hit["chunk_id"],
                "doc_id": hit["doc_id"],
                "doc_name": doc_name,
                "chunk_type": hit["chunk_type"],
                "content": hit["content"],
                "caption": metadata.get("caption"),
                "page_num": hit.get("page_num", 0),
                "heading_text": heading_text,
                "heading_level": heading_level,
                "metadata": metadata,
                "doc_type": hit.get("doc_type", ""),
            }

            # FAQ 类型：添加 FAQ 特有字段
            if hit.get("data_type") == "faq" or hit["chunk_type"] == "faq":
                question = hit.get("content", "")  # 原始问题（content 存问题）
                answer = hit.get("summary", "")    # 答案
                chunk_obj["summary"] = answer
                chunk_obj["section"] = hit.get("section", "")   # 分类
                chunk_obj["tags"] = hit.get("tags", "")         # 关键词
                chunk_obj["priority"] = hit.get("priority", 0)  # 命中次数
                # FAQ 的 content 按 Q: A: 格式返回
                chunk_obj["content"] = f"Q: {question}\nA: {answer}"

            # 图片类型：提取 image_url
            if hit["chunk_type"] == "image":
                image_path = hit.get("image_path") or metadata.get("image_path", "")
                url = metadata.get("url", "")
                image_url = url if url else image_path
                if image_url and image_url.startswith("./"):
                    image_url = "/" + image_url[2:]
                chunk_obj["image_url"] = image_url

            # 处理嵌套表格命中：附带父表格信息
            parent_chunk = None
            if hit["chunk_type"] == "nested_table":
                metadata = hit.get("metadata", {})
                parent_chunk_id = metadata.get("parent_chunk_id")
                if parent_chunk_id:
                    parent = chunk_repo.get_by_chunk_id(parent_chunk_id)
                    if parent:
                        try:
                            parent_table_data = json.loads(parent.content)
                        except (json.JSONDecodeError, TypeError):
                            parent_table_data = {}
                        parent_chunk = {
                            "chunk_id": parent.chunk_id,
                            "chunk_type": parent.chunk_type,
                            "html": parent_table_data.get("html", ""),
                            "has_nested": parent.metadata.get("has_nested") if parent.metadata else False,
                            "nested_refs": parent.metadata.get("nested_refs") if parent.metadata else [],
                            "highlight_row": metadata.get("parent_row"),
                            "highlight_col": metadata.get("parent_col"),
                        }

            # 处理父表格命中：检查是否有嵌套表格
            nested_tables = None
            if hit["chunk_type"] == "table":
                metadata = hit.get("metadata", {})
                if metadata.get("has_nested"):
                    nested_refs = metadata.get("nested_refs", [])
                    nested_tables = []
                    for ref in nested_refs:
                        nested_chunk = chunk_repo.get_nested_table(hit["chunk_id"], ref.get("nested_index", 0))
                        if nested_chunk:
                            try:
                                nested_data = json.loads(nested_chunk.content)
                            except (json.JSONDecodeError, TypeError):
                                nested_data = {}
                            nested_tables.append({
                                "nested_index": ref.get("nested_index"),
                                "chunk_id": nested_chunk.chunk_id,
                                "row_count": nested_data.get("row_count", 0),
                                "col_count": nested_data.get("col_count", 0),
                                "parent_row": ref.get("parent_row"),
                                "parent_col": ref.get("parent_col"),
                            })

            result_hit = {
                "chunk": chunk_obj,
                "score": round(hit["score"], 4),
                "modal": hit.get("modal", default_modal),
                "rerank_score": round(hit.get("rerank_score", 0), 4) if hit.get("rerank_score") else None,
            }

            if parent_chunk:
                result_hit["parent_chunk"] = parent_chunk
            if nested_tables:
                result_hit["nested_tables"] = nested_tables

            result_hits.append(result_hit)
        return result_hits


# 单例
async_search_service = AsyncSearchService()