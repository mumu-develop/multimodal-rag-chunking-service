"""Service: Search (text, image, multimodal)."""
import json
import os
import uuid
from pathlib import Path

import structlog
from fastapi import UploadFile

from app.config import settings
from app.search.hybrid_search import HybridSearcher

logger = structlog.get_logger()


class SearchService:
    """搜索服务：支持纯文本、纯图片、图文联合搜索。"""

    def search(
        self,
        query: str = "",
        query_image: str | None = None,
        top_k: int = 10,
        modal: str = "text",
        filters: dict | None = None,
        doc_ids: list[str] | None = None,
        collection_name: str = None,
        use_rerank: bool = True,
        data_type: str = "",  # 数据类型过滤: faq/chunk/""
    ) -> list[dict]:
        """执行搜索，collection_name 必须已指定。"""
        if not collection_name:
            raise ValueError("collection_name is required")

        hybrid_searcher = HybridSearcher()
        hits = hybrid_searcher.search(
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

    def search_by_image(
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
        """上传图片进行搜索，collection_name 必须已指定。"""
        if not collection_name:
            raise ValueError("collection_name is required")

        # 保存上传的图片到临时路径
        tmp_dir = os.path.join(settings.UPLOAD_DIR, "query_images")
        os.makedirs(tmp_dir, exist_ok=True)

        ext = Path(image.filename).suffix or ".jpg"
        tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{ext}")

        with open(tmp_path, "wb") as f:
            f.write(image.file.read())

        hybrid_searcher = HybridSearcher()
        hits = hybrid_searcher.search(
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
        """格式化搜索结果。Milvus 直接返回完整字段，无需回查 MySQL。

        处理嵌套表格：
        - 嵌套表格命中：附带父表格信息（parent_chunk）
        - 父表格命中：附带嵌套表格引用（nested_refs）

        输出字段增强：
        - doc_name: 文档名称（从 metadata 提取）
        - heading_text: 章节标题（从 metadata 提取）
        - heading_level: 章节层级（从 metadata 提取）
        - image_url: 图片访问 URL（图片类型专用，从 metadata 提取）
        """
        from app.repositories.chunk_repo import chunk_repo

        result_hits = []
        for hit in hits:
            # 从 metadata 提取出处信息
            metadata = hit.get("metadata", {}) or {}
            doc_name = metadata.get("doc_name", "")
            heading_text = metadata.get("heading_text", "")
            heading_level = metadata.get("heading_level", 0)

            chunk_obj = {
                "chunk_id": hit["chunk_id"],
                "doc_id": hit["doc_id"],
                "doc_name": doc_name,  # 文档名称
                "chunk_type": hit["chunk_type"],
                "content": hit["content"],
                "caption": metadata.get("caption"),
                "page_num": hit.get("page_num", 0),
                "heading_text": heading_text,  # 章节标题
                "heading_level": heading_level,  # 章节层级
                "metadata": metadata,
                "doc_type": hit.get("doc_type", ""),
            }

            # 图片类型：提取 image_url 作为顶层字段，方便前端使用
            if hit["chunk_type"] == "image":
                # 优先使用 Milvus 顶层 image_path 字段，其次使用 metadata 中的
                image_path = hit.get("image_path") or metadata.get("image_path", "")
                url = metadata.get("url", "")
                image_url = url if url else image_path
                # 转换路径格式：./uploads/xxx → /uploads/xxx
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
                    # 获取嵌套表格简要信息
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

            # 添加附加信息
            if parent_chunk:
                result_hit["parent_chunk"] = parent_chunk
            if nested_tables:
                result_hit["nested_tables"] = nested_tables

            result_hits.append(result_hit)
        return result_hits


# 单例
search_service = SearchService()
