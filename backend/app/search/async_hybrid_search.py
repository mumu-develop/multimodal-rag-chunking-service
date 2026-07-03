"""
异步混合搜索模块

提供多模态异步混合检索能力，整合：
- Milvus dense 向量检索（异步包装）
- BM25 文本匹配（通过 Milvus Function）
- 交叉编码器重排序（异步 httpx）

搜索模式：
- text: 纯文本查询 → async_encode_query + BM25 混合检索
- image: 纯图片查询 → async_encode_image_query + dense 检索
- both: 文本+图片 → async_encode_multimodal + dense 检索

FAQ 支持：
- FAQ检索：通过 data_type="faq" 过滤
- 混合检索：同时检索 FAQ 和文档切片
"""
import asyncio
import structlog

from app.config import settings
from app.embedders.async_embedder import async_embedder
from app.embedders.async_reranker import async_reranker
from app.repositories.async_milvus_repo import async_milvus_repo

logger = structlog.get_logger()


class AsyncHybridSearcher:
    """
    异步混合搜索器。

    整合异步向量检索和异步重排序，提供统一异步搜索入口。

    流程概述：
    1. 根据查询类型异步编码查询向量
    2. 构建 Milvus 过滤表达式
    3. 异步调用 milvus_repo 执行检索
    4. 可选：异步重排序
    5. 返回 top_k 结果

    与同步版本的区别：
    - encode_query → async encode_query（await）
    - milvus_repo.search → async milvus_repo.search（await）
    - reranker.rerank → async reranker.rerank（await）
    """

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
        """
        执行异步混合搜索。

        流程步骤：
        1. 验证 collection_name 参数（必填）
        2. 构建 Milvus 过滤表达式
        3. 判断查询类型（是否有文本/图片）
        4. 异步编码向量：
           - 文本+图片 → await async_encode_multimodal
           - 仅图片 → await async_encode_image_query
           - 仅文本 → await async_encode_query
        5. 确定检索数量
        6. 异步执行 Milvus 检索
        7. 可选：异步重排序
        8. 返回结果列表

        参数：
            query: 文本查询字符串
            query_image: 图片文件路径
            top_k: 最终返回结果数量
            modal: 查询模式 "text"/"image"/"both"
            filters: 元数据过滤条件
            doc_ids: 限定搜索的文档 ID 列表
            collection_name: Milvus Collection 名称（必填）
            use_rerank: 是否启用重排序
            data_type: 数据类型过滤

        返回：
            list[dict]: 搜索结果列表
        """
        # 步骤1：验证必填参数
        if not collection_name:
            raise ValueError("collection_name is required")

        # 步骤2：构建过滤表达式
        expr = self._build_expr(filters, doc_ids)

        # 步骤3：判断查询类型
        is_image_query = bool(query_image and query_image.strip())
        has_text_query = bool(query and query.strip())

        # 步骤4：异步编码向量
        dense_vector = None
        if is_image_query and has_text_query:
            # 文本+图片：多模态融合向量
            dense_vector = await async_embedder.encode_multimodal(query, query_image)
        elif is_image_query:
            # 仅图片：图片向量
            dense_vector = await async_embedder.encode_image_query(query_image)
        elif has_text_query:
            # 仅文本：文本向量
            dense_vector = await async_embedder.encode_query(query)

        # 步骤5：确定检索数量
        retrieve_k = settings.RERANK_TOP_N if use_rerank else top_k * 2

        # 步骤6：检查向量有效性
        if dense_vector is None:
            logger.warning("async_no_query_vector", query=query, query_image=query_image)
            return []

        # 步骤7：异步执行 Milvus 检索
        dtype_filter = data_type if data_type else ""
        results = await async_milvus_repo.search(
            vector=dense_vector,
            top_k=retrieve_k,
            query_text=query if has_text_query and not is_image_query else None,
            expr=expr,
            collection_name=collection_name,
            data_type=dtype_filter,
        )

        # 步骤8：可选异步重排序
        if use_rerank and settings.RERANK_ENABLED and results:
            candidates = results[:settings.RERANK_TOP_N]
            reranked = await async_reranker.rerank(query, candidates)
            return reranked[:top_k]

        return results[:top_k]

    async def search_faq(
        self,
        query: str,
        top_k: int = 10,
        collection_name: str = None,
        use_rerank: bool = True,
        category: str = "",
    ) -> list[dict]:
        """
        异步 FAQ 问答检索。

        流程步骤：
        1. 验证 collection_name 参数
        2. 异步编码问题向量
        3. 构造过滤表达式
        4. 异步执行 Milvus 检索
        5. 可选异步重排序
        6. 返回结果

        参数：
            query: 查询文本
            top_k: 返回数量
            collection_name: Collection 名称
            use_rerank: 是否启用重排序
            category: 分类过滤

        返回：
            list[dict]: FAQ 搜索结果
        """
        if not collection_name:
            raise ValueError("collection_name is required")

        if not query or not query.strip():
            return []

        # 异步编码问题向量
        dense_vector = await async_embedder.encode_query(query)

        # 构造过滤表达式
        expr = 'data_type == "faq"'
        if category:
            expr = f'{expr} and section == "{category}"'

        # 确定检索数量
        retrieve_k = settings.RERANK_TOP_N if use_rerank else top_k * 2

        # 异步执行 Milvus 检索
        results = await async_milvus_repo.search(
            vector=dense_vector,
            top_k=retrieve_k,
            query_text=query,
            expr=expr,
            collection_name=collection_name,
            data_type="faq",
        )

        # 可选异步重排序
        if use_rerank and settings.RERANK_ENABLED and results:
            candidates = results[:settings.RERANK_TOP_N]
            reranked = await async_reranker.rerank(query, candidates)
            return reranked[:top_k]

        return results[:top_k]

    async def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        collection_name: str = None,
        use_rerank: bool = True,
        faq_weight: float = 0.6,
        category: str = "",
    ) -> list[dict]:
        """
        异步混合检索（FAQ + 文档切片）。

        流程步骤：
        1. 验证 collection_name 参数
        2. 异步编码查询向量
        3. 异步并行执行：
           - FAQ 检索
           - 文档检索
        4. 合并结果（加权）
        5. 可选异步重排序
        6. 返回结果

        参数：
            query: 查询文本
            top_k: 返回数量
            collection_name: Collection 名称
            use_rerank: 是否启用重排序
            faq_weight: FAQ 权重
            category: 分类过滤

        返回：
            list[dict]: 混合结果
        """
        if not collection_name:
            raise ValueError("collection_name is required")

        if not query or not query.strip():
            return []

        # 异步编码查询向量
        dense_vector = await async_embedder.encode_query(query)

        # 确定每路检索数量
        retrieve_k = settings.RERANK_TOP_N if use_rerank else top_k * 2

        # FAQ 过滤表达式
        faq_expr = 'data_type == "faq"'
        if category:
            faq_expr = f'{faq_expr} and section == "{category}"'

        # 文档过滤表达式
        chunk_expr = 'data_type == "chunk"'

        # 异步并行执行两路检索
        faq_results, chunk_results = await asyncio.gather(
            async_milvus_repo.search(
                vector=dense_vector,
                top_k=retrieve_k // 2,
                query_text=query,
                expr=faq_expr,
                collection_name=collection_name,
                data_type="faq",
            ),
            async_milvus_repo.search(
                vector=dense_vector,
                top_k=retrieve_k // 2,
                query_text=query,
                expr=chunk_expr,
                collection_name=collection_name,
                data_type="chunk",
            ),
        )

        # 合并结果（加权）
        combined = []
        for hit in faq_results:
            hit["score"] = hit["score"] * faq_weight
            combined.append(hit)
        for hit in chunk_results:
            hit["score"] = hit["score"] * (1 - faq_weight)
            combined.append(hit)

        # 按加权分数排序
        combined.sort(key=lambda x: x["score"], reverse=True)

        # 可选异步重排序
        if use_rerank and settings.RERANK_ENABLED and combined:
            candidates = combined[:settings.RERANK_TOP_N]
            reranked = await async_reranker.rerank(query, candidates)
            return reranked[:top_k]

        return combined[:top_k]

    def _build_expr(self, filters: dict | None, doc_ids: list[str] | None) -> str:
        """
        构建 Milvus 过滤表达式。

        参数：
            filters: 元数据过滤条件字典
            doc_ids: 文档ID列表

        返回：
            str: Milvus 过滤表达式
        """
        conditions = []

        # 处理 doc_ids 过滤
        if doc_ids:
            if len(doc_ids) == 1:
                conditions.append(f'doc_id == "{doc_ids[0]}"')
            else:
                ids_str = ", ".join(f'"{d}"' for d in doc_ids)
                conditions.append(f"doc_id in [{ids_str}]")

        # 处理 metadata 过滤
        if filters:
            for key, val in filters.items():
                if isinstance(val, str):
                    conditions.append(f'metadata["{key}"] == "{val}"')
                elif isinstance(val, list):
                    vals = ", ".join(f'"{v}"' for v in val)
                    conditions.append(f'metadata["{key}"] in [{vals}]')
                else:
                    conditions.append(f'metadata["{key}"] == {val}')

        return " and ".join(conditions) if conditions else ""


# 单例模式
async_hybrid_searcher = AsyncHybridSearcher()