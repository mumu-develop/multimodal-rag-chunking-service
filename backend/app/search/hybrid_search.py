"""
混合搜索模块

提供多模态混合检索能力，整合：
- Milvus dense 向量检索
- BM25 文本匹配（通过 Milvus Function）
- 交叉编码器重排序（可选）

搜索模式：
- text: 纯文本查询 → encode_query + BM25 混合检索
- image: 纯图片查询 → encode_image_query + dense 检索
- both: 文本+图片 → encode_multimodal + dense 检索

FAQ 支持：
- FAQ检索：通过 data_type="faq" 过滤，仅检索FAQ问答
- 混合检索：同时检索 FAQ 和文档切片，FAQ加权融合
"""
import structlog

from app.config import settings
from app.embedders.reranker import reranker
from app.embedders.unified_embedder import unified_embedder
from app.repositories.milvus_repo import milvus_repo

logger = structlog.get_logger()


class HybridSearcher:
    """
    混合搜索器。

    整合向量检索和重排序，提供统一搜索入口。

    流程概述：
    1. 根据查询类型（文本/图片/图文）编码查询向量
    2. 构建 Milvus 过滤表达式（doc_id、metadata 过滤）
    3. 调用 milvus_repo.search 执行检索
    4. 可选：调用 reranker 进行交叉编码器重排序
    5. 返回 top_k 结果

    数据来源：
    - 搜索结果直接从 Milvus 返回（content、doc_id、chunk_type 等）
    - 不回查 MySQL（减少延迟）
    - 管理接口（/api/v1/chunks）才走 MySQL 获取完整信息
    """

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
        """
        执行混合搜索。

        业务场景：
        - 用户在搜索框输入文本查询
        - 用户上传图片进行图片检索
        - 用户同时输入文本和图片进行图文联合检索

        流程步骤：
        1. 验证 collection_name 参数（必填）
        2. 构建 Milvus 过滤表达式（_build_expr）
        3. 判断查询类型（是否有文本/图片）
        4. 根据查询类型编码向量：
           - 文本+图片 → encode_multimodal（多模态融合向量）
           - 仅图片 → encode_image_query（图片向量）
           - 仅文本 → encode_query（文本向量）
        5. 确定检索数量：
           - 启用重排序：retrieve_k = RERANK_TOP_N（如50）
           - 不重排序：retrieve_k = top_k * 2（多取一些候选）
        6. 调用 milvus_repo.search 执行检索：
           - 纯文本查询：dense + BM25 混合检索，RRF 融合
           - 图片/图文查询：纯 dense 检索
        7. 可选：重排序处理
           - 取前 RERANK_TOP_N 个候选
           - 调用 reranker.rerank 进行交叉编码器重排序
           - 返回 top_k 个最终结果
        8. 返回结果列表

        参数：
            query: 文本查询字符串（可选）
            query_image: 图片文件路径（可选，用于图片检索）
            top_k: 最终返回结果数量（默认10）
            modal: 查询模式，"text"/"image"/"both"（默认text）
            filters: 元数据过滤条件，如 {"category": "技术文档"}
            doc_ids: 限定搜索的文档 ID 列表（用于范围过滤）
            collection_name: Milvus Collection 名称（必填）
            use_rerank: 是否启用交叉编码器重排序（默认True）

        返回：
            list[dict]: 搜索结果列表，每个元素包含：
            - chunk_id: 切片唯一ID
            - doc_id: 文档ID
            - chunk_type: 切片类型（text/table/image/code）
            - doc_type: 文档类型（pdf/docx/pptx/md）
            - page_num: 页码
            - content: 内容文本
            - image_path: 图片路径（仅image切片）
            - metadata: 元数据字典
            - score: 相似度分数
            - modal: 切片类型（与chunk_type相同）
        """
        # 步骤1：验证必填参数
        # - collection_name 指定搜索的目标 Collection
        # - 每个知识库对应一个 Collection
        if not collection_name:
            raise ValueError("collection_name is required")

        # 步骤2：构建 Milvus 过滤表达式
        # - 将 doc_ids 和 filters 转换为 Milvus 查询语法
        # - 例如：doc_id in ["id1", "id2"] and metadata["category"] == "技术文档"
        expr = self._build_expr(filters, doc_ids)

        # 步骤3：判断查询类型
        # - is_image_query: 是否有图片输入
        # - has_text_query: 是否有文本输入
        is_image_query = bool(query_image and query_image.strip())
        has_text_query = bool(query and query.strip())

        # 步骤4：编码查询向量
        # - 根据查询类型选择不同的编码方法
        # - dense_vector: 2560维统一向量（BGE-M3 + OpenCLIP 融合）
        dense_vector = None
        if is_image_query and has_text_query:
            # 文本+图片：多模态融合向量
            # - encode_multimodal 同时处理文本和图片
            # - 文本用 BGE-M3，图片用 OpenCLIP，然后融合
            dense_vector = unified_embedder.encode_multimodal(query, query_image)
        elif is_image_query:
            # 仅图片：图片向量
            # - encode_image_query 用 OpenCLIP 编码图片
            # - 不包含 BM25（图片无文本内容）
            dense_vector = unified_embedder.encode_image_query(query_image)
        elif has_text_query:
            # 仅文本：文本向量
            # - encode_query 用 BGE-M3 编码文本
            # - Milvus 会同时做 dense + BM25 混合检索
            dense_vector = unified_embedder.encode_query(query)

        # 步骤5：确定检索数量
        # - 启用重排序：需要多取候选（RERANK_TOP_N，如50）
        # - 不重排序：取 top_k * 2（稍微多取一些以防不足）
        retrieve_k = settings.RERANK_TOP_N if use_rerank else top_k * 2

        # 步骤6：检查向量是否有效
        # - 无有效查询时返回空列表
        if dense_vector is None:
            logger.warning("no_query_vector", query=query, query_image=query_image)
            return []

        # 步骤7：执行 Milvus 检索
        # - vector: 查询向量
        # - top_k: 检索数量
        # - query_text: 传递给 Milvus 做 BM25 检索（仅纯文本查询）
        # - expr: 过滤表达式
        # - collection_name: 目标 Collection
        #
        # 注意：仅纯文本查询（has_text_query and not is_image_query）才传递 query_text
        # - 此时 Milvus 执行 dense + BM25 混合检索
        # - 图片/图文查询不传 query_text，仅执行 dense 检索
        # - data_type 过滤（faq/chunk/""全部）
        dtype_filter = data_type if data_type else ""
        results = milvus_repo.search(
            vector=dense_vector,
            top_k=retrieve_k,
            query_text=query if has_text_query and not is_image_query else None,
            expr=expr,
            collection_name=collection_name,
            data_type=dtype_filter,
        )

        # 步骤8：可选重排序
        # - 交叉编码器（Cross-Encoder）比双塔模型更精准
        # - 但计算成本高，仅对候选结果重排序
        if use_rerank and settings.RERANK_ENABLED and results:
            # 取前 RERANK_TOP_N 个候选进行重排序
            candidates = results[:settings.RERANK_TOP_N]
            # rerank 返回按重排序分数排序的结果
            reranked = reranker.rerank(query, candidates)
            # 返回 top_k 个最终结果
            return reranked[:top_k]

        # 步骤9：不重排序时直接返回
        return results[:top_k]

    def search_faq(
        self,
        query: str,
        top_k: int = 10,
        collection_name: str = None,
        use_rerank: bool = True,
        category: str = "",
    ) -> list[dict]:
        """
        FAQ 问答检索（仅检索 FAQ 数据）。

        业务场景：
        - 用户查询 FAQ 问答库
        - 仅对问题文本编码，检索相似问题

        流程步骤：
        1. 验证 collection_name 参数
        2. 编码问题向量（仅用文本编码器）
        3. 构造过滤表达式（data_type="faq" + category）
        4. 调用 Milvus 检索
        5. 可选重排序
        6. 返回结果（包含 question + answer）

        参数：
            query: 查询文本
            top_k: 返回数量
            collection_name: Collection 名称
            use_rerank: 是否启用重排序
            category: 分类过滤

        返回：
            list[dict]: FAQ 搜索结果，每个元素包含：
            - faq_id: FAQ ID (chunk_id)
            - question: 问题文本 (content)
            - answer: 答案 (summary)
            - category: 分类 (section)
            - keywords: 关键词 (tags)
            - hit_count: 命中次数 (priority)
            - score: 相似度分数
        """
        if not collection_name:
            raise ValueError("collection_name is required")

        if not query or not query.strip():
            return []

        # 编码问题向量
        dense_vector = unified_embedder.encode_query(query)

        # 构造过滤表达式
        expr = 'data_type == "faq"'
        if category:
            expr = f'{expr} and section == "{category}"'

        # 确定检索数量
        retrieve_k = settings.RERANK_TOP_N if use_rerank else top_k * 2

        # 执行 Milvus 检索（FAQ 也支持 BM25）
        results = milvus_repo.search(
            vector=dense_vector,
            top_k=retrieve_k,
            query_text=query,
            expr=expr,
            collection_name=collection_name,
            data_type="faq",  # 过滤 FAQ 类型
        )

        # 可选重排序
        if use_rerank and settings.RERANK_ENABLED and results:
            candidates = results[:settings.RERANK_TOP_N]
            reranked = reranker.rerank(query, candidates)
            return reranked[:top_k]

        return results[:top_k]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        collection_name: str = None,
        use_rerank: bool = True,
        faq_weight: float = 0.6,
        category: str = "",
    ) -> list[dict]:
        """
        混合检索（FAQ + 文档切片）。

        业务场景：
        - 知识库类型为 "hybrid" 时使用
        - 同时检索 FAQ 问答和文档切片
        - FAQ 结果加权融合

        流程步骤：
        1. 验证 collection_name 参数
        2. 并行执行：
           - FAQ 检索（data_type="faq"）
           - 文档检索（data_type="chunk"）
        3. 合并结果：
           - FAQ 结果加权（faq_weight）
           - 文档结果加权（1 - faq_weight）
        4. 可选重排序（合并后统一重排）
        5. 返回 top_k 结果

        参数：
            query: 查询文本
            top_k: 返回数量
            collection_name: Collection 名称
            use_rerank: 是否启用重排序
            faq_weight: FAQ 权重（0.0-1.0，默认 0.6）
            category: 分类过滤（仅对 FAQ）

        返回：
            list[dict]: 混合结果，包含 FAQ 和文档切片
        """
        if not collection_name:
            raise ValueError("collection_name is required")

        if not query or not query.strip():
            return []

        # 编码查询向量
        dense_vector = unified_embedder.encode_query(query)

        # 确定每路检索数量
        retrieve_k = settings.RERANK_TOP_N if use_rerank else top_k * 2

        # FAQ 过滤表达式
        faq_expr = 'data_type == "faq"'
        if category:
            faq_expr = f'{faq_expr} and section == "{category}"'

        # 文档过滤表达式
        chunk_expr = 'data_type == "chunk"'

        # 并行执行两路检索
        faq_results = milvus_repo.search(
            vector=dense_vector,
            top_k=retrieve_k // 2,
            query_text=query,
            expr=faq_expr,
            collection_name=collection_name,
            data_type="faq",
        )

        chunk_results = milvus_repo.search(
            vector=dense_vector,
            top_k=retrieve_k // 2,
            query_text=query,
            expr=chunk_expr,
            collection_name=collection_name,
            data_type="chunk",
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

        # 可选重排序
        if use_rerank and settings.RERANK_ENABLED and combined:
            candidates = combined[:settings.RERANK_TOP_N]
            reranked = reranker.rerank(query, candidates)
            return reranked[:top_k]

        return combined[:top_k]

    def _build_expr(self, filters: dict | None, doc_ids: list[str] | None) -> str:
        """
        构建 Milvus 过滤表达式（内部方法）。

        业务场景：
        - 用户选择特定文档进行搜索（doc_ids 过滤）
        - 用户按元数据筛选（如 category="技术文档"）

        流程步骤：
        1. 初始化空条件列表
        2. 处理 doc_ids 过滤：
           - 单个 doc_id：doc_id == "xxx"
           - 多个 doc_id：doc_id in ["xxx", "yyy"]
        3. 处理 metadata 过滤：
           - 字符串值：metadata["key"] == "value"
           - 列表值（多选）：metadata["key"] in ["v1", "v2"]
           - 数值：metadata["key"] == 123
        4. 用 "and" 连接所有条件
        5. 返回表达式字符串（空条件返回空字符串）

        参数：
            filters: 元数据过滤条件字典
            doc_ids: 文档ID列表

        返回：
            str: Milvus 过滤表达式，如：
            'doc_id in ["id1", "id2"] and metadata["category"] == "技术文档"'
        """
        # 步骤1：初始化条件列表
        conditions = []

        # 步骤2：处理 doc_ids 过滤
        if doc_ids:
            if len(doc_ids) == 1:
                # 单个 doc_id：使用 == 比较
                conditions.append(f'doc_id == "{doc_ids[0]}"')
            else:
                # 多个 doc_id：使用 in 列表
                # - 格式：doc_id in ["id1", "id2", "id3"]
                ids_str = ", ".join(f'"{d}"' for d in doc_ids)
                conditions.append(f"doc_id in [{ids_str}]")

        # 步骤3：处理 metadata 过滤
        # - metadata 是 JSON 字段，使用 metadata["key"] 语法访问
        if filters:
            for key, val in filters.items():
                if isinstance(val, str):
                    # 字符串值：metadata["key"] == "value"
                    conditions.append(f'metadata["{key}"] == "{val}"')
                elif isinstance(val, list):
                    # 列表值：metadata["key"] in ["v1", "v2"]
                    vals = ", ".join(f'"{v}"' for v in val)
                    conditions.append(f'metadata["{key}"] in [{vals}]')
                else:
                    # 数值/布尔：metadata["key"] == 123
                    conditions.append(f'metadata["{key}"] == {val}')

        # 步骤4：连接所有条件
        # - 使用 "and" 连接（所有条件必须同时满足）
        # - 空条件返回空字符串（不过滤）
        return " and ".join(conditions) if conditions else ""


# 单例模式：全局共享一个 HybridSearcher 实例
hybrid_searcher = HybridSearcher()