"""
Milvus 向量数据库 Repository

提供向量数据库的访问层，负责：
- Collection 创建与管理
- 向量数据的写入（upsert）、删除
- 向量检索（dense + BM25 混合搜索）

蚂蚁生产级统一 Schema（兼容文档切片 + FAQ）：
- chunk_id: 主键（文档切片为 chunk_id，FAQ 为 faq_id）
- doc_id: 文档ID（文档切片）/ FAQ ID（FAQ）
- data_type: 数据类型（"chunk" 文档切片 / "faq" FAQ问答）
- chunk_type: 切片类型（text/table/image/code）
- doc_type: 文档类型（pdf/docx/pptx/md）/ 来源（manual/import）
- content: 文本内容（文档切片原文 / FAQ问题文本）
- title: 标题（文档切片标题 / FAQ问题冗余）
- summary: 摘要（文档切片摘要 / FAQ答案）
- page_num: 页码（文档切片）/ 0（FAQ）
- section: 章节（文档切片章节 / FAQ分类）
- tags: 关键词标签
- priority: 优先级/命中次数
- image_path: 图片路径（仅image切片有效）
- metadata: JSON 元数据
- sparse_vector: BM25稀疏向量（由 Function 自动生成）
- vector: 统一dense向量（2560维）

搜索策略：
- 纯文本查询：dense向量 + BM25混合检索，RRF融合
- 图片/图文查询：纯dense向量检索
- FAQ检索：可通过 data_type="faq" 过滤
"""
import os
# Milvus Lite 使用 Unix socket，macOS 限制路径 ≤103 字符
# 强制使用 /tmp 作为临时目录，避免项目路径过长导致失败
os.environ["TMPDIR"] = "/tmp"

import structlog
from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType, AnnSearchRequest, RRFRanker
from pymilvus.milvus_client.index import IndexParams

from app.config import settings

logger = structlog.get_logger()


class MilvusRepository:
    """
    Milvus 向量数据库 Repository。

    基于 pymilvus 的 MilvusClient API，兼容 Milvus Lite (.db) 和 Milvus Server。
    采用惰性初始化模式，首次使用时才建立连接。

    使用场景：
    - 文档切片向量化后写入 Milvus
    - 用户查询时的向量检索
    - 知识库管理（创建/删除 Collection）
    """

    def __init__(self):
        """
        初始化 Repository。

        步骤：
        1. 设置客户端连接为 None（惰性初始化）
        2. 初始化 last_error 属性
        """
        # 步骤1：惰性初始化标记
        # - _client 在首次调用 _get_client() 时才创建
        # - 避免启动时就建立连接（可能失败）
        self._client = None
        self.last_error = ""

    def _get_client(self):
        """
        惰性初始化 MilvusClient（内部方法）。

        流程步骤：
        1. 检查客户端是否已初始化，已初始化则直接返回
        2. 确定连接 URI：
           - 优先使用本地 db 文件（Milvus Lite）
           - 否则使用 HTTP 连接 Milvus Server
        3. 调用 _create_with_retry 创建客户端（处理并发锁问题）
        4. 记录初始化日志
        5. 返回客户端实例

        返回：
            pymilvus.MilvusClient 实例
        """
        # 步骤1：检查是否已初始化
        if self._client is not None:
            return self._client

        # 步骤2：导入 MilvusClient 类
        from pymilvus import MilvusClient

        # 步骤3：确定连接 URI
        # - MILVUS_LOCAL_DB: 本地 db 文件路径（如 ./milvus_lite.db）
        # - Milvus Lite 使用文件作为存储，无需服务器
        if settings.MILVUS_LOCAL_DB:
            uri = settings.MILVUS_LOCAL_DB
        else:
            # HTTP 连接 Milvus Server
            # - 格式：http://{host}:{port}
            uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"

        # 步骤4：创建客户端（带重试机制）
        # - Milvus Lite 不支持多进程并发访问
        # - Celery worker 和 FastAPI 可能同时访问同一 db 文件
        # - _create_with_retry 处理 "opened by another" 错误
        self._client = self._create_with_retry(MilvusClient, uri)

        # 步骤5：记录初始化日志
        logger.info("milvus_client_init", uri=uri)

        # 步骤6：返回客户端
        return self._client

    def _create_with_retry(self, client_cls, uri: str, max_retries: int = 10, interval: float = 1.0):
        """
        创建 Milvus 客户端（带重试机制）。

        业务场景：
        - Milvus Lite 使用 Unix socket 通信
        - macOS socket 路径限制 ≤103 字符
        - 多进程同时访问同一 db 文件会导致锁冲突
        - FastAPI 持有锁时，Celery worker 需等待

        流程步骤：
        1. 循环尝试创建客户端（最多 max_retries 次）
        2. 每次尝试调用 client_cls(uri=uri)
        3. 如果出现 "opened by another" 或 "failed to start" 错误：
           - 记录警告日志
           - 等待 interval 秒后重试
        4. 其他错误直接抛出（非锁冲突）
        5. 重试耗尽后抛出最后一次错误

        参数：
            client_cls: MilvusClient 类
            uri: 连接地址（文件路径或 HTTP URL）
            max_retries: 最大重试次数（默认10）
            interval: 重试间隔秒数（默认1.0）

        返回：
            MilvusClient 实例

        抛出：
            最后一次异常（重试耗尽）
        """
        from pymilvus.exceptions import MilvusException

        last_err = None
        # 步骤1：循环尝试
        for attempt in range(max_retries):
            try:
                # 步骤2：尝试创建客户端
                return client_cls(uri=uri)
            except (MilvusException, Exception) as e:
                err_msg = str(e)
                # 步骤3：检查是否为锁冲突错误
                # - "opened by another": 其他进程持有锁
                # - "failed to start": 启动失败
                if "opened by another" not in err_msg and "failed to start" not in err_msg:
                    # 步骤4：非锁冲突错误，直接抛出
                    raise
                # 步骤3续：记录警告并等待重试
                last_err = e
                logger.warning(
                    "milvus_lite_busy_retry",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e)[:100],
                )
                import time
                time.sleep(interval)

        # 步骤5：重试耗尽，抛出最后一次错误
        raise last_err

    def connect(self, collection_name: str):
        """
        获取或创建指定 Collection（启动预热调用）。

        业务场景：
        - FastAPI 启动时调用，确保默认 Collection 存在
        - 验证 schema 版本一致性

        流程步骤：
        1. 获取 Milvus 客户端
        2. 检查 Collection 是否存在
        3. 如果不存在：创建新 Collection
        4. 如果存在：检查 schema 是否匹配
           - 不匹配：删除旧 Collection，重新创建
           - 匹配：复用现有 Collection
        5. 返回 Collection 名称

        参数：
            collection_name: Collection 名称（必填）

        返回：
            Collection 名称（字符串）
        """
        # 步骤1：获取客户端
        client = self._get_client()
        name = collection_name

        # 步骤2：检查 Collection 是否存在
        if name not in client.list_collections():
            # 步骤3：不存在则创建
            self._create_collection(name)
            return name
        else:
            # 步骤4：存在则检查 schema
            if not self._schema_matches(client, name):
                # schema 不匹配：删除重建
                logger.warning("milvus_schema_mismatch_recreating", name=name)
                client.drop_collection(name)
                self._create_collection(name)
            else:
                # schema 匹配：复用
                logger.info("milvus_collection_reused", name=name)

        # 步骤5：返回 Collection 名称
        return name

    def _schema_matches(self, client, name: str) -> bool:
        """
        检查 Collection schema 是否与预期一致（内部方法）。

        业务场景：
        - 项目迭代过程中 schema 可能变化
        - 需要检测旧版 schema 并重建

        流程步骤：
        1. 获取 Collection 描述信息
        2. 解析字段列表为字典（按字段名索引）
        3. 检查必需字段是否存在：vector, content, data_type
        4. 检查 vector 维度是否匹配 UNIFIED_VECTOR_DIM
        5. 检查旧版字段是否不存在：text_vector, image_vector（旧版双向量）
        6. 检查 kb_id 字段不存在（已废弃）
        7. 全部通过返回 True，否则返回 False

        参数：
            client: MilvusClient 实例
            name: Collection 名称

        返回：
            bool: schema 匹配返回 True
        """
        try:
            # 步骤1：获取 Collection 描述
            # - describe_collection 返回 schema 信息
            desc = client.describe_collection(name)
            # 步骤2：解析字段为字典
            fields = {f["name"]: f for f in desc.get("fields", [])}

            # 步骤3：检查必需字段
            if "vector" not in fields:
                return False
            if "content" not in fields:
                return False
            # 检查新版统一 Schema 字段
            if "data_type" not in fields:
                return False

            # 步骤4：检查 vector 维度
            # - 从 params.dim 获取维度值
            # - 必须与 settings.UNIFIED_VECTOR_DIM 一致
            v_dim = fields["vector"].get("params", {}).get("dim", 0)
            if v_dim != settings.UNIFIED_VECTOR_DIM:
                return False

            # 步骤5：检查旧版字段不存在
            # - 旧版使用 text_vector + image_vector 双向量
            # - 新版使用统一 vector
            if "text_vector" in fields or "image_vector" in fields:
                return False

            # 步骤6：检查 kb_id 不存在
            # - kb_id 已废弃，改用多 Collection 方案
            if "kb_id" in fields:
                return False

            # 步骤7：全部通过
            return True
        except Exception:
            # 异常时返回 False，触发重建
            return False

    def _create_collection(self, name: str) -> None:
        """
        创建 Collection（内部方法）。

        参数：
            name: Collection 名称（只能包含字母、数字、下划线）

        抛出：
            ValueError: 如果名称格式不符合 Milvus 规范
        """
        # Milvus collection 名称格式检查
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', name):
            raise ValueError(
                f"Collection 名称格式错误: '{name}'。"
                "Milvus collection 名称只能包含字母、数字、下划线(_)，不能包含中文或特殊字符如 -"
            )

        # 步骤1：获取客户端和向量维度
        client = self._get_client()
        dim = settings.UNIFIED_VECTOR_DIM

        # 步骤2：删除已存在的 Collection
        if client.has_collection(name):
            client.drop_collection(name)

        # 步骤3：定义统一 Schema 字段
        # - FieldSchema 指定字段名、类型、约束
        # - enable_analyzer=True 启用分词（BM25 需要）
        # - analyzer_params={"type": "chinese"} 使用中文分词器
        #
        # 字段用途对照：
        # | 字段 | 文档切片用途 | FAQ用途 |
        # |------|-------------|---------|
        # | chunk_id | chunk_id | faq_id |
        # | doc_id | doc_id | faq_id |
        # | data_type | "chunk" | "faq" |
        # | content | 切片内容 | 问题文本 |
        # | title | 标题 | 问题（冗余） |
        # | summary | 摘要 | 答案 |
        # | section | 章节 | 分类 |
        # | tags | 关键词 | 标签 |
        # | priority | - | hit_count |
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="data_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000,
                        enable_analyzer=True,
                        analyzer_params={"type": "chinese"}),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="priority", dtype=DataType.INT64),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]

        # 步骤4：定义 BM25 Function
        # - FunctionType.BM25: BM25 算法
        # - input_field_names: ["content"] 输入字段
        # - output_field_names: ["sparse_vector"] 输出字段
        # - 插入数据时自动生成 sparse_vector
        bm25_function = Function(
            name="bm25_func",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
        )

        # 步骤5：创建 Schema
        # - CollectionSchema 整合字段和 Function
        schema = CollectionSchema(
            fields,
            description="Unified RAG Schema: chunk + FAQ (dense + BM25)",
            functions=[bm25_function],
        )

        # 步骤6：创建 Collection
        client.create_collection(collection_name=name, schema=schema)

        # 步骤7：创建索引
        # - IndexParams 指定索引类型和度量类型
        # - vector: AUTOINDEX（自动选择），IP（内积，适合归一化向量）
        # - sparse_vector: SPARSE_INVERTED_INDEX，BM25 度量
        index_params = IndexParams()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        # 步骤8：创建索引并加载到内存
        client.create_index(collection_name=name, index_params=index_params)
        client.load_collection(collection_name=name)

        # 步骤9：记录日志
        logger.info("milvus_collection_created", name=name, vector_dim=dim, schema="unified")

    def batch_upsert(self, entities: list[dict], collection_name: str = None) -> bool:
        """
        批量写入向量数据（upsert 模式）。

        业务场景：
        - Celery 任务完成切片向量化后调用
        - upsert：存在则更新，不存在则插入

        流程步骤：
        1. 检查 entities 是否为空，空则直接返回 True
        2. 获取客户端和 Collection 名称
        3. 检查 Collection 是否存在，不存在抛出异常
        4. 按 batch_size 分批处理（防止单次请求过大）
        5. 每批调用 client.upsert 写入
        6. 记录完成日志
        7. 返回 True（异常返回 False）

        参数：
            entities: 向量数据列表，每个元素为 dict
                      必须包含所有 Schema 字段
            collection_name: Collection 名称（可选，默认使用配置）

        返回：
            bool: 成功返回 True，失败返回 False

        数据格式示例：
            {
                "chunk_id": "doc123_chunk0",
                "doc_id": "doc123",
                "chunk_type": "text",
                "doc_type": "pdf",
                "page_num": 1,
                "content": "这是切片内容...",
                "image_path": "",
                "metadata": {"heading": "第一章"},
                "vector": [0.1, 0.2, ...],  # 2560维
            }
        """
        # 步骤1：空数据直接返回
        if not entities:
            return True

        # 步骤2：验证 collection_name 参数
        if not collection_name:
            raise ValueError("collection_name is required")

        client = self._get_client()
        name = collection_name

        # 步骤3：检查 Collection 存在
        if name not in client.list_collections():
            logger.error("milvus_collection_not_found", name=name)
            raise ValueError(f"Collection '{name}' does not exist")

        # 步骤4：获取批次大小
        batch_size = settings.MILVUS_BATCH_SIZE

        try:
            # 步骤5：分批写入
            # - range(0, len, batch_size) 生成批次起始索引
            # - entities[i:i+batch_size] 切片获取当前批次
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                # 调用 upsert 写入
                client.upsert(collection_name=name, data=batch)

            # 步骤6：记录日志
            logger.info("milvus_batch_upsert_done", collection=name, count=len(entities))

            # 步骤7：返回成功
            return True
        except Exception:
            # 异常：记录日志并返回 False
            logger.exception("milvus_batch_upsert_failed", collection=name, count=len(entities))
            return False

    def upsert_faq(self, faq_data: dict, collection_name: str = None) -> bool:
        """
        写入 FAQ 向量数据（单条）。

        业务场景：
        - 创建 FAQ 问答对后，编码问题文本并写入 Milvus

        数据格式：
        {
            "faq_id": "abc123",
            "question": "如何上传文档？",
            "answer": "点击右上角上传按钮...",
            "category": "使用指南",
            "keywords": "上传,文档",
            "hit_count": 0,
            "vector": [0.1, 0.2, ...],
        }

        返回：
            bool: 成功返回 True
        """
        if not collection_name:
            raise ValueError("collection_name is required")

        client = self._get_client()
        name = collection_name

        if name not in client.list_collections():
            logger.error("milvus_collection_not_found", name=name)
            raise ValueError(f"Collection '{name}' does not exist")

        # 构造统一 Schema 格式的数据
        entity = {
            "chunk_id": faq_data["faq_id"],  # FAQ 用 faq_id 作为主键
            "doc_id": faq_data["faq_id"],    # FAQ 用 faq_id 作为 doc_id
            "data_type": "faq",              # 标记为 FAQ 类型
            "chunk_type": "faq",             # chunk_type 也标记为 faq
            "doc_type": faq_data.get("source", "manual"),  # manual/import
            "content": faq_data["question"],  # 问题文本（用于 BM25）
            "title": faq_data["question"],    # 问题文本（冗余存储）
            "summary": faq_data["answer"],    # 答案存储在 summary
            "page_num": 0,                    # FAQ 无页码
            "section": faq_data.get("category", ""),  # 分类
            "tags": faq_data.get("keywords", ""),     # 关键词标签
            "priority": faq_data.get("hit_count", 0), # 命中次数
            "image_path": "",                 # FAQ 无图片
            "metadata": {
                "kb_id": faq_data.get("kb_id", ""),
                "source": faq_data.get("source", "manual"),
            },
            "vector": faq_data["vector"],    # 问题向量
        }

        try:
            client.upsert(collection_name=name, data=[entity])
            logger.info("milvus_faq_upserted", collection=name, faq_id=faq_data["faq_id"])
            return True
        except Exception:
            logger.exception("milvus_faq_upsert_failed", collection=name, faq_id=faq_data["faq_id"])
            return False

    def batch_upsert_faq(self, faq_items: list[dict], collection_name: str = None) -> bool:
        """
        批量写入 FAQ 向量数据。

        参数：
            faq_items: FAQ 数据列表，每个元素包含 faq_id, question, answer, vector 等
            collection_name: Collection 名称

        返回：
            bool: 成功返回 True
        """
        if not faq_items:
            return True

        if not collection_name:
            raise ValueError("collection_name is required")

        client = self._get_client()
        name = collection_name

        if name not in client.list_collections():
            logger.error("milvus_collection_not_found", name=name)
            raise ValueError(f"Collection '{name}' does not exist")

        # 转换为统一 Schema 格式
        entities = []
        for faq in faq_items:
            entities.append({
                "chunk_id": faq["faq_id"],
                "doc_id": faq["faq_id"],
                "data_type": "faq",
                "chunk_type": "faq",
                "doc_type": faq.get("source", "manual"),
                "content": faq["question"],
                "title": faq["question"],
                "summary": faq["answer"],
                "page_num": 0,
                "section": faq.get("category", ""),
                "tags": faq.get("keywords", ""),
                "priority": faq.get("hit_count", 0),
                "image_path": "",
                "metadata": {
                    "kb_id": faq.get("kb_id", ""),
                    "source": faq.get("source", "manual"),
                },
                "vector": faq["vector"],
            })

        batch_size = settings.MILVUS_BATCH_SIZE

        try:
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                client.upsert(collection_name=name, data=batch)
            logger.info("milvus_faq_batch_upserted", collection=name, count=len(entities))
            return True
        except Exception:
            logger.exception("milvus_faq_batch_upsert_failed", collection=name, count=len(entities))
            return False

    def delete_faq(self, faq_ids: list[str], collection_name: str = None) -> bool:
        """
        删除 FAQ 向量数据。

        参数：
            faq_ids: FAQ ID 列表
            collection_name: Collection 名称

        返回：
            bool: 成功返回 True
        """
        if not faq_ids:
            return True

        if not collection_name:
            raise ValueError("collection_name is required")

        # FAQ 的 chunk_id 就是 faq_id，直接调用 delete
        return self.delete(faq_ids, collection_name)

    # ============================================================
    # 搜索接口
    # ============================================================

    def search(
        self,
        vector: list[float],
        top_k: int,
        query_text: str | None = None,
        expr: str = "",
        output_fields: list[str] | None = None,
        collection_name: str = None,
        data_type: str = "",  # 过滤数据类型: "chunk" / "faq" / "" (全部)
    ) -> list[dict]:
        """
        统一搜索入口。

        业务场景：
        - 用户查询接口 /api/v1/search 调用
        - 根据 query_text 是否存在决定搜索模式
        - 可通过 data_type 过滤只检索文档切片或 FAQ

        流程步骤：
        1. 确定 Collection 名称
        2. 获取客户端并检查 Collection 存在
        3. 构造 data_type 过滤表达式（如有）
        4. 判断搜索模式：
           - 有 query_text → 混合检索（dense + BM25）
           - 无 query_text → 纯 dense 检索
        5. 调用对应的搜索方法
        6. 返回结果列表

        参数：
            vector: 查询向量（2560维）
            top_k: 返回结果数量
            query_text: 查询文本（可选，用于 BM25）
            expr: 过滤表达式（如 doc_id=="xxx"）
            output_fields: 返回字段列表
            collection_name: Collection 名称（必填）
            data_type: 数据类型过滤（"chunk"/"faq"/""全部）

        返回：
            list[dict]: 搜索结果，每个元素包含：
            - chunk_id, doc_id, data_type, chunk_type, doc_type
            - content, title, summary, page_num, section, tags, priority
            - image_path, metadata
            - score: 相似度分数
            - modal: 切片类型（同 chunk_type）
        """
        # 步骤1：验证必填参数
        if not collection_name:
            raise ValueError("collection_name is required")

        name = collection_name

        # 步骤2：检查 Collection 存在
        client = self._get_client()
        if name not in client.list_collections():
            logger.error("milvus_collection_not_found", name=name)
            raise ValueError(f"Collection '{name}' does not exist")

        # 步骤3：构造 data_type 过滤表达式
        combined_expr = expr
        if data_type:
            dtype_filter = f'data_type == "{data_type}"'
            if combined_expr:
                combined_expr = f"{combined_expr} and {dtype_filter}"
            else:
                combined_expr = dtype_filter

        # 步骤4-5：判断搜索模式并调用
        if query_text and query_text.strip():
            # 有 query_text：混合检索
            return self._hybrid_search(vector, query_text, top_k, combined_expr, output_fields, name)
        # 无 query_text：纯 dense 检索
        return self._dense_search(vector, top_k, combined_expr, output_fields, name)

    def _dense_search(self, vector, top_k, expr, output_fields, collection_name: str = None):
        """
        纯 dense 向量检索（内部方法）。

        业务场景：
        - 图片查询：用户上传图片检索相似内容
        - 图文查询：用户同时输入文本和图片
        - 无 BM25 参与，纯向量相似度

        流程步骤：
        1. 获取客户端和 Collection 名称
        2. 设置默认输出字段（如未指定）
        3. 调用 client.search 执行向量检索
        4. 解析结果返回

        参数：
            vector: 查询向量
            top_k: 返回数量
            expr: 过滤表达式
            output_fields: 输出字段
            collection_name: Collection 名称

        返回：
            list[dict]: 检索结果
        """
        # 步骤1：获取客户端（collection_name 已在外层验证）
        client = self._get_client()
        name = collection_name

        # 步骤2：设置默认输出字段
        if output_fields is None:
            output_fields = self._default_output_fields()

        # 步骤3：执行向量检索
        # - data: 查询向量列表（支持批量，这里单条）
        # - anns_field: 检索的字段名
        # - search_params: IP 度量，ef=max(64, top_k)（HNSW 要求 ef >= k）
        # - limit: 返回数量
        # - filter: 过滤表达式
        # - output_fields: 返回字段
        ef = max(64, top_k)  # 确保 ef >= k
        res = client.search(
            collection_name=name,
            data=[vector],
            anns_field="vector",
            search_params={"metric_type": "IP", "params": {"ef": ef}},
            limit=top_k,
            filter=expr if expr else None,
            output_fields=output_fields,
        )

        # 步骤4：解析结果
        return self._parse_results(res)

    def _hybrid_search(self, vector, query_text, top_k, expr, output_fields, collection_name: str = None):
        """
        dense + BM25 混合检索（内部方法）。

        业务场景：
        - 纯文本查询：用户输入文本检索
        - 同时利用语义向量（dense）和关键词匹配（BM25）
        - 使用 RRF（Reciprocal Rank Fusion）融合两种结果

        流程步骤：
        1. 获取客户端和 Collection 名称
        2. 设置默认输出字段
        3. 构造 dense 向量检索请求（AnnSearchRequest）
        4. 构造 BM25 文本检索请求（AnnSearchRequest）
        5. 调用 client.hybrid_search 执行混合检索
        6. 使用 RRFRanker 融合结果
        7. 解析结果返回

        参数：
            vector: 查询向量
            query_text: 查询文本
            top_k: 最终返回数量
            expr: 过滤表达式
            output_fields: 输出字段
            collection_name: Collection 名称

        返回：
            list[dict]: 融合后的检索结果
        """
        # 步骤1：获取客户端（collection_name 已在外层验证）
        client = self._get_client()
        name = collection_name

        # 步骤2：设置默认输出字段
        if output_fields is None:
            output_fields = self._default_output_fields()

        # 步骤3：构造 dense 向量请求
        # - AnnSearchRequest: 单路检索请求
        # - data: 查询向量
        # - anns_field: 检索字段
        # - param: IP 度量，ef=max(64, limit)（HNSW 要求 ef >= k）
        # - limit: 每路检索数量（top_k * 2，给融合留余量）
        # - expr: 过滤表达式
        dense_limit = top_k * 2
        dense_ef = max(64, dense_limit)  # 确保 ef >= limit
        dense_req = AnnSearchRequest(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"ef": dense_ef}},
            limit=dense_limit,
            expr=expr,
        )

        # 步骤4：构造 BM25 文本请求
        # - data: 查询文本（Milvus 内部会分词）
        # - anns_field: sparse_vector 字段
        # - param: BM25 度量
        # - limit: top_k * 2
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=top_k * 2,
            expr=expr,
        )

        # 步骤5-6：执行混合检索并融合
        # - reqs: 多路检索请求列表
        # - ranker: RRFRanker（倒数排名融合）
        # - limit: 最终返回数量
        res = client.hybrid_search(
            collection_name=name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=top_k,
            output_fields=output_fields,
        )

        # 步骤7：解析结果
        return self._parse_results(res)

    # ============================================================
    # 内部工具
    # ============================================================

    def _default_output_fields(self):
        """
        获取默认输出字段列表（内部方法）。

        返回：
            list[str]: 默认返回的字段名（统一 Schema）
        """
        return [
            "chunk_id", "doc_id", "data_type", "chunk_type", "doc_type",
            "content", "title", "summary", "page_num", "section",
            "tags", "priority", "image_path", "metadata"
        ]

    def _parse_results(self, res):
        """
        解析 Milvus 搜索结果（内部方法）。

        流程步骤：
        1. 遍历结果批次（res 是 list of hits）
        2. 遍历每个 hit
        3. 从 hit.entity 提取字段值
        4. 构造结果 dict，添加 score 和 modal
        5. 返回结果列表

        参数：
            res: Milvus 搜索返回的原始结果

        返回：
            list[dict]: 解析后的结果列表（统一 Schema）
        """
        results = []
        # 步骤1：遍历批次
        for hits in res:
            # 步骤2：遍历每个 hit
            for hit in hits:
                # 步骤3：获取 entity（字段值）
                entity = hit.entity
                # 步骤4：构造结果 dict（统一 Schema）
                results.append({
                    "chunk_id": hit.id,          # 主键
                    "doc_id": entity.get("doc_id", ""),
                    "data_type": entity.get("data_type", "chunk"),  # chunk/faq
                    "chunk_type": entity.get("chunk_type", ""),
                    "doc_type": entity.get("doc_type", ""),
                    "content": entity.get("content", ""),
                    "title": entity.get("title", ""),
                    "summary": entity.get("summary", ""),  # FAQ答案
                    "page_num": entity.get("page_num", 0),
                    "section": entity.get("section", ""),  # FAQ分类
                    "tags": entity.get("tags", ""),
                    "priority": entity.get("priority", 0),
                    "image_path": entity.get("image_path", ""),
                    "metadata": entity.get("metadata"),
                    "score": hit.score,          # 相似度分数
                    "modal": entity.get("chunk_type", "text"),  # 兼容字段
                })
        # 步骤5：返回结果
        return results

    # ============================================================
    # Collection 管理
    # ============================================================

    def create_collection_if_not_exists(self, collection_name: str) -> bool:
        """
        创建指定名称的 Collection（如不存在）。

        参数：
            collection_name: Collection 名称（只能包含字母、数字、下划线）

        返回：
            bool: 成功返回 True
        """
        self.last_error = ""

        # 步骤1：获取客户端
        client = self._get_client()

        # 步骤2-3：检查是否存在
        if collection_name in client.list_collections():
            # 检查 schema 是否有效（是否包含 data_type 字段）
            if not self._check_schema_valid(collection_name):
                logger.warning("milvus_schema_invalid_recreate", name=collection_name)
                # 重建 collection
                try:
                    self._create_collection(collection_name)
                    logger.info("milvus_collection_recreated", name=collection_name)
                    return True
                except Exception as e:
                    self.last_error = str(e)
                    logger.exception("milvus_collection_recreate_failed", name=collection_name)
                    return False
            logger.info("milvus_collection_exists", name=collection_name)
            return True

        # 步骤4：创建 Collection
        try:
            self._create_collection(collection_name)
            logger.info("milvus_collection_created_for_kb", name=collection_name)
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.exception("milvus_collection_create_failed", name=collection_name)
            return False

    def drop_collection_by_name(self, collection_name: str) -> bool:
        """
        删除指定名称的 Collection。

        业务场景：
        - 删除知识库时，同步删除对应 Collection
        - 知识库管理接口调用

        流程步骤：
        1. 获取客户端
        2. 检查 Collection 是否存在
        3. 不存在则返回 True（无需删除）
        4. 存在则调用 drop_collection 删除
        5. 返回删除结果

        参数：
            collection_name: Collection 名称

        返回：
            bool: 成功返回 True
        """
        # 步骤1：获取客户端
        client = self._get_client()

        # 步骤2-3：检查是否存在
        if collection_name not in client.list_collections():
            logger.info("milvus_collection_not_exists", name=collection_name)
            return True

        # 步骤4：删除 Collection
        try:
            client.drop_collection(collection_name)
            logger.info("milvus_collection_dropped", name=collection_name)
            return True
        except Exception:
            logger.exception("milvus_collection_drop_failed", name=collection_name)
            return False

    # ============================================================
    # 数据操作
    # ============================================================

    def delete(self, ids: list[str], collection_name: str = None) -> bool:
        """
        按 chunk_id 删除向量数据。

        业务场景：
        - 删除单个切片时调用
        - 删除文档时批量删除所有切片

        流程步骤：
        1. 检查 ids 是否为空，空则返回 True
        2. 获取客户端和 Collection 名称
        3. 调用 client.delete 删除
        4. 记录日志并返回结果

        参数：
            ids: chunk_id 列表
            collection_name: Collection 名称（必填）

        返回：
            bool: 成功返回 True
        """
        # 步骤1：空列表直接返回
        if not ids:
            return True

        # 步骤2：验证必填参数
        if not collection_name:
            raise ValueError("collection_name is required")

        client = self._get_client()
        name = collection_name

        # 步骤3：执行删除
        try:
            client.delete(collection_name=name, ids=ids)
            logger.info("milvus_delete_done", collection=name, count=len(ids))
            return True
        except Exception:
            logger.exception("milvus_delete_failed", collection=name, count=len(ids))
            return False


# 单例模式：全局共享一个 MilvusRepository 实例
# - 避免重复初始化连接
# - MilvusClient 内部有连接池，单例无副作用
milvus_repo = MilvusRepository()