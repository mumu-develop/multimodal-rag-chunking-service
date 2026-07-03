"""
异步 Milvus 向量数据库 Repository

使用 asyncio.run_in_executor 包装同步 pymilvus 操作，实现异步接口。

pymilvus 暂无官方 async 支持，使用线程池包装是最佳方案。

提供的方法：
- async_search: 异步向量检索
- async_upsert: 异步写入向量数据
- async_delete: 异步删除向量数据
"""
import asyncio
import structlog

from app.repositories.milvus_repo import milvus_repo

logger = structlog.get_logger()


class AsyncMilvusRepository:
    """
    异步 Milvus Repository。

    使用 run_in_executor 包装同步 milvus_repo 方法，
    避免阻塞事件循环。

    流程概述：
    1. 获取事件循环
    2. 使用 run_in_executor 在线程池执行同步操作
    3. 返回结果
    """

    async def search(
        self,
        vector: list[float],
        top_k: int,
        query_text: str | None = None,
        expr: str = "",
        output_fields: list[str] | None = None,
        collection_name: str = None,
        data_type: str = "",
    ) -> list[dict]:
        """
        异步向量检索。

        流程步骤：
        1. 获取事件循环
        2. 在线程池执行 milvus_repo.search
        3. 返回检索结果

        参数：
            vector: 查询向量（2560维）
            top_k: 返回结果数量
            query_text: 查询文本（可选，用于 BM25）
            expr: 过滤表达式
            output_fields: 返回字段列表
            collection_name: Collection 名称（必填）
            data_type: 数据类型过滤（"chunk"/"faq"/""全部）

        返回：
            list[dict]: 搜索结果列表
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            milvus_repo.search,
            vector,
            top_k,
            query_text,
            expr,
            output_fields,
            collection_name,
            data_type,
        )

    async def batch_upsert(
        self,
        entities: list[dict],
        collection_name: str = None,
    ) -> bool:
        """
        异步批量写入向量数据。

        参数：
            entities: 向量数据列表
            collection_name: Collection 名称

        返回：
            bool: 成功返回 True
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            milvus_repo.batch_upsert,
            entities,
            collection_name,
        )

    async def upsert_faq(
        self,
        faq_data: dict,
        collection_name: str = None,
    ) -> bool:
        """
        异步写入 FAQ 向量数据。

        参数：
            faq_data: FAQ 数据字典
            collection_name: Collection 名称

        返回：
            bool: 成功返回 True
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            milvus_repo.upsert_faq,
            faq_data,
            collection_name,
        )

    async def batch_upsert_faq(
        self,
        faq_items: list[dict],
        collection_name: str = None,
    ) -> bool:
        """
        异步批量写入 FAQ 向量数据。

        参数：
            faq_items: FAQ 数据列表
            collection_name: Collection 名称

        返回：
            bool: 成功返回 True
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            milvus_repo.batch_upsert_faq,
            faq_items,
            collection_name,
        )

    async def delete(
        self,
        ids: list[str],
        collection_name: str = None,
    ) -> bool:
        """
        异步删除向量数据。

        参数：
            ids: chunk_id 列表
            collection_name: Collection 名称

        返回：
            bool: 成功返回 True
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            milvus_repo.delete,
            ids,
            collection_name,
        )

    async def delete_faq(
        self,
        faq_ids: list[str],
        collection_name: str = None,
    ) -> bool:
        """
        异步删除 FAQ 向量数据。

        参数：
            faq_ids: FAQ ID 列表
            collection_name: Collection 名称

        返回：
            bool: 成功返回 True
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            milvus_repo.delete_faq,
            faq_ids,
            collection_name,
        )

    async def create_collection_if_not_exists(
        self,
        collection_name: str,
    ) -> bool:
        """
        异步创建 Collection（如不存在）。

        参数：
            collection_name: Collection 名称

        返回：
            bool: 成功返回 True
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            milvus_repo.create_collection_if_not_exists,
            collection_name,
        )

    async def drop_collection_by_name(
        self,
        collection_name: str,
    ) -> bool:
        """
        异步删除 Collection。

        参数：
            collection_name: Collection 名称

        返回：
            bool: 成功返回 True
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            milvus_repo.drop_collection_by_name,
            collection_name,
        )


# 单例模式：全局共享一个 AsyncMilvusRepository 实例
async_milvus_repo = AsyncMilvusRepository()