import threading
import os
import pickle
import time
import json

import structlog
from celery import Celery

# Celery worker 进程不经过 main.py，需独立配置 structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
)

from app.config import settings
from app.converters import get_converter
from app.converters.markdown_chunker import markdown_chunker
from app.converters.models import ConversionError
from app.chunkers.chunk_quality_evaluator import chunk_quality_evaluator
from app.embedders.unified_embedder import unified_embedder
from app.clients.llm_client import llm_client
from app.repositories.milvus_repo import milvus_repo
from app.repositories.kb_repo import kb_repo
from app.repositories.document_repo import doc_repo, get_doc_config_with_defaults
from app.evaluation.testset_generator import RagasTestsetGenerator
from app.models import TaskStatus, get_session
from app.repositories.chunk_repo import chunk_repo
from app.repositories.task_repo import task_repo
from app.repositories.testset_repo import testset_repo

logger = structlog.get_logger()


# ============================================================
# 流水线节点跟踪助手
# ============================================================

def _record_pipeline_node(task_id: str, name: str, node_type: str, status: str,
                          duration_ms: int = 0, message: str = "") -> None:
    """记录流水线节点执行状态到任务 pipeline_nodes 字段。"""
    try:
        task = task_repo.get(task_id)
        if not task:
            return
        nodes = list(task.pipeline_nodes or [])
        nodes.append({
            "name": name,
            "type": node_type,
            "status": status,
            "duration_ms": duration_ms,
            "message": message,
        })
        task_repo.update_pipeline_nodes(task_id, nodes)
    except Exception:
        logger.debug("pipeline_node_record_failed", task_id=task_id, name=name)

celery_app = Celery(
    "rag_chunking",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_pool="solo",
    imports=["app.tasks"],
    task_annotate={
        "*": {"max_retries": 3, "default_retry_delay": 60},
    },
)


from datetime import datetime, timedelta

from sqlalchemy import and_

from celery.signals import worker_ready

def _handle_worker_ready(**kwargs):
    """Worker 启动时自动恢复未完成的任务。"""
    count = recover_pending_tasks()
    if count > 0:
        logger.info("recovered_pending_tasks", count=count)

worker_ready.connect(_handle_worker_ready)

# ============================================================
# Redis 中间结果缓存（重试时跳过 parse/chunk/embed）
# ============================================================

_PIPELINE_CACHE_TTL = 3600  # 1 小时


def _get_redis():
    """获取 Redis 连接单例。"""
    import redis
    return redis.from_url(settings.REDIS_URL)


def _cache_pipeline_state(task_id: str, state: dict) -> None:
    """将流水线中间状态缓存到 Redis（chunks + entities）。"""
    try:
        r = _get_redis()
        key = f"pipeline_state:{task_id}"
        r.setex(key, _PIPELINE_CACHE_TTL, pickle.dumps(state))
        logger.info("pipeline_state_cached", task_id=task_id, keys=list(state.keys()))
    except Exception:
        logger.warning("pipeline_cache_failed", task_id=task_id)


def _restore_pipeline_state(task_id: str) -> dict | None:
    """从 Redis 恢复流水线中间状态。"""
    try:
        r = _get_redis()
        key = f"pipeline_state:{task_id}"
        data = r.get(key)
        if data:
            state = pickle.loads(data)
            logger.info("pipeline_state_restored", task_id=task_id, keys=list(state.keys()))
            return state
    except Exception:
        logger.warning("pipeline_cache_restore_failed", task_id=task_id)
    return None


def _delete_pipeline_state(task_id: str) -> None:
    """任务完成后清理缓存。"""
    try:
        r = _get_redis()
        r.delete(f"pipeline_state:{task_id}")
    except Exception:
        pass


# ============================================================
# MySQL 补偿同步缓存（用于 Milvus 成功但 MySQL 失败的场景）
# ============================================================

_PENDING_MYSQL_TTL = 86400  # 24 小时（补偿数据保留时间更长）


def _cache_pending_mysql_sync(task_id: str, pending_data: dict) -> None:
    """缓存待同步的 MySQL 数据（chunks + 元信息）。"""
    try:
        r = _get_redis()
        key = f"pending_mysql:{task_id}"
        r.setex(key, _PENDING_MYSQL_TTL, pickle.dumps(pending_data))
        logger.info("pending_mysql_cached", task_id=task_id, chunk_count=len(pending_data.get("chunks", [])))
    except Exception:
        logger.warning("pending_mysql_cache_failed", task_id=task_id)


def _restore_pending_mysql_sync(task_id: str) -> dict | None:
    """恢复待同步的 MySQL 数据。"""
    try:
        r = _get_redis()
        key = f"pending_mysql:{task_id}"
        data = r.get(key)
        if data:
            pending = pickle.loads(data)
            logger.info("pending_mysql_restored", task_id=task_id, chunk_count=len(pending.get("chunks", [])))
            return pending
    except Exception:
        logger.warning("pending_mysql_restore_failed", task_id=task_id)
    return None


def _delete_pending_mysql_sync(task_id: str) -> None:
    """补偿完成后清理缓存。"""
    try:
        r = _get_redis()
        r.delete(f"pending_mysql:{task_id}")
    except Exception:
        pass


def _migrate_pipeline_cache(old_task_id: str, new_task_id: str) -> bool:
    """恢复任务时，将旧 task_id 的 Redis 缓存迁移到新 task_id。"""
    try:
        r = _get_redis()
        old_key = f"pipeline_state:{old_task_id}"
        new_key = f"pipeline_state:{new_task_id}"
        data = r.get(old_key)
        if data:
            r.setex(new_key, _PIPELINE_CACHE_TTL, data)
            r.delete(old_key)
            logger.info("pipeline_cache_migrated", old=old_task_id, new=new_task_id)
            return True
        return False
    except Exception:
        return False


# ============================================================
# 独立重试装饰器（Milvus / MySQL 各自重试，不重跑整条流水线）
# ============================================================

def _retry_with_backoff(func, name: str, max_retries: int = 3, base_delay: float = 2.0):
    """
    对单个存储操作进行指数退避重试。

    Args:
        func: 可调用对象，返回 bool 表示成功/失败
        name: 操作名称（用于日志）
        max_retries: 最大重试次数
        base_delay: 基础延迟秒数

    Returns:
        bool: 操作是否成功
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            ok = func()
            if ok:
                return True
        except Exception as exc:
            last_err = exc

        if attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"{name}_retry",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
                error=str(last_err)[:100] if last_err else "",
            )
            time.sleep(delay)

    logger.error(f"{name}_exhausted", max_retries=max_retries, error=str(last_err)[:100] if last_err else "")
    return False


@celery_app.task(bind=True, name="tasks.parse_and_chunk")
def parse_and_chunk_task(self, file_path: str, doc_id: str, creator: str = "", kb_id: str = "", collection_name: str = "", source_type: str = "local_file", source_url: str = "", doc_name: str = "", file_size: int = 0, doc_type: str = "", resume_stage: str = None, chunk_strategy: str = "smart") -> dict:
    """
    Celery 任务：解析 -> 分块 -> VLM caption -> 单向量嵌入 -> 存储到 MySQL -> 存储到 Milvus。

    流程阶段：
    1. 解析文档（10%）
    2. 语义分块（20%）
    3. VLM caption（30%）       — 为图片 chunk 生成描述
    4. 单向量嵌入（50%）        — 统一编码所有 content 文本
    5. Milvus 批量存储（70%）   — vector + sparse_vector，带独立重试，支持指定 collection
    6. MySQL 批量存储（85%）    — 带独立重试

    重试优化：
    - 中间状态缓存到 Redis，重试时跳过 parse/chunk/embed
    - Milvus / MySQL 各自独立重试，互不影响
    """
    task_id = self.request.id
    total_start = time.time()

    try:
        # ========== 检查是否有缓存的中间状态（重试场景） ==========
        cached = _restore_pipeline_state(task_id)

        if cached and "chunks" in cached and "entities" in cached:
            # 跳过 parse/chunk/embed，直接从存储阶段开始
            logger.info("restored_from_cache", task_id=task_id)
            chunks = cached["chunks"]
            entities = cached["entities"]
            task_repo.upsert(task_id, status="processing", stage="storing_milvus", progress=70, doc_id=doc_id)
        else:
            # 正常流程：parse → chunk → embed
            # 阶段 1：解析（使用 Converter）
            task_repo.upsert(task_id, status="processing", stage="parsing", progress=10, doc_id=doc_id)
            stage_start = time.time()
            logger.info("stage_start", stage="parsing", task_id=task_id, doc_id=doc_id, doc_type=doc_type)

            # 获取 Converter
            converter = get_converter(doc_type)
            if not converter:
                raise ConversionError(
                    f"不支持的文档类型: {doc_type}",
                    doc_type=doc_type,
                    file_path=file_path
                )

            # 获取文件大小
            file_size = 0
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)

            # 调用 convert_full 获取完整结构化数据
            result = converter.convert_full(file_path, doc_id)

            # 处理 ConverterOutput 模型或 dict
            if hasattr(result, 'markdown'):
                markdown_text = result.markdown
                tables_data = [t.model_dump() if hasattr(t, 'model_dump') else t for t in result.tables]
                images_data = [img.model_dump() if hasattr(img, 'model_dump') else img for img in result.images]
                code_blocks_data = [cb.model_dump() if hasattr(cb, 'model_dump') else cb for cb in result.code_blocks]
            else:
                markdown_text = result.get("markdown", "")
                tables_data = result.get("tables", [])
                images_data = result.get("images", [])
                code_blocks_data = result.get("code_blocks", [])
            image_paths = [img.get("path") if isinstance(img, dict) else img.path for img in images_data]

            parsing_duration = int((time.time() - stage_start) * 1000)
            logger.info(
                "stage_done", stage="parsing",
                duration_ms=parsing_duration,
                tables=len(tables_data),
                images=len(image_paths),
                code_blocks=len(code_blocks_data),
            )
            _record_pipeline_node(task_id, "parsing", "parse", "completed",
                                  duration_ms=parsing_duration,
                                  message=f"tables={len(tables_data)}, images={len(image_paths)}")

            # 阶段 2：分块（统一入口）
            task_repo.upsert(task_id, stage="chunking", progress=20)
            stage_start = time.time()
            logger.info("stage_start", stage="chunking", task_id=task_id)

            # 统一调用 markdown_chunker，它会根据文档类型选择策略
            # - 纯表格文档（xlsx）：直接从 tables_data 生成切片
            # - 其他文档：按标题切分 + 内容片段识别
            chunks, tree_data = markdown_chunker.chunk(
                markdown_text, doc_id, doc_type, image_paths, creator, kb_id,
                doc_name=doc_name,
                tables_data=tables_data,
                code_blocks_data=code_blocks_data,
                images_data=images_data,
                chunk_strategy=chunk_strategy,
            )

            # 为每个 chunk 添加 source 信息
            for c in chunks:
                c["source_type"] = source_type
                c["source_url"] = source_url
                c["file_size"] = file_size

            logger.info(
                "stage_done", stage="chunking",
                duration_ms=int((time.time() - stage_start) * 1000),
                chunk_count=len(chunks),
            )
            _record_pipeline_node(task_id, "chunking", "chunk", "completed",
                                  duration_ms=int((time.time() - stage_start) * 1000),
                                  message=f"{len(chunks)} chunks")

            # 阶段 2.5：切分质量评估（闭环）
            task_repo.upsert(task_id, stage="quality_evaluation", progress=22)
            stage_start = time.time()
            logger.info("stage_start", stage="quality_evaluation", task_id=task_id)

            quality_metrics = chunk_quality_evaluator.evaluate(
                chunks,
                markdown_text,
                doc_type,
                chunk_strategy,
            )
            quality_report = chunk_quality_evaluator.generate_report(quality_metrics)

            logger.info(
                "stage_done", stage="quality_evaluation",
                duration_ms=int((time.time() - stage_start) * 1000),
                quality_score=quality_metrics.quality_score,
                avg_chunk_tokens=int(quality_metrics.avg_chunk_tokens),
                is_qualified=quality_metrics.quality_score >= chunk_quality_evaluator.TARGET_QUALITY,
            )
            _record_pipeline_node(task_id, "quality_evaluation", "evaluate", "completed",
                                  duration_ms=int((time.time() - stage_start) * 1000),
                                  message=f"quality_score={quality_metrics.quality_score:.4f}")

            # 缓存中间状态（重试时可用）
            _cache_pipeline_state(task_id, {
                "chunks": chunks,
                "entities": [],  # 后续填充
                "tree_data": tree_data,
                "quality_report": quality_report,
            })

            # 阶段 3：LLM 图片摘要
            # 图片处理策略：
            # - MySQL content：保持原始路径信息（不变）
            # - metadata.vlm_caption：存储 VLM 生成的描述（用于 Milvus）
            task_repo.upsert(task_id, stage="vlm_caption", progress=30)
            stage_start = time.time()
            logger.info("stage_start", stage="vlm_caption", task_id=task_id)

            image_chunks = [c for c in chunks if c["chunk_type"] == "image"]
            total_imgs = len(image_chunks)
            for idx, c in enumerate(image_chunks):
                image_path = c.get("metadata", {}).get("image_path", "")
                context = c.get("metadata", {}).get("context_text", "")
                heading = c.get("metadata", {}).get("heading_text", "")

                # VLM caption 生成（存到 metadata.vlm_caption，不覆盖 content）
                vlm_caption = ""
                if image_path:
                    img_name = os.path.basename(image_path) if os.path.exists(image_path) else "图片"
                    if len(context) >= 20:
                        summary = llm_client.summarize_image(context, heading)
                        vlm_caption = summary if summary else f"图片: {img_name}"
                    else:
                        vlm_caption = f"图片: {img_name}"
                elif len(context) >= 20:
                    summary = llm_client.summarize_image(context, heading)
                    vlm_caption = summary if summary else context
                else:
                    vlm_caption = "图片"

                # 存储到 metadata.vlm_caption（用于 Milvus）
                c["metadata"]["vlm_caption"] = vlm_caption

                if total_imgs > 1:
                    pct = int(30 + ((idx + 1) / total_imgs) * 20)  # 30~50%
                    task_repo.upsert(task_id, stage=f"vlm_caption:{idx+1}/{total_imgs}", progress=pct)

            logger.info(
                "stage_done", stage="vlm_caption",
                duration_ms=int((time.time() - stage_start) * 1000),
                img_count=len(image_chunks),
            )
            _record_pipeline_node(task_id, "vlm_caption", "vlm", "completed",
                                  duration_ms=int((time.time() - stage_start) * 1000),
                                  message=f"{len(image_chunks)} images captioned")

            # 阶段 4：单向量嵌入
            task_repo.upsert(task_id, stage="embedding", progress=50)
            stage_start = time.time()
            logger.info("stage_start", stage="embedding", task_id=task_id)

            text_embeddable = [c for c in chunks if c["chunk_type"] in ("text", "table", "code") and c["content"]]
            image_embeddable = [c for c in chunks if c["chunk_type"] == "image" and c["content"]]

            vector_map: dict[str, list[float]] = {}
            if text_embeddable:
                texts = [c["content"] for c in text_embeddable]
                dense_vectors = unified_embedder.encode_texts(texts)
                for c, v in zip(text_embeddable, dense_vectors):
                    vector_map[c["chunk_id"]] = v

            total_imgs_embed = len(image_embeddable)
            for idx, c in enumerate(image_embeddable):
                image_path = c.get("metadata", {}).get("image_path", "")
                # 图片向量编码使用 vlm_caption
                vlm_caption = c.get("metadata", {}).get("vlm_caption", "")
                caption = vlm_caption if vlm_caption else c["content"]
                if image_path:
                    try:
                        vec = unified_embedder.encode_multimodal(caption, image_path)
                    except Exception:
                        logger.warning("image_encoding_failed", chunk_id=c["chunk_id"], path=image_path)
                        vec = unified_embedder.encode_query(caption)
                else:
                    vec = unified_embedder.encode_query(caption)
                vector_map[c["chunk_id"]] = vec
                if total_imgs_embed > 1:
                    pct = int(50 + ((idx + 1) / total_imgs_embed) * 20)  # 50~70%
                    task_repo.upsert(task_id, stage=f"embedding:img_{idx+1}/{total_imgs_embed}", progress=pct)

            # 构建 Milvus 实体
            dim = settings.UNIFIED_VECTOR_DIM
            entities = []

            for c in chunks:
                cid = c["chunk_id"]
                dense_vec = vector_map.get(cid)
                if dense_vec is None:
                    dense_vec = [0.0] * dim

                # 图片：content 使用 vlm_caption，image_path 存图片路径
                image_path = ""
                milvus_content = c["content"]
                if c["chunk_type"] == "image":
                    image_path = c.get("metadata", {}).get("image_path", "")
                    # Milvus 用 vlm_caption 作为 content（用于向量搜索）
                    vlm_caption = c.get("metadata", {}).get("vlm_caption", "")
                    if vlm_caption:
                        milvus_content = vlm_caption

                # 表格/嵌套表格：content 使用 HTML 格式（便于 BM25 检索）
                # MySQL 存储 JSON 结构化数据，Milvus 存储扁平化 HTML
                if c["chunk_type"] in ("table", "nested_table"):
                    try:
                        table_data = json.loads(c["content"])
                        if table_data.get("html"):
                            milvus_content = table_data["html"]
                    except (json.JSONDecodeError, TypeError):
                        pass  # 保持原样

                caption = c.get("caption")
                metadata = c.get("metadata", {}).copy()
                if caption:
                    metadata["caption"] = caption
                # 添加文档名称到 metadata（用于搜索结果显示出处）
                doc_name = c.get("doc_name", "")
                if doc_name:
                    metadata["doc_name"] = doc_name

                entity = {
                    "chunk_id": cid,
                    "doc_id": c["doc_id"],
                    "chunk_type": c["chunk_type"],
                    "doc_type": metadata.get("doc_type", ""),
                    "page_num": c["page_num"],
                    "content": milvus_content,  # 图片用 vlm_caption
                    "image_path": image_path,
                    "metadata": metadata,
                    "vector": dense_vec,
                }
                entities.append(entity)

            logger.info(
                "stage_done", stage="embedding",
                duration_ms=int((time.time() - stage_start) * 1000),
                vector_count=len(vector_map),
            )
            _record_pipeline_node(task_id, "embedding", "embed", "completed",
                                  duration_ms=int((time.time() - stage_start) * 1000),
                                  message=f"{len(vector_map)} vectors")

            # 存储前缓存中间状态（重试时可跳过前面的计算）
            _cache_pipeline_state(task_id, {"chunks": chunks, "entities": entities})

        # ========== 阶段 5：Milvus 批量存储（独立重试） ==========
        task_repo.upsert(task_id, stage="storing_milvus", progress=70)
        stage_start = time.time()
        logger.info("stage_start", stage="storing_milvus", task_id=task_id, collection=collection_name)

        # 使用指定 collection 或默认 collection
        target_collection = collection_name or settings.MILVUS_COLLECTION_NAME
        # 确保 collection 存在
        milvus_repo.create_collection_if_not_exists(target_collection)
        milvus_ok = _retry_with_backoff(
            lambda: milvus_repo.batch_upsert(entities, collection_name=target_collection),
            name="milvus",
            max_retries=5,
            base_delay=2.0,
        )
        if not milvus_ok:
            raise Exception("Milvus storage failed after retry, MySQL not touched")

        logger.info(
            "stage_done", stage="storing_milvus",
            duration_ms=int((time.time() - stage_start) * 1000),
            entity_count=len(entities),
            collection=target_collection,
        )
        _record_pipeline_node(task_id, "storing_milvus", "storage", "completed",
                              duration_ms=int((time.time() - stage_start) * 1000),
                              message=f"{len(entities)} entities in Milvus ({target_collection})")

        # ========== 阶段 6：MySQL 批量存储（独立重试） ==========
        task_repo.upsert(task_id, stage="storing_mysql", progress=85)
        stage_start = time.time()
        logger.info("stage_start", stage="storing_mysql", task_id=task_id)

        mysql_ok = _retry_with_backoff(
            lambda: chunk_repo.insert_chunks(chunks),
            name="mysql",
            max_retries=5,
            base_delay=2.0,
        )
        logger.info(
            "stage_done", stage="storing_mysql",
            duration_ms=int((time.time() - stage_start) * 1000),
        )
        _record_pipeline_node(task_id, "storing_mysql", "storage", "completed" if mysql_ok else "failed",
                              duration_ms=int((time.time() - stage_start) * 1000),
                              message=f"{len(chunks)} chunks in MySQL")

        # ========== 完成 ==========
        total_duration = int((time.time() - total_start) * 1000)

        # 处理 MySQL 失败但 Milvus 成功的情况
        if milvus_ok and not mysql_ok:
            # Milvus 数据已可搜索，但 MySQL 元数据缺失
            # 将 chunks 信息缓存，启动补偿任务
            logger.warning(
                "milvus_ok_mysql_pending",
                task_id=task_id,
                doc_id=doc_id,
                chunk_count=len(chunks),
                message="Milvus 成功，MySQL 失败，启动补偿任务",
            )
            # 缓存待同步的 chunks 数据（用于后续补偿）
            pending_data = {
                "chunks": chunks,
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "kb_id": kb_id,
                "file_size": file_size,
                "file_path": file_path,
                "source_type": source_type,
                "source_url": source_url,
                "collection_name": target_collection,
            }
            _cache_pending_mysql_sync(task_id, pending_data)

            # 任务标记为 partial_success（向量可搜索，元数据待同步）
            task_repo.upsert(
                task_id,
                status="partial_success",
                progress=95,
                stage="mysql_pending",
                chunk_count=len(chunks),
                error_message="MySQL sync pending, compensation task enqueued",
            )
            _record_pipeline_node(task_id, "mysql_pending", "storage", "warning",
                                  duration_ms=total_duration,
                                  message=f"Milvus OK ({len(chunks)}), MySQL pending")

            # 启动补偿任务（延迟 30 秒执行）
            mysql_compensation_task.apply_async(
                args=[task_id],
                countdown=30,
            )

            _cleanup_input_file(file_path)
            return {
                "status": "partial_success",
                "chunk_count": len(chunks),
                "duration_ms": total_duration,
                "message": "Vectors searchable in Milvus, MySQL metadata pending",
            }

        if mysql_ok and milvus_ok:
            # 任务成功后创建 Document 记录
            try:
                doc_repo.create(
                    doc_id=doc_id,
                    kb_id=int(kb_id) if kb_id else 0,
                    doc_name=doc_name,
                    doc_type=doc_type,
                    file_size=file_size,
                    file_path=file_path,
                    source_type=source_type,
                    source_url=source_url,
                    status="success",
                    chunk_count=len(chunks),
                )
            except Exception:
                logger.exception("doc_create_failed_after_task", doc_id=doc_id)

            task_repo.upsert(
                task_id,
                status="completed",
                progress=100,
                stage="done",
                chunk_count=len(chunks),
            )
            _record_pipeline_node(task_id, "done", "pipeline", "completed",
                                  duration_ms=total_duration,
                                  message=f"Total: {len(chunks)} chunks, {total_duration}ms")
            logger.info(
                "task_completed",
                task_id=task_id,
                duration_ms=total_duration,
                chunk_count=len(chunks),
            )

            # 获取质量报告（从缓存中）
            cached_state = _restore_pipeline_state(task_id)
            quality_report = cached_state.get("quality_report", {}) if cached_state else {}

            _cleanup_input_file(file_path)
            _delete_pipeline_state(task_id)  # 清理缓存
            return {
                "status": "completed",
                "chunk_count": len(chunks),
                "duration_ms": total_duration,
                "quality_report": quality_report,
            }
        else:
            raise Exception(f"Storage failed: mysql={mysql_ok}, milvus={milvus_ok}")

    except Exception as exc:
        logger.exception("task_failed", task_id=task_id, error=str(exc))
        task_repo.upsert(
            task_id,
            status="failed",
            progress=0,
            stage="error",
            error_message=str(exc),
        )
        _record_pipeline_node(task_id, "error", "pipeline", "failed",
                              message=str(exc)[:200])
        _cleanup_input_file(file_path)
        self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


def _cleanup_input_file(file_path: str) -> None:
    """安全删除输入文件（静默失败，不抛异常）。"""
    try:
        if file_path and os.path.isfile(file_path):
            os.unlink(file_path)
            logger.debug("input_file_cleaned", path=file_path)
    except Exception:
        logger.debug("input_file_cleanup_failed", path=file_path)


def enqueue_parse(file_path: str, doc_id: str, creator: str = "",
                   kb_id: str = "", source_type: str = "local_file",
                   source_url: str = "", doc_name: str = "",
                   file_size: int = 0, doc_type: str = "",
                   chunk_strategy: str = "smart") -> str:
    """提交解析任务到 Celery 并返回 task_id。

    Args:
        chunk_strategy: 切分策略（smart/heading/semantic/recursive/fixed）
    """
    # 获取 KB 的 collection_name
    collection_name = settings.MILVUS_COLLECTION_NAME  # 默认
    if kb_id:
        from app.repositories.kb_repo import kb_repo
        kb = kb_repo.get_by_id(int(kb_id))
        if kb:
            collection_name = kb.collection_name

    task = parse_and_chunk_task.apply_async(args=[file_path, doc_id, creator, kb_id, collection_name, source_type, source_url, doc_name, file_size, doc_type, None, chunk_strategy])
    task_args = {
        "file_path": file_path, "doc_id": doc_id, "creator": creator,
        "kb_id": kb_id, "source_type": source_type,
        "source_url": source_url, "doc_name": doc_name,
        "file_size": file_size, "doc_type": doc_type,
        "collection_name": collection_name,
        "chunk_strategy": chunk_strategy,
    }
    task_repo.upsert(task.id, doc_id=doc_id, status="pending", progress=0, stage="queued",
                task_args=task_args, task_type="parse_and_chunk",
                doc_name=doc_name)
    logger.info("task_enqueued", task_id=task.id, doc_id=doc_id, kb_id=kb_id, collection=collection_name, chunk_strategy=chunk_strategy)
    return task.id


# ============================================================
# MySQL 补偿同步任务
# ============================================================

@celery_app.task(bind=True, name="tasks.mysql_compensation", max_retries=5)
def mysql_compensation_task(self, task_id: str) -> dict:
    """
    MySQL 补偿同步任务：当 Milvus 成功但 MySQL 失败时，尝试重新同步 MySQL。

    该任务会：
    1. 从 Redis 缓存中恢复待同步的 chunks 数据
    2. 尝试写入 MySQL（最多 5 次重试）
    3. 如果成功，更新任务状态为 completed，清理缓存
    4. 如果失败，继续重试（每次延迟 2 分钟）

    Args:
        task_id: 原解析任务的 ID

    Returns:
        dict: {"status": "completed" | "failed", "chunk_count": int}
    """
    logger.info("mysql_compensation_start", task_id=task_id, retry=self.request.retries)

    # 从缓存恢复数据
    pending_data = _restore_pending_mysql_sync(task_id)
    if not pending_data:
        logger.warning("mysql_compensation_no_data", task_id=task_id)
        return {"status": "failed", "message": "No pending data found in cache"}

    chunks = pending_data.get("chunks", [])
    doc_id = pending_data.get("doc_id", "")
    doc_name = pending_data.get("doc_name", "")
    doc_type = pending_data.get("doc_type", "")
    kb_id = pending_data.get("kb_id", "")
    file_size = pending_data.get("file_size", 0)
    file_path = pending_data.get("file_path", "")
    source_type = pending_data.get("source_type", "local_file")
    source_url = pending_data.get("source_url", "")

    # 尝试写入 MySQL
    mysql_ok = _retry_with_backoff(
        lambda: chunk_repo.insert_chunks(chunks),
        name="mysql_compensation",
        max_retries=3,
        base_delay=2.0,
    )

    if mysql_ok:
        # MySQL 成功，更新任务状态
        logger.info("mysql_compensation_success", task_id=task_id, chunk_count=len(chunks))

        # 创建 Document 记录
        try:
            doc_repo.create(
                doc_id=doc_id,
                kb_id=int(kb_id) if kb_id else 0,
                doc_name=doc_name,
                doc_type=doc_type,
                file_size=file_size,
                file_path=file_path,
                source_type=source_type,
                source_url=source_url,
                status="success",
                chunk_count=len(chunks),
            )
        except Exception:
            logger.exception("doc_create_failed_compensation", doc_id=doc_id)

        # 更新任务状态
        task_repo.upsert(
            task_id,
            status="completed",
            progress=100,
            stage="done",
            chunk_count=len(chunks),
            error_message="",  # 清除之前的错误信息
        )
        _record_pipeline_node(task_id, "mysql_compensated", "storage", "completed",
                              message=f"MySQL sync completed: {len(chunks)} chunks")

        # 清理缓存
        _delete_pending_mysql_sync(task_id)
        _cleanup_input_file(file_path)

        return {
            "status": "completed",
            "chunk_count": len(chunks),
            "message": "MySQL sync compensation completed",
        }
    else:
        # MySQL 仍然失败，继续重试
        logger.warning(
            "mysql_compensation_failed",
            task_id=task_id,
            retry=self.request.retries,
            max_retries=5,
        )

        if self.request.retries >= 5:
            # 达到最大重试次数，标记为最终失败
            logger.error("mysql_compensation_max_retries", task_id=task_id)
            task_repo.upsert(
                task_id,
                status="partial_success",  # 保持 partial_success，向量仍可搜索
                stage="mysql_failed_final",
                error_message=f"MySQL sync failed after {self.request.retries} compensation retries",
            )
            _record_pipeline_node(task_id, "mysql_failed_final", "storage", "failed",
                                  message="Max retries reached, MySQL sync abandoned")
            return {
                "status": "failed",
                "chunk_count": len(chunks),
                "message": "MySQL sync failed after max retries",
            }

        # 继续重试（延迟 2 分钟）
        raise self.retry(countdown=120)


# ============================================================
# RAGAS 测试集生成任务
# ============================================================

class _HeartbeatProgress:
    """后台线程定期更新任务进度，避免长时间无进度变化。"""

    def __init__(self, task_id: str, start_pct: int, end_pct: int, stage: str = "generating"):
        self.task_id = task_id
        self.start_pct = start_pct
        self.end_pct = end_pct
        self.stage = stage
        self._stop_event = threading.Event()
        self._start_time = time.time()

    def _worker(self, interval: float = 15):
        """后台线程：每 interval 秒递增进度。"""
        while not self._stop_event.wait(interval):
            elapsed = time.time() - self._start_time
            # 按时间线性推算进度（保守估计，不超过 end_pct）
            ratio = min(elapsed / (elapsed + 60), 0.95)  # 假设总时长约 elapsed + 60s
            pct = int(self.start_pct + ratio * (self.end_pct - self.start_pct))
            pct = max(self.start_pct, min(self.end_pct, pct))
            task_repo.upsert(
                self.task_id,
                stage=f"{self.stage}:elapsed_{int(elapsed)}s",
                progress=pct,
            )

    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)


@celery_app.task(bind=True, name="tasks.generate_ragas_testset")
def generate_ragas_testset_task(
    self,
    set_id: str,
    set_name: str,
    doc_ids: list[str],
    testset_size: int,
    chunk_type_filter: str | None,
    llm_config: dict,
    embed_config: dict,
) -> dict:
    """
    Celery 任务：异步生成 RAGAS 测试集。

    流程：
    1. 初始化 Generator（10%）
    2. 逐文档生成测试用例（20%~80%）
    3. 持久化到数据库（90%）
    4. 完成（100%）
    """
    task_id = self.request.id
    total_start = time.time()

    try:
        # 初始化进度
        task_repo.upsert(
            task_id,
            status="processing",
            stage="initializing",
            progress=10,
        )

        # 初始化 Generator
        stage_start = time.time()
        logger.info("stage_start", stage="testset_generation", task_id=task_id, doc_ids=doc_ids)

        generator = RagasTestsetGenerator(
            llm_config=llm_config,
            embed_config=embed_config,
            use_dashscope=True,
        )

        task_repo.upsert(task_id, stage="generating", progress=20)

        raw_cases = []
        total_docs = len(doc_ids)

        for idx, single_doc_id in enumerate(doc_ids):
            # 跳过已完成的文档（恢复/重试场景），避免重复 LLM 调用
            if testset_repo.has_records_for(set_id, single_doc_id):
                logger.info("testset_skip_already_done", set_id=set_id, doc_id=single_doc_id, idx=idx)
                doc_start_pct = 20 + int((idx / total_docs) * 60)
                doc_end_pct = 20 + int(((idx + 1) / total_docs) * 60)
                task_repo.upsert(task_id, stage=f"generating:{idx+1}/{total_docs}:skipped", progress=doc_end_pct)
                continue

            # 启动后台进度更新
            doc_start = 20 + int((idx / total_docs) * 60)
            doc_end = 20 + int(((idx + 1) / total_docs) * 60)
            heartbeat = _HeartbeatProgress(
                task_id, start_pct=doc_start, end_pct=doc_end,
                stage=f"generating:{idx+1}/{total_docs}",
            )
            heartbeat.start()

            try:
                doc_cases = generator.generate_from_doc(
                    doc_id=single_doc_id,
                    testset_size=testset_size,
                    chunk_type_filter=chunk_type_filter,
                )
            finally:
                heartbeat.stop()

            if doc_cases:
                raw_cases.extend(doc_cases)

                db_items = [
                    {
                        "set_id": set_id,
                        "set_name": set_name,
                        "query": tc["query"],
                        "ground_truth": tc["ground_truth"],
                        "doc_id": tc["doc_id"],
                        "tags": tc["tags"],
                    }
                    for tc in doc_cases
                ]
                insert_ok = testset_repo.batch_insert(db_items)
                logger.info("testset_partial_saved", doc_id=single_doc_id, count=len(doc_cases), ok=insert_ok)

            task_repo.upsert(
                task_id,
                stage=f"generating:{idx+1}/{total_docs}:done",
                progress=doc_end,
            )

        logger.info(
            "stage_done", stage="testset_generation",
            duration_ms=int((time.time() - stage_start) * 1000),
            case_count=len(raw_cases),
        )

        if not raw_cases:
            task_repo.upsert(
                task_id,
                status="completed",
                progress=100,
                stage="done",
            )
            logger.info("task_completed_no_cases", task_id=task_id)
            return {
                "status": "completed",
                "set_id": set_id,
                "generated": 0,
                "duration_ms": int((time.time() - total_start) * 1000),
            }

        # 每个 doc 已在上面持久化，这里只做最终确认
        task_repo.upsert(task_id, stage="storing", progress=90)

        # 完成
        task_repo.upsert(
            task_id,
            status="completed",
            progress=100,
            stage="done",
        )

        total_duration = int((time.time() - total_start) * 1000)
        logger.info(
            "task_completed",
            task_id=task_id,
            duration_ms=total_duration,
            set_id=set_id,
            generated=len(raw_cases),
        )
        return {
            "status": "completed",
            "set_id": set_id,
            "generated": len(raw_cases),
            "duration_ms": total_duration,
        }

    except Exception as exc:
        logger.exception("testset_generation_failed", task_id=task_id, error=str(exc))
        task_repo.upsert(
            task_id,
            status="failed",
            progress=0,
            stage="error",
            error_message=str(exc),
        )
        self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


def enqueue_generate_testset(
    set_id: str,
    set_name: str,
    doc_ids: list[str],
    testset_size: int,
    chunk_type_filter: str | None,
    llm_config: dict,
    embed_config: dict,
    method: str = "fast",
) -> str:
    """提交测试集生成任务到 Celery 并返回 task_id。

    Args:
        method: "fast"=LLM Prompt 直接生成（默认），"ragas"=RAGAS 框架
    """
    # 保存任务参数到数据库，用于 worker 重启后恢复
    task_args = {
        "set_id": set_id,
        "set_name": set_name,
        "doc_ids": doc_ids,
        "testset_size": testset_size,
        "chunk_type_filter": chunk_type_filter,
        "llm_config": llm_config,
        "embed_config": embed_config,
        "method": method,
    }

    if method == "ragas":
        task = generate_ragas_testset_task.apply_async(
            args=[set_id, set_name, doc_ids, testset_size, chunk_type_filter, llm_config, embed_config]
        )
        task_repo.upsert(task.id, status="pending", progress=0, stage="queued",
                    task_args=task_args, task_type="generate_testset")
    else:
        task = generate_ragas_testset_fast_task.apply_async(
            args=[set_id, set_name, doc_ids, testset_size, chunk_type_filter, llm_config]
        )
        task_repo.upsert(task.id, status="pending", progress=0, stage="queued",
                    task_args=task_args, task_type="generate_testset")

    logger.info("testset_task_enqueued", task_id=task.id, set_id=set_id, method=method)
    return task.id


# ============================================================
# 方案 B：LLM Prompt 直接生成 QA 对（快速）
# ============================================================

# 要点提取：每组 chunk 数量（控制上下文窗口）
_EXTRACT_BATCH_CHUNKS = 50


# 要点提取：每组 chunk 数量（控制上下文窗口）
_EXTRACT_BATCH_CHUNKS = 50


def _call_llm(
    llm_config: dict,
    system_prompt: str,
    user_msg: str,
    max_tokens: int = 4096,
    timeout: int = 180,
) -> str:
    """通用 LLM 调用函数。"""
    import json as _json
    import urllib.request

    model = llm_config.get("model_name", "qwen3.6-plus")
    api_key = llm_config.get("api_key", "")
    base_url = llm_config.get("base_url", "https://coding.dashscope.aliyuncs.com/v1")

    payload = _json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    })

    url = f"{base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url,
        data=payload.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = _json.loads(resp.read().decode("utf-8"))

    content = ""
    if "choices" in body and body["choices"]:
        content = body["choices"][0].get("message", {}).get("content", "").strip()

    return content


def _extract_key_points(llm_config: dict, chunks_text: str) -> str:
    """从一批 chunk 中提取关键要点（一句话列表）。"""
    system_prompt = (
        "你是一个文档分析助手。请从提供的文档内容中提取关键信息要点。\n\n"
        "要求：\n"
        "1. 列出所有重要的事实、数字、概念、因果关系等\n"
        "2. 每条要点一句话，包含足够细节\n"
        "3. 不要遗漏重要信息\n"
        "4. 只输出要点列表，不要其他内容\n"
    )

    user_msg = f"请从以下文档中提取关键要点：\n\n{chunks_text}"

    return _call_llm(llm_config, system_prompt, user_msg, max_tokens=4096, timeout=120)


def _generate_qa_from_keypoints(llm_config: dict, key_points: str, num_qa: int) -> list[dict]:
    """从关键要点列表生成 QA 对。

    Returns:
        [{"query": "...", "ground_truth": "...", "tags": ["simple_query"]}, ...]
    """
    import json as _json

    system_prompt = (
        "你是一个专业的 QA 测试集生成助手。请根据提供的文档关键要点，"
        f"生成 {num_qa} 个问题和参考答案。\n\n"
        "要求：\n"
        "1. 每个问题基于一个或多个要点\n"
        "2. 答案应准确、简洁，可以直接作为检索系统的参考答案\n"
        "3. 问题类型多样化：数字查询、概念解释、因果关系等\n"
        "4. 不要生成要点中没有信息的开放式问题\n"
        "5. 只输出 JSON 数组格式，不要输出其他内容\n\n"
        'JSON 格式：[{"query": "问题", "ground_truth": "答案"}, ...]'
    )

    user_msg = f"请从以下关键要点中生成 {num_qa} 个 QA 对：\n\n{key_points}"

    content = _call_llm(llm_config, system_prompt, user_msg, max_tokens=8192, timeout=180)

    # 解析 JSON
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1]
        if content.startswith("json"):
            content = content[4:].strip()

    qa_list = _json.loads(content)

    result = []
    for item in qa_list:
        q = item.get("query", "").strip()
        a = item.get("ground_truth", "").strip()
        if q and a:
            result.append({
                "query": q,
                "ground_truth": a,
                "tags": ["simple_query"],
            })

    return result


@celery_app.task(bind=True, name="tasks.generate_ragas_testset_fast")
def generate_ragas_testset_fast_task(
    self,
    set_id: str,
    set_name: str,
    doc_ids: list[str],
    testset_size: int,
    chunk_type_filter: str | None,
    llm_config: dict,
) -> dict:
    """
    Celery 任务：快速生成 RAGAS 测试集（方案 B）。

    直接用 LLM Prompt 生成 QA 对，跳过 RAGAS 框架的 SummaryExtractor 等慢步骤。
    进度更新策略：每完成一个 batch 的 LLM 请求才更新，不做估算。
    """
    task_id = self.request.id
    total_start = time.time()

    try:
        task_repo.upsert(task_id, status="processing", stage="initializing", progress=10)

        all_cases = []
        total_docs = len(doc_ids)

        for idx, single_doc_id in enumerate(doc_ids):
            doc_start = 10 + int((idx / total_docs) * 80)
            doc_end = 10 + int(((idx + 1) / total_docs) * 80)

            # 获取文档 chunk
            chunks = chunk_repo.get_all_by_doc(single_doc_id)

            # 过滤 chunk 类型
            if chunk_type_filter:
                chunks = [c for c in chunks if getattr(c, "chunk_type", None) == chunk_type_filter]
            else:
                chunks = [c for c in chunks if getattr(c, "chunk_type", None) in ("text", "table", "code")]

            logger.info("fast_doc_start", doc_id=single_doc_id, total_chunks=len(chunks), idx=f"{idx+1}/{total_docs}")
            task_repo.upsert(task_id, stage=f"generating:{idx+1}/{total_docs}:preparing", progress=doc_start)

            # 跳过已完成的文档（恢复/重试场景），避免重复 LLM 调用
            if testset_repo.has_records_for(set_id, single_doc_id):
                logger.info("fast_skip_already_done", set_id=set_id, doc_id=single_doc_id, idx=idx)
                task_repo.upsert(task_id, stage=f"generating:{idx+1}/{total_docs}:skipped", progress=doc_end)
                continue

            # 提取有效 chunk 文本
            chunk_texts = []
            for c in chunks:
                content = getattr(c, "content", "") or ""
                if len(content.strip()) >= 50:
                    chunk_texts.append(content)

            logger.info("fast_doc_chunks", doc_id=single_doc_id, usable_chunks=len(chunk_texts))

            if not chunk_texts:
                task_repo.upsert(task_id, stage=f"generating:{idx+1}/{total_docs}:no_content", progress=doc_end)
                continue

            # ===== Stage 1: 分组提取关键要点 =====
            num_groups = max(1, len(chunk_texts) // _EXTRACT_BATCH_CHUNKS + 1)
            all_key_points = []

            for gi in range(num_groups):
                g_start = gi * _EXTRACT_BATCH_CHUNKS
                g_end = g_start + _EXTRACT_BATCH_CHUNKS
                group_chunks = chunk_texts[g_start:g_end]
                if not group_chunks:
                    break

                group_text = "\n\n---\n\n".join(group_chunks)
                extract_progress = doc_start + int((gi / num_groups) * (doc_end - doc_start) * 0.5)
                task_repo.upsert(
                    task_id,
                    stage=f"generating:{idx+1}/{total_docs}:extract_{gi+1}/{num_groups}",
                    progress=extract_progress,
                )
                logger.info(
                    "fast_extract_points",
                    doc_id=single_doc_id,
                    group=f"{gi+1}/{num_groups}",
                    chunk_count=len(group_chunks),
                )

                try:
                    t0 = time.time()
                    points = _extract_key_points(llm_config, group_text)
                    elapsed = round((time.time() - t0) * 1000, 0)
                    logger.info("fast_extract_done", doc_id=single_doc_id, group=f"{gi+1}/{num_groups}", latency_ms=elapsed)
                    if points:
                        all_key_points.append(points)
                except Exception as e:
                    logger.warning("fast_extract_failed", doc_id=single_doc_id, group=f"{gi+1}/{num_groups}", error=str(e)[:200])

            if not all_key_points:
                logger.warning("fast_no_keypoints", doc_id=single_doc_id)
                task_repo.upsert(task_id, stage=f"generating:{idx+1}/{total_docs}:no_keypoints", progress=doc_end)
                continue

            # ===== Stage 2: 合并要点生成 QA =====
            merged_points = "\n\n".join(all_key_points)
            generate_progress = doc_start + int((doc_end - doc_start) * 0.7)
            task_repo.upsert(
                task_id,
                stage=f"generating:{idx+1}/{total_docs}:generating_qa",
                progress=generate_progress,
            )
            logger.info("fast_generate_qa", doc_id=single_doc_id, points_len=len(merged_points), num_qa=testset_size)

            try:
                t0 = time.time()
                doc_cases = _generate_qa_from_keypoints(llm_config, merged_points, testset_size)
                elapsed = round((time.time() - t0) * 1000, 0)
                logger.info("fast_generate_qa_done", doc_id=single_doc_id, qa_count=len(doc_cases), latency_ms=elapsed)
            except Exception as e:
                logger.warning("fast_generate_qa_failed", doc_id=single_doc_id, error=str(e)[:300])
                doc_cases = []

            if doc_cases:
                db_items = [
                    {
                        "set_id": set_id,
                        "set_name": set_name,
                        "query": tc["query"],
                        "ground_truth": tc["ground_truth"],
                        "doc_id": single_doc_id,
                        "tags": tc["tags"],
                    }
                    for tc in doc_cases
                ]
                insert_ok = testset_repo.batch_insert(db_items)
                logger.info("testset_partial_saved", doc_id=single_doc_id, count=len(doc_cases), ok=insert_ok)

                all_cases.extend(doc_cases)
                logger.info("fast_doc_done", doc_id=single_doc_id, qa_count=len(doc_cases), total_generated=len(all_cases))

            task_repo.upsert(
                task_id,
                stage=f"generating:{idx+1}/{total_docs}:done",
                progress=doc_end,
            )

        if not all_cases:
            task_repo.upsert(task_id, status="completed", progress=100, stage="done")
            return {
                "status": "completed",
                "set_id": set_id,
                "generated": 0,
                "duration_ms": int((time.time() - total_start) * 1000),
            }

        # 完成
        task_repo.upsert(task_id, stage="storing", progress=90)
        task_repo.upsert(task_id, status="completed", progress=100, stage="done")

        total_duration = int((time.time() - total_start) * 1000)
        logger.info(
            "testset_fast_completed",
            task_id=task_id,
            duration_ms=total_duration,
            set_id=set_id,
            generated=len(all_cases),
        )
        return {
            "status": "completed",
            "set_id": set_id,
            "generated": len(all_cases),
            "duration_ms": total_duration,
        }

    except Exception as exc:
        logger.exception("testset_fast_failed", task_id=task_id, error=str(exc))
        task_repo.upsert(
            task_id,
            status="failed",
            progress=0,
            stage="error",
            error_message=str(exc),
        )
        self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


# ============================================================
# Worker 启动时恢复未完成的任务
# ============================================================

def recover_pending_tasks() -> int:
    """Worker 启动时扫描数据库，重新提交未完成的任务。

    Returns:
        恢复的任务数量。
    """
    session = get_session()
    try:
        # 查找 pending 和 processing 状态的任务，且超过 5 分钟未更新（僵尸任务）
        stale_threshold = datetime.now() - timedelta(minutes=5)
        tasks = (
            session.query(TaskStatus)
            .filter(
                and_(
                    TaskStatus.status.in_(["pending", "processing"]),
                    TaskStatus.gmt_modified < stale_threshold,
                )
            )
            .all()
        )

        recovered = 0
        for task in tasks:
            task_args = getattr(task, "task_args", None)
            task_type = getattr(task, "task_type", None)
            last_stage = task.stage or ""

            if not task_args:
                logger.warning("recover_skip_no_args", task_id=task.task_id)
                continue

            # 保存旧 task_id（用于迁移 Redis 缓存）
            old_task_id = task.task_id

            if task_type == "generate_testset":
                method = task_args.get("method", "fast")
                set_name = task_args.get("set_name", "未命名测试集")
                if method == "ragas":
                    new_task = generate_ragas_testset_task.apply_async(
                        args=[
                            task_args["set_id"],
                            set_name,
                            task_args["doc_ids"],
                            task_args["testset_size"],
                            task_args["chunk_type_filter"],
                            task_args["llm_config"],
                            task_args["embed_config"],
                        ]
                    )
                else:
                    new_task = generate_ragas_testset_fast_task.apply_async(
                        args=[
                            task_args["set_id"],
                            set_name,
                            task_args["doc_ids"],
                            task_args["testset_size"],
                            task_args["chunk_type_filter"],
                            task_args["llm_config"],
                        ]
                    )
                # 用新的 task_id 更新记录
                task.task_id = new_task.id
                task.status = "pending"
                task.progress = 0
                task.stage = "recovered"
                session.commit()
                recovered += 1
                logger.info(
                    "task_recovered",
                    old_task_id=old_task_id,
                    new_task_id=new_task.id,
                    last_stage=last_stage,
                    method=method,
                )

            elif task_type == "parse_and_chunk":
                if "file_path" in task_args:
                    # parse_and_chunk 任务（文件上传）
                    new_task = parse_and_chunk_task.apply_async(
                        args=[
                            task_args.get("file_path", ""),
                            task_args.get("doc_id", ""),
                            task_args.get("creator", ""),
                            task_args.get("kb_id", ""),
                        ],
                        kwargs={"resume_stage": last_stage} if last_stage else {},
                    )
                    _migrate_pipeline_cache(old_task_id, new_task.id)
                else:
                    logger.warning("recover_parse_no_args", task_id=task.task_id)
                    continue

                # 更新数据库记录
                task.task_id = new_task.id
                task.status = "pending"
                task.progress = 0
                task.stage = "recovered"
                session.commit()
                recovered += 1
                logger.info(
                    "task_recovered",
                    old_task_id=old_task_id,
                    new_task_id=new_task.id,
                    last_stage=last_stage,
                    method="parse_and_chunk",
                )
            else:
                logger.warning("recover_unknown_type", task_id=task.task_id, task_type=task_type)

        return recovered
    except Exception:
        session.rollback()
        logger.exception("recover_tasks_failed")
        return 0
    finally:
        session.close()


# ============================================================
# 向量重建任务
# ============================================================

@celery_app.task(bind=True, name="tasks.rebuild_vectors")
def rebuild_vectors_task(self, doc_id: str):
    """对文档所有切片重新向量化并写入 Milvus。"""
    from app.repositories.chunk_repo import chunk_repo
    from app.embedders.unified_embedder import unified_embedder
    from app.milvus_repo import milvus_repo

    self.update_state(state="PROGRESS", meta={"stage": "rebuild_vectors", "progress": 0})
    logger.info("rebuild_vectors_started", doc_id=doc_id)

    chunks = chunk_repo.get_all_by_doc(doc_id)
    if not chunks:
        logger.warning("rebuild_vectors_no_chunks", doc_id=doc_id)
        return {"status": "completed", "chunks_processed": 0}

    # 获取 collection_name
    first_chunk = chunks[0]
    collection_name = settings.MILVUS_COLLECTION_NAME
    if first_chunk.kb_id:
        from app.repositories.kb_repo import kb_repo
        kb = kb_repo.get_by_id(int(first_chunk.kb_id))
        if kb:
            collection_name = kb.collection_name

    total = len(chunks)
    vectors = []
    for i, chunk in enumerate(chunks):
        progress = int((i + 1) / total * 100)
        self.update_state(state="PROGRESS", meta={"stage": "rebuild_vectors", "progress": progress})

        embedding = unified_embedder.embed_text(chunk.content)
        vectors.append({
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "vector": embedding,
            "content": chunk.content,
            "chunk_type": chunk.chunk_type,
        })

        if (i + 1) % 10 == 0:
            logger.info("rebuild_vectors_progress", doc_id=doc_id, completed=i + 1, total=total)

    # 先删除旧向量再插入新向量
    chunk_ids = [c.chunk_id for c in chunks]
    milvus_repo.delete(chunk_ids, collection_name=collection_name)
    # 注意：insert 方法不存在，需要重构为 batch_upsert

    logger.info("rebuild_vectors_completed", doc_id=doc_id, chunks_processed=total)
    return {"status": "completed", "chunks_processed": total}


def enqueue_rebuild_vectors(doc_id: str) -> str:
    """提交向量重建任务。"""
    task = rebuild_vectors_task.apply_async(args=[doc_id])
    task_repo.upsert(
        task.id,
        status="pending",
        doc_id=doc_id,
        stage="rebuild_vectors",
        task_type="rebuild_vectors",
    )
    logger.info("rebuild_vectors_enqueued", doc_id=doc_id, task_id=task.id)
    return task.id


# ============================================================
# 文档重处理任务
# ============================================================

@celery_app.task(bind=True, name="tasks.reprocess_document")
def reprocess_document_task(self, doc_id: str):
    """对已有文档重新处理（删除旧切片 → 重新解析 → 分块 → 向量化）。

    需要原始文件仍存在于 uploads 目录下。
    """
    from app.repositories.chunk_repo import chunk_repo
    from app.milvus_repo import milvus_repo

    task_id = self.request.id
    logger.info("reprocess_document_started", task_id=task_id, doc_id=doc_id)

    try:
        # 查找文档文件
        file_path = _find_doc_file(doc_id)
        if not file_path:
            raise Exception(f"Original file not found for doc_id={doc_id}")

        # 删除旧切片（MySQL + Milvus）
        old_chunks = chunk_repo.get_all_by_doc(doc_id)
        chunk_ids = [c.chunk_id for c in old_chunks]

        # 获取 collection_name
        collection_name = settings.MILVUS_COLLECTION_NAME
        if old_chunks and old_chunks[0].kb_id:
            from app.repositories.kb_repo import kb_repo
            kb = kb_repo.get_by_id(int(old_chunks[0].kb_id))
            if kb:
                collection_name = kb.collection_name

        try:
            milvus_repo.delete(chunk_ids, collection_name=collection_name)
        except Exception:
            logger.warning("reprocess_milvus_delete_failed", doc_id=doc_id)
        chunk_repo.batch_delete(chunk_ids)
        logger.info("reprocess_old_chunks_deleted", doc_id=doc_id, count=len(chunk_ids))

        # 重置任务状态
        task_repo.upsert(task_id, status="processing", progress=0, stage="queued", doc_id=doc_id)

        # 重新走解析流水线
        result = parse_and_chunk_task(file_path, doc_id)
        logger.info("reprocess_document_completed", task_id=task_id, doc_id=doc_id)
        return result

    except Exception as exc:
        logger.exception("reprocess_document_failed", task_id=task_id, doc_id=doc_id, error=str(exc))
        task_repo.upsert(task_id, status="failed", stage="error", error_message=str(exc))
        raise


def enqueue_reprocess_document(doc_id: str) -> str:
    """提交文档重处理任务。"""
    task = reprocess_document_task.apply_async(args=[doc_id])
    task_repo.upsert(
        task.id,
        status="pending",
        doc_id=doc_id,
        stage="reprocess",
        task_type="reprocess_document",
    )
    logger.info("reprocess_document_enqueued", doc_id=doc_id, task_id=task.id)
    return task.id


# ============================================================
# URL 文档抓取任务
# ============================================================

@celery_app.task(bind=True, name="tasks.fetch_url_document")
def fetch_url_document_task(self, url: str, kb_id: str = "",
                            doc_name: str = "") -> dict:
    """从 URL 下载文档并提交处理。

    下载到 uploads 目录后，调用 enqueue_parse 走标准流水线。
    """
    import requests
    import uuid as uuid_mod
    from urllib.parse import urlparse

    task_id = self.request.id
    logger.info("url_fetch_started", task_id=task_id, url=url)

    try:
        self.update_state(state="PROGRESS", meta={"stage": "fetching", "progress": 10})

        # 下载文件
        resp = requests.get(url, timeout=120, stream=True,
                            headers={"User-Agent": "RAG-Chunking-Service/1.0"})
        resp.raise_for_status()

        # 从 URL 提取文件名
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path) or "downloaded_file"
        doc_id = str(uuid_mod.uuid4())

        # 确定扩展名
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext not in settings.ALLOWED_EXTENSIONS:
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" in content_type:
                ext = "pdf"
            elif "word" in content_type or "docx" in content_type:
                ext = "docx"
            elif "presentation" in content_type or "pptx" in content_type:
                ext = "pptx"
            else:
                ext = "pdf"
            filename = f"{filename}.{ext}"

        file_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}.{ext}")

        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = int(10 + (downloaded / total_size) * 30)  # 10~40%
                        self.update_state(state="PROGRESS",
                                          meta={"stage": "fetching", "progress": progress})

        file_size = os.path.getsize(file_path)
        if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            os.unlink(file_path)
            raise Exception(f"Downloaded file too large ({file_size} bytes)")

        self.update_state(state="PROGRESS", meta={"stage": "submitting", "progress": 50})

        # 提交 Celery 处理流水线
        final_doc_name = doc_name or filename
        sub_task_id = enqueue_parse(
            file_path, doc_id,
            kb_id=kb_id, source_type="url",
            source_url=url, doc_name=final_doc_name,
        )

        logger.info("url_fetch_completed", task_id=task_id, url=url,
                    doc_id=doc_id, sub_task_id=sub_task_id)
        return {"status": "completed", "doc_id": doc_id, "sub_task_id": sub_task_id}

    except Exception as exc:
        logger.exception("url_fetch_failed", task_id=task_id, url=url, error=str(exc))
        task_repo.upsert(task_id, status="failed", stage="error", error_message=str(exc))
        raise


def enqueue_fetch_url(url: str, kb_id: str = "", doc_name: str = "") -> str:
    """提交 URL 文档抓取任务。"""
    task = fetch_url_document_task.apply_async(args=[url, kb_id, doc_name])
    task_repo.upsert(
        task.id,
        status="pending",
        stage="fetch_url",
        task_type="fetch_url",
        task_args={"url": url, "kb_id": kb_id, "doc_name": doc_name},
        doc_name=doc_name,
    )
    logger.info("url_fetch_enqueued", task_id=task.id, url=url)
    return task.id


def _find_doc_file(doc_id: str) -> str | None:
    """在 uploads 目录查找文档文件（按 doc_id.extension 匹配）。"""
    for ext in settings.ALLOWED_EXTENSIONS:
        candidate = os.path.join(settings.UPLOAD_DIR, f"{doc_id}.{ext}")
        if os.path.isfile(candidate):
            return candidate
    # 也搜索所有以 doc_id 开头的文件
    try:
        for fname in os.listdir(settings.UPLOAD_DIR):
            if fname.startswith(doc_id):
                return os.path.join(settings.UPLOAD_DIR, fname)
    except Exception:
        pass
    return None
