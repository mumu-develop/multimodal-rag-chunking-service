"""Repository: MySQL evaluation_records data access."""
import math
from typing import Optional

import structlog

from app.models import EvaluationRecord, KnowledgeBase, DocChunk, get_session

logger = structlog.get_logger()


def sanitize_json(obj):
    """清理 JSON 数据中的无效值（NaN、Infinity 等）。"""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None  # MySQL JSON 不支持 NaN/Infinity，转为 None
        return obj
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)  # 其他类型转为字符串


class EvaluationRecordRepository:
    """MySQL evaluation_records table repository."""

    def __init__(self):
        self.last_error: str = ""

    def save(self, record: dict) -> bool:
        """保存评估记录。

        Args:
            record: 包含 kb_id, doc_id, set_id, metrics 的字典。

        Returns:
            成功返回 True，失败返回 False。
        """
        session = get_session()
        try:
            # 清理 metrics 中的无效 JSON 值
            clean_metrics = sanitize_json(record["metrics"])

            eval_record = EvaluationRecord(
                kb_id=record["kb_id"],
                doc_id=record["doc_id"],
                set_id=record["set_id"],
                metrics=clean_metrics,
            )
            session.add(eval_record)
            session.commit()
            logger.info(
                "eval_record_insert_ok",
                kb_id=record["kb_id"],
                doc_id=record["doc_id"],
                set_id=record["set_id"],
            )
            self.last_error = ""
            return True
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("eval_record_insert_failed", error=self.last_error)
            return False
        finally:
            session.close()

    def find(
        self,
        kb_id: str,
        doc_id: str,
        set_id: str,
    ) -> Optional[EvaluationRecord]:
        """查询评估记录。

        Args:
            kb_id: 知识库ID
            doc_id: 文档ID
            set_id: 测试集批次ID

        Returns:
            找到返回 EvaluationRecord，否则返回 None。
        """
        session = get_session()
        try:
            return (
                session.query(EvaluationRecord)
                .filter(
                    EvaluationRecord.kb_id == kb_id,
                    EvaluationRecord.doc_id == doc_id,
                    EvaluationRecord.set_id == set_id,
                )
                .first()
            )
        finally:
            session.close()

    def list(
        self,
        kb_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        set_id: Optional[str] = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[int, list[dict]]:
        """分页查询评估记录，返回包含 kb_name 和 doc_name 的详细信息。

        Returns:
            (total_count, [{"kb_name": ..., "doc_name": ..., "set_id": ..., "metrics": ..., "gmt_create": ...}, ...])
        """
        session = get_session()
        try:
            q = session.query(EvaluationRecord)
            if kb_id:
                q = q.filter(EvaluationRecord.kb_id == kb_id)
            if doc_id:
                q = q.filter(EvaluationRecord.doc_id == doc_id)
            if set_id:
                q = q.filter(EvaluationRecord.set_id == set_id)

            total = q.count()
            records = (
                q.order_by(EvaluationRecord.gmt_create.desc())
                .offset((page - 1) * size)
                .limit(size)
                .all()
            )

            # 获取 kb_name 和 doc_name
            result = []
            for r in records:
                kb_name = ""
                doc_name = ""

                # 查询知识库名称
                kb = session.query(KnowledgeBase).filter(KnowledgeBase.id == int(r.kb_id) if r.kb_id.isdigit() else 0).first()
                if kb:
                    kb_name = kb.name
                else:
                    # 尝试按 collection_name 查询（如果 kb_id 存的是 collection_name）
                    kb = session.query(KnowledgeBase).filter(KnowledgeBase.collection_name == r.kb_id).first()
                    if kb:
                        kb_name = kb.name

                # 查询文档名称（从 DocChunk 表获取）
                chunk = session.query(DocChunk).filter(DocChunk.doc_id == r.doc_id).first()
                if chunk and chunk.doc_name:
                    doc_name = chunk.doc_name

                result.append({
                    "kb_name": kb_name,
                    "doc_name": doc_name,
                    "set_id": r.set_id,
                    "metrics": r.metrics or {},
                    "gmt_create": r.gmt_create.isoformat() if r.gmt_create else "",
                })

            return total, result
        finally:
            session.close()


# 单例
eval_record_repo = EvaluationRecordRepository()