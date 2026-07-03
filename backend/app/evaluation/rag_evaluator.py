"""RAG 检索质量评估器：离线评估（人工标注测试集）。"""

import time
from dataclasses import dataclass, field

import structlog

from app.services.search_service import search_service

logger = structlog.get_logger()


@dataclass
class EvaluationMetrics:
    """单轮评估结果指标。"""
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    hit_rate: float = 0.0
    latency_ms: float = 0.0


@dataclass
class TestCase:
    """单条测试用例。

    Attributes:
        query: 查询文本
        relevant_chunk_ids: 人工标注的相关 chunk_id 集合
        doc_id: 限定在哪个文档内搜索（可选）
        tags: 用例标签，如 "表格查询", "图片查询", "代码查询"
    """
    query: str
    relevant_chunk_ids: list[str]
    doc_id: str = ""
    tags: list[str] = field(default_factory=list)


class RAGEvaluator:
    """RAG 检索质量评估器。

    通过人工标注的测试集（100-500 条 query），定期评估检索质量，
    计算 Recall@K, Precision@K, MRR, NDCG@K 等指标。

    用法:
        evaluator = RAGEvaluator(test_cases)
        report = evaluator.evaluate(top_k=10)
        print(f"Recall@10: {report['recall_at_10']:.3f}")
    """

    def __init__(self, test_cases: list[TestCase]):
        """
        Args:
            test_cases: 人工标注的测试用例列表。
        """
        self.test_cases = test_cases
        logger.info(
            "evaluator_initialized",
            num_test_cases=len(test_cases),
            tagged_cases=sum(1 for tc in test_cases if tc.tags),
        )

    def evaluate(
        self,
        collection_name: str,
        top_k: int = 10,
        per_tag: bool = True,
        doc_id_filter: str | None = None,
    ) -> dict:
        """
        对全部测试用例执行评估，返回综合指标报告。

        Args:
            collection_name: 必传：指定 Milvus collection
            top_k: 取前 K 个结果计算指标
            per_tag: 是否按 tag 分组输出指标
            doc_id_filter: 限定在某个 doc_id 内评估（默认评估全部）

        Returns:
            {
                "recall_at_k": float,
                "precision_at_k": float,
                "mrr": float,
                "ndcg_at_k": float,
                "hit_rate": float,
                "avg_latency_ms": float,
                "num_queries": int,
                "per_tag": { "标签名": {指标...}, ... },  # per_tag=True 时
            }
        """
        cases = self.test_cases
        if doc_id_filter:
            cases = [tc for tc in cases if tc.doc_id == doc_id_filter]

        if not cases:
            return {"error": "no test cases", "num_queries": 0}

        # 全局指标聚合
        agg = {"recall": [], "precision": [], "rr": [], "ndcg": [], "hit": [], "latency": []}

        # 按 tag 分组聚合
        tag_agg: dict[str, dict[str, list]] = {}

        for idx, case in enumerate(cases):
            t0 = time.monotonic()
            metrics = self._evaluate_case(case, top_k, collection_name)
            latency_ms = (time.monotonic() - t0) * 1000

            agg["recall"].append(metrics.recall_at_k)
            agg["precision"].append(metrics.precision_at_k)
            agg["rr"].append(metrics.mrr)
            agg["ndcg"].append(metrics.ndcg_at_k)
            agg["hit"].append(metrics.hit_rate)
            agg["latency"].append(latency_ms)

            # 按 tag 聚合
            if per_tag:
                tags = case.tags if case.tags else ["_untagged_"]
                for tag in tags:
                    if tag not in tag_agg:
                        tag_agg[tag] = {"recall": [], "precision": [], "rr": [], "ndcg": [], "hit": [], "latency": []}
                    tag_agg[tag]["recall"].append(metrics.recall_at_k)
                    tag_agg[tag]["precision"].append(metrics.precision_at_k)
                    tag_agg[tag]["rr"].append(metrics.mrr)
                    tag_agg[tag]["ndcg"].append(metrics.ndcg_at_k)
                    tag_agg[tag]["hit"].append(metrics.hit_rate)
                    tag_agg[tag]["latency"].append(latency_ms)

            if (idx + 1) % 20 == 0:
                logger.info("evaluation_progress", completed=idx + 1, total=len(cases))

        # 构建报告
        report = self._aggregate(agg)
        report["num_queries"] = len(cases)
        report["avg_latency_ms"] = round(sum(agg["latency"]) / len(agg["latency"]), 2)

        if per_tag and tag_agg:
            report["per_tag"] = {tag: self._aggregate(vals) for tag, vals in tag_agg.items()}

        logger.info(
            "evaluation_completed",
            num_queries=report["num_queries"],
            recall_at_k=report["recall_at_k"],
            precision_at_k=report["precision_at_k"],
            mrr=report["mrr"],
            ndcg_at_k=report["ndcg_at_k"],
        )
        return report

    def evaluate_batch(
        self,
        top_k_values: list[int] = None,
    ) -> list[dict]:
        """
        对多个 top_k 值批量评估。

        Args:
            top_k_values: 要评估的 top_k 值列表，默认 [1, 3, 5, 10, 20]

        Returns:
            每个 top_k 对应的指标列表
        """
        if top_k_values is None:
            top_k_values = [1, 3, 5, 10, 20]
        return [self.evaluate(top_k=k, per_tag=False) for k in top_k_values]

    # ============================================================
    # 内部方法
    # ============================================================

    @staticmethod
    def _evaluate_case(case: TestCase, top_k: int, collection_name: str) -> EvaluationMetrics:
        """评估单条测试用例。"""
        relevant_ids = set(case.relevant_chunk_ids)

        # 执行检索（使用传入的 collection_name，评估时不启用 rerank）
        hits = search_service.search(
            query=case.query,
            top_k=top_k,
            doc_ids=[case.doc_id] if case.doc_id else None,
            use_rerank=False,
            collection_name=collection_name,
        )
        retrieved_ids = [h["chunk"]["chunk_id"] for h in hits]

        metrics = EvaluationMetrics()

        if not relevant_ids:
            metrics.recall_at_k = 1.0  # 空标注视为全部命中
            metrics.hit_rate = 1.0
            return metrics

        # Recall@K
        hit_ids = relevant_ids & set(retrieved_ids)
        metrics.recall_at_k = len(hit_ids) / len(relevant_ids)

        # Precision@K
        metrics.precision_at_k = len(hit_ids) / len(retrieved_ids) if retrieved_ids else 0

        # MRR (Mean Reciprocal Rank)
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                metrics.mrr = 1.0 / rank
                break

        # NDCG@K
        dcg = sum(
            (1.0 if rid in relevant_ids else 0.0) / (i + 1)
            for i, rid in enumerate(retrieved_ids)
        )
        ideal_len = min(len(relevant_ids), top_k)
        idcg = sum(1.0 / (i + 1) for i in range(ideal_len)) if ideal_len > 0 else 1.0
        metrics.ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

        # Hit Rate（至少命中一个）
        metrics.hit_rate = 1.0 if hit_ids else 0.0

        return metrics

    @staticmethod
    def _aggregate(vals: dict[str, list[float]]) -> dict:
        """计算指标均值。"""
        result = {}
        key_map = {
            "recall": "recall_at_k",
            "precision": "precision_at_k",
            "rr": "mrr",
            "ndcg": "ndcg_at_k",
            "hit": "hit_rate",
            "latency": "avg_latency_ms",
        }
        for k, out_key in key_map.items():
            items = vals.get(k, [])
            if items:
                result[out_key] = round(sum(items) / len(items), 4)
            else:
                result[out_key] = 0.0
        return result
