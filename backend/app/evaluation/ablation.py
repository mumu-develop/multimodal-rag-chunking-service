"""模块化消融实验框架：逐项关闭模块，评估各组件对检索质量的影响。"""

import structlog

from app.evaluation.rag_evaluator import RAGEvaluator, TestCase, EvaluationMetrics
from app.repositories.milvus_repo import milvus_repo
from app.services.search_service import search_service

logger = structlog.get_logger()

# 支持的消融配置
ABLATION_CONFIGS = {
    "full": {"use_rerank": True, "hybrid": True},
    "no_rerank": {"use_rerank": False, "hybrid": True},
    "no_hybrid": {"use_rerank": True, "hybrid": False},
    "no_rerank_no_hybrid": {"use_rerank": False, "hybrid": False},
}


class AblationStudy:
    """
    模块化消融实验。

    通过逐项关闭模块（reranker、混合检索），量化各组件对检索质量的贡献。

    用法:
        study = AblationStudy(test_cases)
        report = study.run(top_k=10)
        study.print_comparison(report)
    """

    def __init__(self, test_cases: list[TestCase]):
        """
        Args:
            test_cases: 测试用例列表
        """
        self.test_cases = test_cases
        self.evaluator = RAGEvaluator(test_cases)

    def run(self, top_k: int = 10) -> dict:
        """
        执行全部消融配置，返回对比报告。

        Args:
            top_k: 评估的 top_k 值

        Returns:
            {
                "full": {指标...},
                "no_rerank": {指标...},
                "no_hybrid": {指标...},
                "no_rerank_no_hybrid": {指标...},
                "comparison": {增益分析...},
            }
        """
        results = {}
        for config_name, config in ABLATION_CONFIGS.items():
            logger.info("ablation_start", config=config_name)
            report = self._run_config(config_name, config, top_k)
            results[config_name] = report

        results["comparison"] = self._compute_comparison(results)
        logger.info("ablation_done", comparison=results["comparison"])
        return results

    def run_single(
        self,
        config_name: str,
        top_k: int = 10,
    ) -> dict:
        """
        单独执行某个消融配置。

        Args:
            config_name: 配置名（如 "full", "no_rerank"）
            top_k: 评估的 top_k 值
        """
        config = ABLATION_CONFIGS.get(config_name)
        if config is None:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(ABLATION_CONFIGS.keys())}")
        return self._run_config(config_name, config, top_k)

    def print_comparison(self, report: dict) -> str:
        """格式化输出消融实验对比结果。"""
        lines = [
            "=" * 70,
            "消融实验结果对比",
            "=" * 70,
            f"{'配置':<25} {'Recall@K':>10} {'Precision@K':>12} {'MRR':>8} {'NDCG@K':>10}",
            "-" * 70,
        ]

        for config_name in ABLATION_CONFIGS:
            r = report.get(config_name, {})
            lines.append(
                f"{config_name:<25} {r.get('recall_at_k', 0):>10.4f} "
                f"{r.get('precision_at_k', 0):>12.4f} {r.get('mrr', 0):>8.4f} "
                f"{r.get('ndcg_at_k', 0):>10.4f}"
            )

        # 增益分析
        comp = report.get("comparison", {})
        lines.append("-" * 70)
        lines.append("增益分析:")
        lines.append(f"  Reranker Recall 增益:  {comp.get('rerank_recall_gain', 0):+.4f}")
        lines.append(f"  Reranker NDCG 增益:    {comp.get('rerank_ndcg_gain', 0):+.4f}")
        lines.append(f"  Hybrid  Recall 增益:   {comp.get('hybrid_recall_gain', 0):+.4f}")
        lines.append(f"  Hybrid  NDCG 增益:     {comp.get('hybrid_ndcg_gain', 0):+.4f}")
        lines.append("=" * 70)

        output = "\n".join(lines)
        logger.info("ablation_report", report=output)
        return output

    # ============================================================
    # 内部方法
    # ============================================================

    def _run_config(self, config_name: str, config: dict, top_k: int) -> dict:
        """执行单个消融配置。"""
        results: list[dict] = []

        for case in self.test_cases:
            relevant_ids = set(case.relevant_chunk_ids)

            # 执行检索（根据配置调整参数）
            hits = search_service.search(
                query=case.query,
                top_k=top_k,
                doc_ids=[case.doc_id] if case.doc_id else None,
                use_rerank=config["use_rerank"],
            )
            retrieved_ids = [h["chunk"]["chunk_id"] for h in hits]

            # 计算单条指标
            result = self._compute_single_metrics(relevant_ids, retrieved_ids, top_k)
            results.append(result)

        # 聚合
        return self._aggregate_results(results)

    @staticmethod
    def _compute_single_metrics(relevant_ids: set, retrieved_ids: list, top_k: int) -> dict:
        """计算单条测试用例指标。"""
        if not relevant_ids:
            return {"recall": 1.0, "precision": 1.0, "rr": 1.0, "ndcg": 1.0}

        hit_ids = relevant_ids & set(retrieved_ids)
        recall = len(hit_ids) / len(relevant_ids)
        precision = len(hit_ids) / len(retrieved_ids) if retrieved_ids else 0

        # MRR
        rr = 0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                rr = 1.0 / rank
                break

        # NDCG@K
        dcg = sum(
            (1.0 if rid in relevant_ids else 0.0) / (i + 1)
            for i, rid in enumerate(retrieved_ids)
        )
        ideal_len = min(len(relevant_ids), top_k)
        idcg = sum(1.0 / (i + 1) for i in range(ideal_len)) if ideal_len > 0 else 1.0
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {"recall": recall, "precision": precision, "rr": rr, "ndcg": ndcg}

    @staticmethod
    def _aggregate_results(results: list[dict]) -> dict:
        """聚合多条测试用例结果。"""
        if not results:
            return {"recall_at_k": 0, "precision_at_k": 0, "mrr": 0, "ndcg_at_k": 0, "num_queries": 0}
        return {
            "recall_at_k": round(sum(r["recall"] for r in results) / len(results), 4),
            "precision_at_k": round(sum(r["precision"] for r in results) / len(results), 4),
            "mrr": round(sum(r["rr"] for r in results) / len(results), 4),
            "ndcg_at_k": round(sum(r["ndcg"] for r in results) / len(results), 4),
            "num_queries": len(results),
        }

    @staticmethod
    def _compute_comparison(results: dict) -> dict:
        """计算增益对比。"""
        full = results.get("full", {})
        no_rerank = results.get("no_rerank", {})
        no_hybrid = results.get("no_hybrid", {})

        return {
            "rerank_recall_gain": round(
                full.get("recall_at_k", 0) - no_rerank.get("recall_at_k", 0), 4
            ),
            "rerank_ndcg_gain": round(
                full.get("ndcg_at_k", 0) - no_rerank.get("ndcg_at_k", 0), 4
            ),
            "hybrid_recall_gain": round(
                full.get("recall_at_k", 0) - no_hybrid.get("recall_at_k", 0), 4
            ),
            "hybrid_ndcg_gain": round(
                full.get("ndcg_at_k", 0) - no_hybrid.get("ndcg_at_k", 0), 4
            ),
        }
