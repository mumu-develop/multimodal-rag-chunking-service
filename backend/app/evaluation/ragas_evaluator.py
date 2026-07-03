"""RAGAS 评估器：使用 LLM-as-Judge 评估 RAG 系统全链路质量。

支持的指标：
- Context Precision: 检索内容是否都相关
- Context Recall: ground_truth 内容是否都被检索到
- Faithfulness: 答案是否忠实于检索内容（不编造）
- Answer Relevance: 答案是否真正回答了问题
- Answer Correctness: 答案是否正确

与 RAGEvaluator（基于 chunk_id 精确匹配）不同，RagasEvaluator 只需要
query + ground_truth 答案，无需人工标注 chunk_id。LLM 会自动判断检索
内容是否相关、是否覆盖了答案中的信息。
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock

import structlog

from app.config import settings
from app.services.search_service import search_service

logger = structlog.get_logger()


def _safe_float(val) -> float | None:
    """安全转换数值，NaN/Infinity 返回 None。"""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


@dataclass
class RagasTestCase:
    """单条 RAGAS 测试用例。

    Attributes:
        query: 查询文本
        ground_truth: 期望的答案文本（ground truth）
        doc_id: 限定在哪个文档内搜索（可选）
        tags: 用例标签，如 "财务", "数字查询"
    """

    query: str
    ground_truth: str
    doc_id: str = ""
    tags: list[str] = field(default_factory=list)


class RagasEvaluator:
    """RAGAS 全链路评估器。

    使用 ragas 框架的 LLM-as-Judge 指标，自动判断检索内容的相关性和
    答案覆盖率。无需人工标注 chunk_id，只需准备 query + ground_truth。

    支持的指标：
    - context_precision: 检索精确度
    - context_recall: 检索召回率
    - faithfulness: 答案忠实度（需要生成答案）
    - answer_relevance: 答案相关性（需要生成答案）
    - answer_correctness: 答案正确性（需要生成答案）

    用法:
        cases = [
            RagasTestCase(
                query="公司 2024 年营收多少？",
                ground_truth="公司 2024 年营收 100 亿元",
                doc_id="doc_abc123",
                tags=["财务"],
            ),
        ]
        evaluator = RagasEvaluator(
            cases,
            eval_llm_config={
                "model_name": "gpt-4o-mini",
                "api_key": "sk-xxx",
                "base_url": "https://api.openai.com/v1",
            },
        )
        report = evaluator.evaluate(
            top_k=5,
            collection_name="kb_xxx",
            metrics=["context_precision", "context_recall", "faithfulness"],
            generate_answer=True,  # 是否生成答案（Faithfulness/Answer Relevance 需要）
        )
    """

    def __init__(
        self,
        test_cases: list[RagasTestCase],
        eval_llm_config: dict | None = None,
        gen_llm_config: dict | None = None,  # 用于生成答案的 LLM 配置
    ):
        """
        Args:
            test_cases: 测试用例列表（query + ground_truth）
            eval_llm_config: 评估 LLM 配置
                - model_name: 模型名（默认 gpt-4o-mini）
                - api_key: API Key
                - base_url: API 端点（OpenAI 兼容格式）
            gen_llm_config: 生成答案的 LLM 配置（可选，默认与 eval_llm_config 相同）
        """
        self.test_cases = test_cases
        self._eval_llm_config = eval_llm_config or self._default_llm_config()
        self._gen_llm_config = gen_llm_config or self._eval_llm_config

    def evaluate(
        self,
        collection_name: str,
        top_k: int = 5,
        per_tag: bool = True,
        max_workers: int | None = None,
        metrics: list[str] | None = None,
        generate_answer: bool = False,
    ) -> dict:
        """执行 RAGAS 评估（高并发）。

        Args:
            top_k: 每个 query 取 top_k 个检索结果
            collection_name: 必传：指定 Milvus collection
            per_tag: 是否按 tag 分组输出指标
            max_workers: 并发线程数，默认读取 settings.EVAL_MAX_WORKERS
            metrics: 要计算的指标列表，默认 ["context_precision", "context_recall"]
                可选: "faithfulness", "answer_relevance", "answer_correctness"
            generate_answer: 是否生成答案（Faithfulness/Answer Relevance 需要）

        Returns:
            {
                "context_precision": float,
                "context_recall": float,
                "faithfulness": float,  # 如果 metrics 包含
                "answer_relevance": float,  # 如果 metrics 包含
                "answer_correctness": float,  # 如果 metrics 包含
                "num_queries": int,
                "avg_latency_ms": float,
                "per_tag": { "tag": {...} },
                "details": [ {...} ],
            }
        """
        if not self.test_cases:
            return {"error": "no test cases", "num_queries": 0}

        # 默认指标
        if metrics is None:
            metrics = ["context_precision", "context_recall"]

        # Faithfulness/Answer Relevance 需要生成答案
        need_answer = generate_answer or any(
            m in metrics for m in ["faithfulness", "answer_relevance", "answer_correctness"]
        )

        n = len(self.test_cases)
        workers = max_workers if max_workers is not None else settings.EVAL_MAX_WORKERS
        logger.info(
            "ragas_evaluation_started",
            num_queries=n,
            top_k=top_k,
            max_workers=workers,
            metrics=metrics,
            generate_answer=need_answer,
            model=self._eval_llm_config.get("model_name"),
        )

        # 全局聚合（线程安全）
        agg: dict[str, list] = {m: [] for m in metrics}
        agg["latency"] = []
        tag_agg: dict[str, dict[str, list]] = {}
        details: list[dict] = []
        lock = Lock()
        completed_count = [0]

        def evaluate_case(idx: int, case: RagasTestCase) -> dict:
            t0 = time.monotonic()

            # 检索
            hits = search_service.search(
                query=case.query,
                top_k=top_k,
                doc_ids=[case.doc_id] if case.doc_id else None,
                use_rerank=False,
                collection_name=collection_name,
            )
            contexts = [h["chunk"]["content"] for h in hits]
            if not contexts:
                contexts = [""]

            retrieval_latency = (time.monotonic() - t0) * 1000

            # 生成答案（如果需要）
            answer = ""
            gen_latency = 0.0
            if need_answer:
                gen_t0 = time.monotonic()
                answer = self._generate_answer(case.query, contexts)
                gen_latency = (time.monotonic() - gen_t0) * 1000

            total_latency = retrieval_latency + gen_latency

            # 评分
            scores = {}
            try:
                llm = _build_llm(self._eval_llm_config)
                from ragas import EvaluationDataset, evaluate as ragas_evaluate

                # 构建样本
                sample = {
                    "user_input": case.query,
                    "reference": case.ground_truth,
                    "retrieved_contexts": contexts,
                }
                if answer:
                    sample["response"] = answer

                # 选择指标
                ragas_metrics = []
                for m in metrics:
                    if m == "context_precision":
                        from ragas.metrics._context_precision import ContextPrecision
                        ragas_metrics.append(ContextPrecision(llm=llm))
                    elif m == "context_recall":
                        from ragas.metrics._context_recall import ContextRecall
                        ragas_metrics.append(ContextRecall(llm=llm))
                    elif m == "faithfulness" and answer:
                        from ragas.metrics._faithfulness import Faithfulness
                        ragas_metrics.append(Faithfulness(llm=llm))
                    elif m == "answer_relevance" and answer:
                        from ragas.metrics._answer_relevance import AnswerRelevance
                        ragas_metrics.append(AnswerRelevance(llm=llm))
                    elif m == "answer_correctness" and answer:
                        from ragas.metrics._answer_correctness import AnswerCorrectness
                        ragas_metrics.append(AnswerCorrectness(llm=llm))

                dataset = EvaluationDataset.from_list([sample])
                result = ragas_evaluate(dataset, metrics=ragas_metrics)

                for m in metrics:
                    raw_val = result.scores[0].get(m)
                    safe_val = _safe_float(raw_val)
                    scores[m] = safe_val
                    if safe_val is None:
                        logger.warning(
                            "ragas_score_nan",
                            query=case.query[:50],
                            metric=m,
                            raw_value=str(raw_val),
                            contexts_count=len(contexts),
                            contexts_empty=not any(c.strip() for c in contexts),
                        )
            except Exception as e:
                logger.warning("ragas_score_failed", query=case.query[:50], error=str(e), contexts_count=len(contexts))
                for m in metrics:
                    scores[m] = None

            detail = {
                "query": case.query,
                "ground_truth": case.ground_truth,
                "answer": answer if answer else None,
                "retrieval_latency_ms": round(retrieval_latency, 2),
                "generation_latency_ms": round(gen_latency, 2) if gen_latency else None,
                "total_latency_ms": round(total_latency, 2),
            }
            for m in metrics:
                val = scores.get(m)
                detail[m] = round(val, 4) if val is not None else None

            tags = case.tags if case.tags else ["_untagged_"]

            # 线程安全写入
            with lock:
                for m in metrics:
                    val = scores.get(m)
                    if val is not None:
                        agg[m].append(val)
                agg["latency"].append(total_latency)
                details.append(detail)
                completed_count[0] += 1

                if per_tag:
                    for tag in tags:
                        if tag not in tag_agg:
                            tag_agg[tag] = {m: [] for m in metrics}
                            tag_agg[tag]["latency"] = []
                        for m in metrics:
                            val = scores.get(m)
                            if val is not None:
                                tag_agg[tag][m].append(val)
                        tag_agg[tag]["latency"].append(total_latency)

                if completed_count[0] % 10 == 0 or completed_count[0] == n:
                    logger.info("ragas_evaluation_progress", completed=completed_count[0], total=n)

            return detail

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate_case, idx, case): idx
                for idx, case in enumerate(self.test_cases)
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    idx = futures[future]
                    logger.error("ragas_evaluate_case_failed", idx=idx, error=str(e))

        # 构建报告
        report = {}
        for m in metrics:
            if agg[m]:
                report[m] = round(sum(agg[m]) / len(agg[m]), 4)
            else:
                report[m] = None  # 所有值都是 NaN，无法计算平均值
        report["num_queries"] = n
        report["avg_latency_ms"] = round(sum(agg["latency"]) / n, 2)

        if per_tag and tag_agg:
            report["per_tag"] = {}
            for tag, vals in tag_agg.items():
                tag_report = {}
                for m in metrics:
                    if vals[m]:
                        tag_report[m] = round(sum(vals[m]) / len(vals[m]), 4)
                    else:
                        tag_report[m] = None
                tag_report["num_queries"] = len(vals.get("latency", []))
                report["per_tag"][tag] = tag_report

        report["details"] = details

        logger.info(
            "ragas_evaluation_completed",
            num_queries=n,
            metrics=report,
        )
        return report

    # ============================================================
    # 内部方法
    # ============================================================

    def _generate_answer(self, query: str, contexts: list[str]) -> str:
        """用 LLM 从检索内容生成答案。"""
        import httpx
        from openai import OpenAI

        cfg = self._gen_llm_config
        base_url = _normalize_base_url(cfg.get("base_url", ""))
        transport = httpx.HTTPTransport()
        http_client = httpx.Client(transport=transport)
        client = OpenAI(api_key=cfg.get("api_key"), base_url=base_url, http_client=http_client)

        context_text = "\n\n".join(contexts[:5])  # 取前 5 个 context
        prompt = f"""请根据以下检索到的内容回答问题。如果检索内容中没有相关信息，请说"检索内容中没有相关信息"。

检索内容：
{context_text}

问题：{query}

答案："""

        try:
            response = client.chat.completions.create(
                model=cfg.get("model_name", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("generate_answer_failed", query=query, error=str(e))
            return ""

    def _default_llm_config(self) -> dict:
        """从 settings 获取默认 LLM 配置。"""
        return {
            "model_name": getattr(settings, "EVAL_MODEL_NAME", "gpt-4o-mini"),
            "api_key": getattr(settings, "EVAL_API_KEY", "") or getattr(settings, "DASHSCOPE_API_KEY", ""),
            "base_url": getattr(settings, "EVAL_BASE_URL", "") or getattr(settings, "VLM_API_URL", ""),
        }


def _build_llm(llm_config: dict):
    """创建独立的 LLM 实例（线程安全，每个线程调用一次）。"""
    import httpx
    from openai import OpenAI
    from ragas.llms import llm_factory

    cfg = llm_config
    base_url = _normalize_base_url(cfg.get("base_url", ""))
    # 显式指定 transport 避免 httpx 继承 ALL_PROXY(socks5h) 导致 socksio 缺失报错
    transport = httpx.HTTPTransport()
    http_client = httpx.Client(transport=transport)
    client = OpenAI(api_key=cfg.get("api_key"), base_url=base_url, http_client=http_client)
    return llm_factory(cfg["model_name"], client=client)


def _normalize_base_url(base_url: str) -> str:
    """将 OpenAI 兼容端点标准化，确保 ragas llm_factory 兼容。"""
    base_url = base_url.rstrip("/")
    for suffix in (
        "/v1/chat/completions",
        "/v1/embeddings",
        "/chat/completions",
        "/compatible-mode/v1",
        "/v1",
    ):
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]
            break
    return base_url + "/v1"
