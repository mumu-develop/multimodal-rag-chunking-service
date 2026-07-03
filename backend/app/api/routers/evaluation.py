"""Evaluation router: 离线评估 + 消融实验 + 线上反馈。"""
import structlog
from fastapi import APIRouter, Request

from app.schemas import (
    RunEvaluationRequest,
    RunEvaluationResponse,
    RunAblationRequest,
    RunAblationResponse,
    FeedbackRequest,
    FeedbackStatsResponse,
    DatasetStatsResponse,
    LoadRagasDatasetRequest,
    LoadRagasDatasetResponse,
    RunRagasEvaluationRequest,
    RunRagasEvaluationResponse,
    RagasDatasetStatsResponse,
    GenerateRagasTestsetRequest,
    GenerateRagasTestsetResponse,
    TestSetListResponse,
    TestSetBatchListResponse,
    SaveEvaluationRequest,
    EvaluationRecordListResponse,
)
from app.evaluation import RAGEvaluator, TestDatasetBuilder, AblationStudy, FeedbackCollector
from app.evaluation import RagasEvaluator, RagasTestCase

logger = structlog.get_logger()

router = APIRouter()

# 全局单例（生产环境应持久化到数据库）
_evaluator: RAGEvaluator | None = None
_ablation: AblationStudy | None = None
_builder = TestDatasetBuilder()
_feedback = FeedbackCollector()

# RAGAS 评估器单例
_ragas_cases: list[RagasTestCase] = []


# ============================================================
# 数据集管理
# ============================================================

@router.get("/datasets/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats():
    """获取测试集统计信息。"""
    stats = _builder.stats()
    return _ok(trace_id, data={
        "total": stats.get("total", 0),
        "avg_relevant_chunks": stats.get("avg_relevant_chunks", 0),
        "by_tag": stats.get("by_tag", {}),
        "by_doc": stats.get("by_doc", {}),
    })


# ============================================================
# 离线评估
# ============================================================

@router.post("/run", response_model=RunEvaluationResponse)
async def run_evaluation(body: RunEvaluationRequest, request: Request):
    """执行离线评估，计算 Recall@K, Precision@K, MRR, NDCG@K 等指标。

    需要预先加载测试集（通过 `/api/v1/evaluate/datasets/load`）。
    """
    trace_id = _get_trace_id(request)

    if not _evaluator or not _evaluator.test_cases:
        return _err(
            trace_id=trace_id,
            message="No test cases loaded. Load a dataset first via /datasets/load.",
            code="NO_TEST_CASES",
        )

    report = _evaluator.evaluate(
        top_k=body.top_k,
        per_tag=body.per_tag,
        doc_id_filter=body.doc_id_filter,
    )

    if "error" in report:
        return _err(trace_id=trace_id, message=report["error"], code="EVALUATION_ERROR")

    return _ok(trace_id, data={
        "num_queries": report["num_queries"],
        "recall_at_k": report["recall_at_k"],
        "precision_at_k": report["precision_at_k"],
        "mrr": report["mrr"],
        "ndcg_at_k": report["ndcg_at_k"],
        "hit_rate": report["hit_rate"],
        "avg_latency_ms": report["avg_latency_ms"],
        "per_tag": report.get("per_tag"),
    })


# ============================================================
# 消融实验
# ============================================================

@router.post("/ablation/run", response_model=RunAblationResponse)
async def run_ablation(body: RunAblationRequest, request: Request):
    """执行消融实验，对比 full / no_rerank / no_hybrid / no_rerank_no_hybrid 配置。"""
    trace_id = _get_trace_id(request)

    if not _ablation or not _ablation.test_cases:
        return _err(
            trace_id=trace_id,
            message="No test cases loaded. Load a dataset first.",
            code="NO_TEST_CASES",
        )

    report = _ablation.run(top_k=body.top_k)

    def _to_config_result(r: dict) -> dict:
        return {
            "recall_at_k": r.get("recall_at_k", 0),
            "precision_at_k": r.get("precision_at_k", 0),
            "mrr": r.get("mrr", 0),
            "ndcg_at_k": r.get("ndcg_at_k", 0),
            "num_queries": r.get("num_queries", 0),
        }

    comp = report.get("comparison", {})
    return _ok(trace_id, data={
        "full": _to_config_result(report.get("full", {})),
        "no_rerank": _to_config_result(report.get("no_rerank", {})),
        "no_hybrid": _to_config_result(report.get("no_hybrid", {})),
        "no_rerank_no_hybrid": _to_config_result(report.get("no_rerank_no_hybrid", {})),
        "comparison": {
            "rerank_recall_gain": comp.get("rerank_recall_gain", 0),
            "rerank_ndcg_gain": comp.get("rerank_ndcg_gain", 0),
            "hybrid_recall_gain": comp.get("hybrid_recall_gain", 0),
            "hybrid_ndcg_gain": comp.get("hybrid_ndcg_gain", 0),
        },
    })


# ============================================================
# 线上反馈
# ============================================================

@router.post("/feedback", response_model=FeedbackStatsResponse)
async def submit_feedback(body: FeedbackRequest, request: Request):
    """提交用户反馈（点击行为）。"""
    trace_id = _get_trace_id(request)

    _feedback.record(
        query=body.query,
        top_result_ids=body.top_result_ids,
        clicked_ids=body.clicked_ids,
        dwell_times_ms=body.dwell_times_ms,
        doc_id=body.doc_id,
    )

    return _ok(trace_id, data={"success": True})


@router.get("/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(window_minutes: int = 60, request: Request = None):
    """获取线上反馈统计指标。"""
    trace_id = _get_trace_id(request) if request else ""

    metrics = _feedback.get_metrics(window_minutes=window_minutes)
    low_sq = _feedback.get_low_satisfaction_queries(window_minutes=window_minutes)

    return _ok(trace_id, data={
        "metrics": metrics,
        "low_satisfaction_queries": low_sq,
    })


# ============================================================
# RAGAS 评估
# ============================================================

@router.post("/ragas/dataset", response_model=LoadRagasDatasetResponse)
async def load_ragas_dataset(body: LoadRagasDatasetRequest, request: Request):
    """加载 RAGAS 测试集（JSON body 或文件路径）。"""
    trace_id = _get_trace_id(request)
    import uuid
    from app.repositories.testset_repo import testset_repo

    if body.items:
        # 从 JSON body 加载
        set_id = body.set_id or str(uuid.uuid4())
        records = [
            {
                "set_id": set_id,
                "query": item.query,
                "ground_truth": item.ground_truth,
                "doc_id": item.doc_id or "",
                "tags": item.tags or [],
            }
            for item in body.items
        ]
        ok = testset_repo.batch_insert(records)
        if not ok:
            return _err(trace_id, testset_repo.last_error or "插入失败", "DB_ERROR")
        return _ok(trace_id, data={"set_id": set_id, "total": len(records)})
    elif body.file_path:
        # 从文件加载
        import json
        import os
        if not os.path.exists(body.file_path):
            return _err(trace_id, f"文件不存在: {body.file_path}", "FILE_NOT_FOUND")
        try:
            with open(body.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            set_id = body.set_id or str(uuid.uuid4())
            records = []
            for item in data:
                records.append({
                    "set_id": set_id,
                    "query": item.get("query", ""),
                    "ground_truth": item.get("ground_truth", ""),
                    "doc_id": item.get("doc_id", ""),
                    "tags": item.get("tags", []),
                })
            ok = testset_repo.batch_insert(records)
            if not ok:
                return _err(trace_id, testset_repo.last_error or "插入失败", "DB_ERROR")
            return _ok(trace_id, data={"set_id": set_id, "total": len(records)})
        except Exception as e:
            return _err(trace_id, f"解析文件失败: {str(e)}", "PARSE_ERROR")
    else:
        return _err(trace_id, "需要提供 items 或 file_path", "BAD_REQUEST")


# ============================================================
# RAGAS 测试集生成（Synthetic Test Data）
# ============================================================

@router.post("/ragas/testset/generate", response_model=GenerateRagasTestsetResponse)
async def generate_ragas_testset(body: GenerateRagasTestsetRequest, request: Request):
    """从知识库文档自动生成 RAGAS 测试集（异步）。

    method=fast: LLM Prompt 直接生成（几秒~1分钟）
    method=ragas: RAGAS 框架生成（约90分钟）- 暂不支持
    """
    trace_id = _get_trace_id(request)

    # ragas 方式暂不支持
    if body.method == "ragas":
        return _err(trace_id, "暂不支持 ragas 方式，请使用 method=fast", "NOT_SUPPORTED")

    # fast 方式：调用任务队列
    import uuid
    from app.tasks import enqueue_generate_testset
    from app.config import settings

    set_id = str(uuid.uuid4())
    llm_config = {
        "model_name": body.llm_override or settings.EVAL_MODEL_NAME,
        "api_key": settings.EVAL_API_KEY or settings.DASHSCOPE_API_KEY,
        "base_url": settings.EVAL_BASE_URL or settings.LLM_API_URL or settings.VLM_API_URL,
    }
    embed_config = {
        "model_name": settings.EMBED_MODEL_NAME,
        "base_url": settings.EMBED_BASE_URL,
    }

    task_id = enqueue_generate_testset(
        set_id=set_id,
        set_name=body.set_name,
        doc_ids=body.doc_ids,
        testset_size=body.testset_size,
        chunk_type_filter=body.chunk_type_filter,
        llm_config=llm_config,
        embed_config=embed_config,
        method=body.method,
    )

    return _ok(trace_id, data={"task_id": task_id, "set_id": set_id})


@router.get("/ragas/testsets/sets", response_model=TestSetBatchListResponse)
async def list_ragas_sets(
    page: int = 1,
    size: int = 20,
    request: Request = None,
):
    """分页查询历史生成的 RAGAS 测试集批次列表（按 set_id 聚合）。"""
    trace_id = _get_trace_id(request)
    from app.models import TestSet, get_session
    from sqlalchemy import func as sql_func

    session = get_session()
    try:
        # 按 set_id 聚合查询
        q = session.query(
            TestSet.set_id,
            TestSet.set_name,
            sql_func.count(TestSet.id).label("total"),
            sql_func.min(TestSet.gmt_create).label("created_at"),
        ).group_by(TestSet.set_id, TestSet.set_name).order_by(TestSet.set_id.desc())

        total = q.count()
        items = q.offset((page - 1) * size).limit(size).all()

        result = [
            {
                "set_id": row.set_id,
                "set_name": row.set_name,
                "total": row.total,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            }
            for row in items
        ]

        return _ok(trace_id, data={"total": total, "page": page, "result": result})
    finally:
        session.close()


@router.get("/ragas/testsets/names")
async def list_testset_names(request: Request):
    """查询所有测试集名称列表（供前端下拉选择使用）。"""
    trace_id = _get_trace_id(request)
    from app.repositories.testset_repo import testset_repo

    result = testset_repo.list_set_names()
    return _ok(trace_id, data=result)


@router.get("/ragas/testsets", response_model=TestSetListResponse)
async def list_ragas_testsets(
    page: int = 1,
    size: int = 20,
    set_id: str | None = None,
    doc_id: str | None = None,
    set_name: str | None = None,
    request: Request = None,
):
    """分页查询历史生成的 RAGAS 测试集列表。"""
    trace_id = _get_trace_id(request)
    from app.repositories.testset_repo import testset_repo

    total, items = testset_repo.list_sets(
        page=page,
        size=size,
        set_id_filter=set_id,
        doc_id_filter=doc_id,
        set_name_filter=set_name,
    )

    result = [
        {
            "id": item.id,
            "set_id": item.set_id,
            "set_name": item.set_name,
            "query": item.query,
            "ground_truth": item.ground_truth,
            "doc_id": item.doc_id,
            "tags": item.tags or [],
            "gmt_create": item.gmt_create.isoformat() if item.gmt_create else "",
        }
        for item in items
    ]

    return _ok(trace_id, data={"total": total, "page": page, "result": result})


@router.get("/ragas/datasets/stats", response_model=RagasDatasetStatsResponse)
async def ragas_dataset_stats(request: Request):
    """获取 RAGAS 测试集统计信息。"""
    trace_id = _get_trace_id(request)
    from app.models import TestSet, get_session
    from sqlalchemy import func as sql_func

    session = get_session()
    try:
        # 总测试用例数
        total_cases = session.query(sql_func.count(TestSet.id)).scalar() or 0

        # 按 set_id 聚合统计
        set_stats = session.query(
            TestSet.set_id,
            sql_func.count(TestSet.id).label("count"),
        ).group_by(TestSet.set_id).all()

        sets = [
            {"set_id": row.set_id, "count": row.count}
            for row in set_stats
        ]

        return _ok(trace_id, data={
            "total_cases": total_cases,
            "total_sets": len(sets),
            "sets": sets,
        })
    finally:
        session.close()


@router.post("/ragas/run", response_model=RunRagasEvaluationResponse)
async def run_ragas_evaluation(body: RunRagasEvaluationRequest, request: Request):
    """执行 RAGAS Context Precision/Recall 评估。"""
    trace_id = _get_trace_id(request)
    from app.repositories.testset_repo import testset_repo
    from app.repositories.eval_record_repo import eval_record_repo
    from app.evaluation.ragas_evaluator import RagasEvaluator, RagasTestCase
    from app.config import settings

    # 获取测试集（支持 set_id 或 set_name）
    set_id = body.set_id
    set_name = body.set_name

    if not set_id and not set_name:
        return _err(trace_id, "需要提供 set_id 或 set_name", "BAD_REQUEST")

    # 根据 set_id 或 set_name 查询测试集
    if set_id:
        total, db_records = testset_repo.list_sets(set_id_filter=set_id)
    else:
        total, db_records = testset_repo.list_sets(set_name_filter=set_name)

    if total == 0:
        return _err(trace_id, f"测试集不存在: {set_name or set_id}", "NO_TEST_CASES")

    # 从第一条记录获取真实的 set_id（用于保存记录）
    actual_set_id = db_records[0].set_id if db_records else ""

    # 转换数据库记录为 RagasTestCase 对象
    test_cases = [
        RagasTestCase(
            query=r.query,
            ground_truth=r.ground_truth,
            doc_id=r.doc_id or "",
            tags=r.tags or [],
        )
        for r in db_records
    ]

    if not test_cases:
        return _err(trace_id, "测试集为空", "NO_TEST_CASES")

    # 构建 LLM 配置
    llm_config = {
        "model_name": body.llm_override or settings.EVAL_MODEL_NAME,
        "api_key": settings.EVAL_API_KEY or settings.DASHSCOPE_API_KEY,
        "base_url": settings.EVAL_BASE_URL or settings.LLM_API_URL or settings.VLM_API_URL,
    }

    # 构建 RAGAS 评估器
    evaluator = RagasEvaluator(test_cases, eval_llm_config=llm_config)

    try:
        metrics = evaluator.evaluate(
            collection_name=body.collection_name,
            top_k=body.top_k,
            per_tag=body.per_tag,
            metrics=body.metrics,
            generate_answer=body.generate_answer,
        )

        # 自动保存评估记录到数据库（只要有 set_id 就保存）
        if body.save_record and actual_set_id:
            # 保存评估结果和评估配置
            record_data = {
                "kb_id": body.kb_id or "",
                "doc_id": body.doc_id or "",
                "set_id": actual_set_id,
                "metrics": {
                    # 评估配置
                    "config": {
                        "top_k": body.top_k,
                        "per_tag": body.per_tag,
                        "metrics_requested": body.metrics or ["context_precision", "context_recall"],
                        "generate_answer": body.generate_answer,
                        "llm_override": body.llm_override,
                        "set_name": set_name,
                    },
                    # 评估结果
                    "result": metrics,
                },
            }
            ok = eval_record_repo.save(record_data)
            if not ok:
                logger.warning("eval_record_save_failed", error=eval_record_repo.last_error)

        return _ok(trace_id, data=metrics)
    except Exception as e:
        logger.exception("ragas_evaluation_failed", error=str(e))
        return _err(trace_id, f"评估失败: {str(e)}", "EVAL_ERROR")


def _ragas_tag_stats() -> dict:
    """RAGAS 测试集按 tag 和 doc 的统计。"""
    if not _ragas_cases:
        return {"by_tag": {}, "by_doc": {}}

    tag_counts: dict[str, int] = {}
    doc_counts: dict[str, int] = {}

    for tc in _ragas_cases:
        for tag in (tc.tags if tc.tags else ["_untagged_"]):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if tc.doc_id:
            doc_counts[tc.doc_id] = doc_counts.get(tc.doc_id, 0) + 1

    return {"by_tag": tag_counts, "by_doc": doc_counts}


# ============================================================
# 内部方法
# ============================================================

def get_evaluator() -> RAGEvaluator | None:
    return _evaluator


def set_evaluator(evaluator: RAGEvaluator) -> None:
    """设置全局评估器。"""
    global _evaluator, _ablation
    _evaluator = evaluator
    _ablation = AblationStudy(evaluator.test_cases)


def get_feedback_collector() -> FeedbackCollector:
    return _feedback


def load_dataset_from_json(file_path: str) -> int:
    """从 JSON 文件加载测试集并初始化评估器。"""
    cases = _builder.load_from_json(file_path)
    _builder.deduplicate()
    if cases:
        set_evaluator(RAGEvaluator(_builder.cases))
    return len(_builder.cases)


def _get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "")


def _ok(trace_id: str = "", data=None, code: str = "OK"):
    # 如果 data 是 Pydantic 模型，转换为 dict
    if data is not None and hasattr(data, 'model_dump'):
        data = data.model_dump()
    elif data is not None and hasattr(data, 'dict'):
        data = data.dict()
    return {"success": True, "code": code, "message": None, "trace_id": trace_id, "data": data}


def _err(trace_id: str = "", message: str = "Error", code: str = "ERROR"):
    return {"success": False, "code": code, "message": message, "trace_id": trace_id, "data": None}


# ============================================================
# 评估记录管理
# ============================================================

@router.post("/records/save")
async def save_evaluation_record(body: SaveEvaluationRequest, request: Request):
    """保存评估记录到数据库。

    评估完成后，将评估指标持久化到 evaluation_records 表。
    """
    trace_id = _get_trace_id(request)
    from app.repositories.eval_record_repo import eval_record_repo

    ok = eval_record_repo.save({
        "kb_id": body.kb_id,
        "doc_id": body.doc_id,
        "set_id": body.set_id,
        "metrics": body.metrics,
    })
    if not ok:
        return _err(trace_id, eval_record_repo.last_error or "保存失败", "DB_ERROR")

    return _ok(trace_id, data={"success": True})


@router.get("/records", response_model=EvaluationRecordListResponse)
async def list_evaluation_records(
    kb_id: str = None,
    doc_id: str = None,
    set_id: str = None,
    page: int = 1,
    size: int = 20,
    request: Request = None,
):
    """查询评估记录列表。

    以知识库ID、文档ID、测试集批次ID作为查询条件，
    返回知识库名称、文档名称、测试集ID、评估指标的集合。
    """
    trace_id = _get_trace_id(request)
    from app.repositories.eval_record_repo import eval_record_repo

    total, result = eval_record_repo.list(
        kb_id=kb_id,
        doc_id=doc_id,
        set_id=set_id,
        page=page,
        size=size,
    )

    return _ok(trace_id, data={"total": total, "page": page, "result": result})
