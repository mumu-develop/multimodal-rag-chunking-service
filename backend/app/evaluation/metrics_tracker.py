"""RAGAS 指标追踪器。

功能：
1. 持续记录每次评估的指标
2. 分析指标趋势和变化
3. 自动报警（指标低于阈值）
4. 生成优化建议

使用示例：
    tracker = MetricsTracker()

    # 记录评估结果
    tracker.record({
        "context_precision": 0.85,
        "context_recall": 0.90,
        "faithfulness": 0.88,
        "answer_relevance": 0.92,
    })

    # 获取趋势分析
    trend = tracker.get_trend("context_recall")

    # 检查是否达标
    if tracker.is_target_met():
        print("All metrics meet target!")
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from app.config import settings

logger = structlog.get_logger()


# 目标指标阈值
TARGET_METRICS = {
    "context_precision": 0.95,
    "context_recall": 0.95,
    "faithfulness": 0.95,
    "answer_relevance": 0.95,
    "answer_correctness": 0.95,
}


@dataclass
class MetricRecord:
    """单次评估记录。

    Attributes:
        timestamp: 记录时间戳
        kb_id: 知识库 ID
        doc_id: 文档 ID（可选）
        strategy: 切分策略
        config: 配置参数
        metrics: 指标值字典
        num_queries: 查询数量
        latency_ms: 平均延迟
    """
    timestamp: str
    kb_id: str = ""
    doc_id: str = ""
    strategy: str = "hierarchical"
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    num_queries: int = 0
    latency_ms: float = 0.0


class MetricsTracker:
    """RAGAS 指标追踪器。

    功能：
    1. 记录评估结果
    2. 存储历史数据
    3. 分析趋势
    4. 报警机制

    Attributes:
        _records: 评估记录列表
        _storage_path: 存储路径
        _targets: 目标阈值
    """

    _records: list[MetricRecord] = []
    _storage_path: str = ""
    _targets: dict[str, float] = TARGET_METRICS

    def __init__(self, storage_path: str | None = None):
        """初始化追踪器。

        Args:
            storage_path: 存储路径（默认为 settings.UPLOAD_DIR/metrics）
        """
        if storage_path:
            self._storage_path = storage_path
        else:
            self._storage_path = os.path.join(settings.UPLOAD_DIR, "metrics_history.json")

        # 加载历史数据
        self._load_history()

    def record(
        self,
        metrics: dict,
        kb_id: str = "",
        doc_id: str = "",
        strategy: str = "hierarchical",
        config: dict | None = None,
        num_queries: int = 0,
        latency_ms: float = 0.0,
    ) -> MetricRecord:
        """记录评估结果。

        Args:
            metrics: 指标值字典
            kb_id: 知识库 ID
            doc_id: 文档 ID
            strategy: 切分策略
            config: 配置参数
            num_queries: 查询数量
            latency_ms: 平均延迟

        Returns:
            MetricRecord: 创建的记录
        """
        record = MetricRecord(
            timestamp=datetime.now().isoformat(),
            kb_id=kb_id,
            doc_id=doc_id,
            strategy=strategy,
            config=config or {},
            metrics=metrics,
            num_queries=num_queries,
            latency_ms=latency_ms,
        )

        self._records.append(record)
        self._save_history()

        logger.info(
            "metrics_recorded",
            kb_id=kb_id,
            strategy=strategy,
            metrics=metrics,
            num_queries=num_queries,
        )

        # 检查报警
        self._check_alerts(metrics)

        return record

    def get_trend(self, metric_name: str, limit: int = 10) -> list[tuple[str, float]]:
        """获取指标趋势。

        Args:
            metric_name: 指标名称
            limit: 返回记录数量

        Returns:
            list of (timestamp, value): 时间戳和指标值列表
        """
        trend = []
        for record in self._records[-limit:]:
            value = record.metrics.get(metric_name)
            if value is not None:
                trend.append((record.timestamp, value))
        return trend

    def get_average(self, metric_name: str, limit: int = 10) -> Optional[float]:
        """获取指标平均值。

        Args:
            metric_name: 指标名称
            limit: 计算记录数量

        Returns:
            平均值或 None
        """
        values = []
        for record in self._records[-limit:]:
            value = record.metrics.get(metric_name)
            if value is not None:
                values.append(value)

        if values:
            return sum(values) / len(values)
        return None

    def is_target_met(self, metric_name: str | None = None) -> bool:
        """检查是否达标。

        Args:
            metric_name: 指标名称（可选，不传则检查所有指标）

        Returns:
            True 如果达标
        """
        if metric_name:
            avg = self.get_average(metric_name)
            if avg is None:
                return False
            return avg >= self._targets.get(metric_name, 0.95)

        # 检查所有指标
        for name, target in self._targets.items():
            avg = self.get_average(name)
            if avg is None or avg < target:
                return False
        return True

    def get_gap(self, metric_name: str) -> Optional[float]:
        """获取指标与目标的差距。

        Args:
            metric_name: 指标名称

        Returns:
            差距值（目标 - 当前平均）或 None
        """
        avg = self.get_average(metric_name)
        if avg is None:
            return None
        target = self._targets.get(metric_name, 0.95)
        return target - avg

    def get_recommendations(self) -> list[dict]:
        """获取优化建议。

        Returns:
            建议列表 [{"metric": "...", "gap": float, "suggestions": [...]}, ...]
        """
        recommendations = []

        for metric_name, target in self._targets.items():
            avg = self.get_average(metric_name)
            if avg is None:
                continue

            gap = target - avg
            if gap <= 0:
                continue  # 已达标

            suggestion = {
                "metric": metric_name,
                "current_avg": round(avg, 4),
                "target": target,
                "gap": round(gap, 4),
                "suggestions": self._get_suggestions(metric_name, gap),
            }
            recommendations.append(suggestion)

        return recommendations

    def _get_suggestions(self, metric_name: str, gap: float) -> list[str]:
        """获取指标优化建议。

        Args:
            metric_name: 指标名称
            gap: 差距值

        Returns:
            建议列表
        """
        suggestions_map = {
            "context_precision": [
                "提高重排序模型质量（使用更强的 reranker）",
                "增加 RERANK_TOP_N（如 50 → 100）",
                "调整混合检索权重（dense vs BM25）",
                "添加 Query 扩展（同义词、相关词）",
                "降低检索数量（减少噪声）",
            ],
            "context_recall": [
                "使用 SemanticChunker（保持语义连贯）",
                "减小切分粒度（更小的 chunk）",
                "增加 chunk_overlap（防止边界信息丢失）",
                "为表格/图片添加 caption",
                "确保标题路径正确继承",
                "检查是否过滤了有效内容",
            ],
            "faithfulness": [
                "优化答案生成 Prompt",
                "增加 context 数量",
                "添加事实校验机制",
                "使用更强的生成模型",
            ],
            "answer_relevance": [
                "优化 Query 理解",
                "添加意图识别",
                "使用多轮检索",
                "增加 context 相关性",
            ],
            "answer_correctness": [
                "使用更强的生成模型",
                "添加答案验证机制",
                "参考多个 context",
            ],
        }

        return suggestions_map.get(metric_name, ["需要进一步分析"])

    def _check_alerts(self, metrics: dict) -> None:
        """检查报警。

        Args:
            metrics: 当前指标值
        """
        for metric_name, target in self._targets.items():
            value = metrics.get(metric_name)
            if value is None:
                continue

            # 严重低于目标（差距 > 10%）
            if value < target - 0.10:
                logger.warning(
                    "metric_severely_below_target",
                    metric=metric_name,
                    value=value,
                    target=target,
                    gap=target - value,
                )

            # 低于目标
            elif value < target:
                logger.info(
                    "metric_below_target",
                    metric=metric_name,
                    value=value,
                    target=target,
                    gap=target - value,
                )

    def _load_history(self) -> None:
        """加载历史数据。"""
        if not self._storage_path:
            return

        path = Path(self._storage_path)
        if not path.exists():
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
                self._records = [MetricRecord(**r) for r in data]
            logger.info("metrics_history_loaded", count=len(self._records))
        except Exception as e:
            logger.warning("metrics_history_load_failed", error=str(e))

    def _save_history(self) -> None:
        """保存历史数据。"""
        if not self._storage_path:
            return

        try:
            path = Path(self._storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = [asdict(r) for r in self._records]
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("metrics_history_save_failed", error=str(e))

    def clear_history(self) -> None:
        """清空历史数据。"""
        self._records.clear()
        self._save_history()

    def get_summary(self) -> dict:
        """获取指标摘要（用于 API 返回）。

        Returns:
            摘要信息字典
        """
        summary = {
            "total_records": len(self._records),
            "targets": self._targets,
            "current_status": {},
            "trends": {},
            "recommendations": self.get_recommendations(),
        }

        for metric_name in self._targets.keys():
            avg = self.get_average(metric_name)
            trend = self.get_trend(metric_name, limit=5)

            summary["current_status"][metric_name] = {
                "average": round(avg, 4) if avg else None,
                "target": self._targets[metric_name],
                "is_met": avg >= self._targets[metric_name] if avg else False,
                "gap": round(self._targets[metric_name] - avg, 4) if avg else None,
            }
            summary["trends"][metric_name] = trend

        return summary


# 全局单例
metrics_tracker = MetricsTracker()