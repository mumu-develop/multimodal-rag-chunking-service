"""切分质量评估器。

适配 markdown_chunker 输出的质量评估模块。

功能：
1. 切分完成后自动评估质量
2. 生成质量报告
3. 提供优化建议

指标：
- size_consistency: chunk 大小一致性
- content_coverage: 内容覆盖率
- heading_inheritance: 标题继承率
- token_distribution: token 分布合理性
- quality_score: 综合质量评分
"""
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.config import settings

# Token counting function - use fallback if token_utils is unavailable
try:
    from app.utils.token_utils import num_tokens_from_string
except ImportError:
    # Fallback: simple character-based counting
    def num_tokens_from_string(text: str) -> int:
        """Fallback token counting using character length."""
        if not text:
            return 0
        # Approximate: Chinese ~1.5 chars/token, English ~4 chars/token
        # Use conservative estimate: ~2 chars/token
        return len(text) // 2

logger = structlog.get_logger()


@dataclass
class ChunkQualityMetrics:
    """切分质量指标。

    Attributes:
        chunk_count: 切片数量
        avg_chunk_size: 平均切片大小（字符）
        max_chunk_size: 最大切片大小
        min_chunk_size: 最小切片大小
        avg_chunk_tokens: 平均切片 token 数
        size_consistency: 大小一致性（0-1）
        content_coverage: 内容覆盖率（0-1）
        heading_inheritance: 标题继承率（0-1）
        type_distribution: 类型分布 {type: count}
        token_distribution_score: token 分布合理性（0-1）
        quality_score: 综合质量评分（0-1）
        recommendations: 优化建议列表
    """
    chunk_count: int = 0
    avg_chunk_size: float = 0.0
    max_chunk_size: int = 0
    min_chunk_size: int = 0
    avg_chunk_tokens: float = 0.0
    size_consistency: float = 0.0
    content_coverage: float = 0.0
    heading_inheritance: float = 0.0
    type_distribution: dict = field(default_factory=dict)
    token_distribution_score: float = 0.0
    quality_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    evaluation_time_ms: float = 0.0


class ChunkQualityEvaluator:
    """切分质量评估器。

    评估切分结果的质量，生成报告和优化建议。

    使用示例：
        evaluator = ChunkQualityEvaluator()
        metrics = evaluator.evaluate(chunks, original_text)
        if metrics.quality_score < 0.8:
            logger.warning("quality_low", recommendations=metrics.recommendations)
    """

    # 目标指标阈值
    TARGET_QUALITY = 0.85
    TARGET_SIZE_CONSISTENCY = 0.7
    TARGET_HEADING_INHERITANCE = 0.8
    MAX_TOKEN_LIMIT = 1024  # 嵌入模型 token 上限

    def evaluate(
        self,
        chunks: list[dict],
        original_text: str = "",
        doc_type: str = "",
        chunk_strategy: str = "",
    ) -> ChunkQualityMetrics:
        """评估切分质量。

        Args:
            chunks: 切片列表
            original_text: 原始文本（用于计算覆盖率）
            doc_type: 文档类型
            chunk_strategy: 切分策略

        Returns:
            ChunkQualityMetrics: 质量评估结果
        """
        t0 = time.monotonic()

        if not chunks:
            return ChunkQualityMetrics(
                recommendations=["切片列表为空，请检查解析流程"],
            )

        metrics = ChunkQualityMetrics()
        metrics.chunk_count = len(chunks)

        # 1. 计算大小指标
        sizes = []
        token_counts = []
        for c in chunks:
            content = c.get("content", "")
            size = len(content)
            sizes.append(size)
            # 计算 token 数
            try:
                tokens = num_tokens_from_string(content)
                token_counts.append(tokens)
            except Exception:
                # token_utils 可能失败，使用近似值
                token_counts.append(size // 4)

        metrics.avg_chunk_size = sum(sizes) / len(sizes) if sizes else 0
        metrics.max_chunk_size = max(sizes) if sizes else 0
        metrics.min_chunk_size = min(sizes) if sizes else 0
        metrics.avg_chunk_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

        # 2. 大小一致性
        if len(sizes) > 1 and metrics.avg_chunk_size > 0:
            variance = sum((s - metrics.avg_chunk_size) ** 2 for s in sizes) / len(sizes)
            metrics.size_consistency = 1.0 - min(variance / (metrics.avg_chunk_size ** 2), 1.0)
        else:
            metrics.size_consistency = 1.0

        # 3. 内容覆盖率
        if original_text:
            chunk_chars = sum(sizes)
            coverage = min(chunk_chars / len(original_text), 1.0)
            metrics.content_coverage = round(coverage, 4)
        else:
            metrics.content_coverage = 1.0

        # 4. 标题继承率
        with_heading = sum(1 for c in chunks if c.get("heading_text"))
        metrics.heading_inheritance = with_heading / len(chunks) if chunks else 0

        # 5. 类型分布
        type_counts = {}
        for c in chunks:
            ctype = c.get("chunk_type", "text")
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
        metrics.type_distribution = type_counts

        # 6. Token 分布合理性
        # 检查是否有超长 chunk（超过 token 限制）
        oversized = sum(1 for t in token_counts if t > self.MAX_TOKEN_LIMIT)
        undersized = sum(1 for t in token_counts if t < 50)  # 过短 chunk
        total = len(token_counts)

        if total > 0:
            oversized_ratio = oversized / total
            undersized_ratio = undersized / total
            metrics.token_distribution_score = 1.0 - (oversized_ratio * 0.5 + undersized_ratio * 0.3)
        else:
            metrics.token_distribution_score = 1.0

        # 7. 综合质量评分
        # 权重：token 分布 40% + 大小一致性 20% + 标题继承 20% + 覆盖率 20%
        metrics.quality_score = round(
            metrics.token_distribution_score * 0.4 +
            metrics.size_consistency * 0.2 +
            metrics.heading_inheritance * 0.2 +
            metrics.content_coverage * 0.2,
            4
        )

        # 8. 生成优化建议
        metrics.recommendations = self._generate_recommendations(
            metrics, chunk_strategy, oversized, undersized
        )

        metrics.evaluation_time_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "chunk_quality_evaluated",
            chunk_count=metrics.chunk_count,
            quality_score=metrics.quality_score,
            avg_chunk_tokens=int(metrics.avg_chunk_tokens),
            oversized=oversized,
            heading_inheritance=metrics.heading_inheritance,
        )

        return metrics

    def _generate_recommendations(
        self,
        metrics: ChunkQualityMetrics,
        chunk_strategy: str,
        oversized: int,
        undersized: int,
    ) -> list[str]:
        """生成优化建议。

        Args:
            metrics: 质量指标
            chunk_strategy: 当前策略
            oversized: 超长 chunk 数量
            undersized: 过短 chunk 数量

        Returns:
            建议列表
        """
        recommendations = []

        # 超长 chunk 建议
        if oversized > 0:
            pct = oversized / metrics.chunk_count * 100
            recommendations.append(
                f"有 {oversized} 个切片超过 token 限制（{pct:.1f}%），"
                f"建议减小 chunk_size 或使用 {chunk_strategy != 'smart' and 'smart' or 'semantic'} 策略"
            )

        # 过短 chunk 建议
        if undersized > metrics.chunk_count * 0.3:
            recommendations.append(
                f"过短切片占比过高，建议增大 chunk_size 或减少切分粒度"
            )

        # 大小一致性建议
        if metrics.size_consistency < self.TARGET_SIZE_CONSISTENCY:
            recommendations.append(
                f"切片大小一致性较低（{metrics.size_consistency:.2f}），"
                f"建议使用 semantic 策略保持语义完整"
            )

        # 标题继承建议
        if metrics.heading_inheritance < self.TARGET_HEADING_INHERITANCE:
            recommendations.append(
                f"标题继承率较低（{metrics.heading_inheritance:.2f}），"
                f"建议使用 smart 或 heading 策略保持文档结构"
            )

        # 综合评分建议
        if metrics.quality_score < self.TARGET_QUALITY:
            recommendations.append(
                f"综合质量评分未达标（{metrics.quality_score:.2f} < {self.TARGET_QUALITY}），"
                f"建议尝试不同切分策略或调整参数"
            )

        # 如果质量达标，给予正向反馈
        if metrics.quality_score >= self.TARGET_QUALITY:
            recommendations.append(
                f"切分质量达标，当前策略 {chunk_strategy} 效果良好"
            )

        return recommendations

    def generate_report(self, metrics: ChunkQualityMetrics) -> dict:
        """生成质量报告。

        Args:
            metrics: 质量指标

        Returns:
            报告字典
        """
        return {
            "summary": {
                "quality_score": metrics.quality_score,
                "is_qualified": metrics.quality_score >= self.TARGET_QUALITY,
                "chunk_count": metrics.chunk_count,
            },
            "metrics": {
                "avg_chunk_size": int(metrics.avg_chunk_size),
                "max_chunk_size": metrics.max_chunk_size,
                "min_chunk_size": metrics.min_chunk_size,
                "avg_chunk_tokens": int(metrics.avg_chunk_tokens),
                "size_consistency": round(metrics.size_consistency, 4),
                "content_coverage": round(metrics.content_coverage, 4),
                "heading_inheritance": round(metrics.heading_inheritance, 4),
                "token_distribution_score": round(metrics.token_distribution_score, 4),
            },
            "distribution": metrics.type_distribution,
            "recommendations": metrics.recommendations,
            "evaluation_time_ms": round(metrics.evaluation_time_ms, 2),
        }


# 全局单例
chunk_quality_evaluator = ChunkQualityEvaluator()