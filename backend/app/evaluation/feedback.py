"""线上用户反馈收集：点击行为记录 + 搜索满意度指标。"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import structlog

from app.config import settings

logger = structlog.get_logger()


@dataclass
class FeedbackEntry:
    """单条用户反馈。"""
    query: str
    top_result_ids: list[str]
    clicked_ids: list[str]
    dwell_times_ms: list[int] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    doc_id: str = ""


class FeedbackCollector:
    """
    线上搜索质量反馈收集器。

    通过前端上报用户点击行为，持续监测搜索满意度，
    在指标持续低于阈值时触发告警。

    用法:
        collector = FeedbackCollector()

        # 记录用户点击
        collector.record_click(query="...", clicked_ids=["id1", "id2"], dwell_times=[3000, 5000])

        # 计算当前窗口内的满意度
        metrics = collector.get_metrics(window_minutes=60)
        if metrics["click_through_rate"] < 0.3:
            logger.warning("low_search_satisfaction")
    """

    def __init__(self, storage_dir: str | None = None):
        """
        Args:
            storage_dir: 反馈数据存储目录（默认 uploads/feedback/）
        """
        self._storage_dir = storage_dir or os.path.join(settings.UPLOAD_DIR, "feedback")
        os.makedirs(self._storage_dir, exist_ok=True)
        self._entries: list[FeedbackEntry] = []

    def record(
        self,
        query: str,
        top_result_ids: list[str],
        clicked_ids: list[str],
        dwell_times_ms: list[int] | None = None,
        doc_id: str = "",
    ) -> FeedbackEntry:
        """
        记录单次搜索反馈。

        Args:
            query: 用户查询
            top_result_ids: 展示给用户的 top_k 结果 chunk_id 列表
            clicked_ids: 用户实际点击的 chunk_id 列表
            dwell_times_ms: 每个点击的停留时间（毫秒）
            doc_id: 文档 ID（可选）

        Returns:
            FeedbackEntry 对象
        """
        entry = FeedbackEntry(
            query=query,
            top_result_ids=top_result_ids,
            clicked_ids=clicked_ids,
            dwell_times_ms=dwell_times_ms or [],
            timestamp=time.time(),
            doc_id=doc_id,
        )
        self._entries.append(entry)

        # 持久化到文件（每 100 条刷盘一次）
        if len(self._entries) % 100 == 0:
            self._flush()

        return entry

    def get_metrics(self, window_minutes: int = 60) -> dict:
        """
        计算最近 N 分钟内的搜索满意度指标。

        Args:
            window_minutes: 时间窗口（分钟）

        Returns:
            {
                "total_queries": int,
                "unique_queries": int,
                "click_through_rate": float,      # 有点击的 query 占比
                "avg_clicked_per_query": float,    # 平均每次查询点击数
                "avg_dwell_time_ms": float,        # 平均停留时间
                "top_clicked_chunks": [...],       # 最常被点击的 chunk 排名
            }
        """
        cutoff = time.time() - window_minutes * 60
        recent = [e for e in self._entries if e.timestamp >= cutoff]

        if not recent:
            return {"total_queries": 0, "window_minutes": window_minutes}

        # 有点击的查询占比
        with_clicks = [e for e in recent if e.clicked_ids]
        click_through_rate = len(with_clicks) / len(recent) if recent else 0

        # 平均每次查询点击数
        avg_clicked = sum(len(e.clicked_ids) for e in recent) / len(recent)

        # 平均停留时间
        all_dwell = [dt for e in with_clicks for dt in e.dwell_times_ms]
        avg_dwell = sum(all_dwell) / len(all_dwell) if all_dwell else 0

        # 最常点击的 chunk
        chunk_click_count: dict[str, int] = {}
        for e in with_clicks:
            for cid in e.clicked_ids:
                chunk_click_count[cid] = chunk_click_count.get(cid, 0) + 1

        top_clicked = sorted(chunk_click_count.items(), key=lambda x: -x[1])[:20]

        # 唯一 query 数量
        unique_queries = len({e.query for e in recent})

        return {
            "total_queries": len(recent),
            "unique_queries": unique_queries,
            "window_minutes": window_minutes,
            "click_through_rate": round(click_through_rate, 4),
            "avg_clicked_per_query": round(avg_clicked, 2),
            "avg_dwell_time_ms": round(avg_dwell, 2),
            "top_clicked_chunks": [{"chunk_id": c, "clicks": n} for c, n in top_clicked],
        }

    def get_low_satisfaction_queries(
        self,
        window_minutes: int = 60,
        threshold: float = 0.3,
    ) -> list[str]:
        """
        找出点击率低于阈值的查询，用于排查搜索质量问题。

        Args:
            window_minutes: 时间窗口
            threshold: 点击率阈值（默认 0.3）

        Returns:
            点击率低于阈值的 query 列表
        """
        cutoff = time.time() - window_minutes * 60
        recent = [e for e in self._entries if e.timestamp >= cutoff]

        # 按 query 分组统计
        query_stats: dict[str, dict] = {}
        for e in recent:
            if e.query not in query_stats:
                query_stats[e.query] = {"total": 0, "clicked": 0}
            query_stats[e.query]["total"] += 1
            if e.clicked_ids:
                query_stats[e.query]["clicked"] += 1

        low_satisfaction = []
        for query, stats in query_stats.items():
            ctr = stats["clicked"] / stats["total"] if stats["total"] > 0 else 0
            if ctr < threshold and stats["total"] >= 3:  # 至少出现 3 次
                low_satisfaction.append(query)

        if low_satisfaction:
            logger.warning(
                "low_satisfaction_queries_detected",
                queries=low_satisfaction,
                threshold=threshold,
                window_minutes=window_minutes,
            )

        return low_satisfaction

    def export(self, output_path: str | None = None) -> int:
        """导出全部反馈数据到 JSON 文件。"""
        self._flush()

        out_path = output_path or os.path.join(self._storage_dir, "feedback_export.json")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        data = []
        for entry in self._entries:
            data.append({
                "query": entry.query,
                "top_result_ids": entry.top_result_ids,
                "clicked_ids": entry.clicked_ids,
                "dwell_times_ms": entry.dwell_times_ms,
                "timestamp": datetime.fromtimestamp(entry.timestamp).isoformat(),
                "doc_id": entry.doc_id,
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("feedback_exported", path=out_path, count=len(data))
        return len(data)

    def stats(self) -> dict:
        """返回反馈数据整体统计。"""
        if not self._entries:
            return {"total_entries": 0}

        oldest = min(e.timestamp for e in self._entries)
        newest = max(e.timestamp for e in self._entries)
        total = len(self._entries)
        with_clicks = sum(1 for e in self._entries if e.clicked_ids)

        return {
            "total_entries": total,
            "entries_with_clicks": with_clicks,
            "click_through_rate": round(with_clicks / total, 4) if total > 0 else 0,
            "time_range_hours": round((newest - oldest) / 3600, 2),
            "unique_queries": len({e.query for e in self._entries}),
        }

    # ============================================================
    # 内部方法
    # ============================================================

    def _flush(self) -> None:
        """将内存中的条目追加写入文件。"""
        if not self._entries:
            return

        date_str = datetime.now().strftime("%Y%m%d")
        file_path = os.path.join(self._storage_dir, f"feedback_{date_str}.jsonl")

        with open(file_path, "a", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps({
                    "q": entry.query,
                    "t": entry.top_result_ids,
                    "c": entry.clicked_ids,
                    "d": entry.dwell_times_ms,
                    "ts": entry.timestamp,
                    "doc": entry.doc_id,
                }, ensure_ascii=False) + "\n")

        self._entries.clear()
