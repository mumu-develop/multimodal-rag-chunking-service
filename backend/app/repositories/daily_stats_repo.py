"""Repository: 日活统计数据访问（按天维度）。"""
import json
from datetime import datetime, date, timedelta
from typing import Optional

import structlog
from sqlalchemy import text

from app.models import get_session

logger = structlog.get_logger()


class DailyStatsRepository:
    """日活统计 Repository：按天维度统计。"""

    def __init__(self):
        self.last_error: str = ""

    def update_today(
        self,
        active_users: int,
        total_requests: int,
    ) -> bool:
        """更新今日日活统计（插入或更新）。"""
        s = get_session()
        try:
            today = date.today()

            # 检查今日记录是否存在
            existing = s.execute(
                text("""
                SELECT id, active_users, total_requests FROM daily_stats
                WHERE date = :date
                LIMIT 1
                """),
                {"date": today},
            ).fetchone()

            if existing:
                # 更新已有记录（累加）
                s.execute(
                    text("""
                    UPDATE daily_stats
                    SET active_users = :active_users,
                        total_requests = total_requests + :total_requests,
                        gmt_modified = :gmt_modified
                    WHERE date = :date
                    """),
                    {
                        "date": today,
                        "active_users": active_users,
                        "total_requests": total_requests,
                        "gmt_modified": datetime.now(),
                    },
                )
            else:
                # 创建新记录
                s.execute(
                    text("""
                    INSERT INTO daily_stats (date, active_users, total_requests, gmt_create)
                    VALUES (:date, :active_users, :total_requests, :gmt_create)
                    """),
                    {
                        "date": today,
                        "active_users": active_users,
                        "total_requests": total_requests,
                        "gmt_create": datetime.now(),
                    },
                )

            s.commit()
            self.last_error = ""
            return True
        except Exception as e:
            s.rollback()
            self.last_error = str(e)
            logger.exception("daily_stats_update_failed", error=str(e))
            return False
        finally:
            s.close()

    def get_today(self) -> Optional[dict]:
        """获取今日日活统计。"""
        s = get_session()
        try:
            result = s.execute(
                text("""
                SELECT date, active_users, total_requests, gmt_modified
                FROM daily_stats WHERE date = :date LIMIT 1
                """),
                {"date": date.today()},
            ).fetchone()

            if result:
                return {
                    "date": str(result[0]),
                    "active_users": result[1],
                    "total_requests": result[2],
                    "gmt_modified": str(result[3]),
                }
            return None
        except Exception:
            return None
        finally:
            s.close()

    def get_stats_by_date(self, target_date: date) -> Optional[dict]:
        """获取指定日期的统计。"""
        s = get_session()
        try:
            result = s.execute(
                text("""
                SELECT date, active_users, total_requests, gmt_modified
                FROM daily_stats WHERE date = :date LIMIT 1
                """),
                {"date": target_date},
            ).fetchone()

            if result:
                return {
                    "date": str(result[0]),
                    "active_users": result[1],
                    "total_requests": result[2],
                    "gmt_modified": str(result[3]),
                }
            return None
        except Exception:
            return None
        finally:
            s.close()

    def get_trend(self, days: int = 30) -> list[dict]:
        """获取最近 N 天的日活趋势（用于报表）。"""
        s = get_session()
        try:
            results = s.execute(
                text("""
                SELECT date, active_users, total_requests
                FROM daily_stats
                WHERE date >= :start_date
                ORDER BY date DESC
                """),
                {"start_date": date.today() - timedelta(days=days)},
            ).fetchall()

            return [
                {
                    "date": str(r[0]),
                    "active_users": r[1],
                    "total_requests": r[2],
                }
                for r in results
            ]
        except Exception:
            return []
        finally:
            s.close()

    def get_monthly_summary(self, year: int, month: int) -> dict:
        """获取月度汇总（用于报表）。"""
        s = get_session()
        try:
            # 计算该月起止日期
            start_date = date(year, month, 1)
            if month == 12:
                end_date = date(year + 1, 1, 1)
            else:
                end_date = date(year, month + 1, 1)

            result = s.execute(
                text("""
                SELECT
                    COUNT(*) as days,
                    SUM(active_users) as total_users,
                    AVG(active_users) as avg_users,
                    MAX(active_users) as max_users,
                    MIN(active_users) as min_users,
                    SUM(total_requests) as total_requests
                FROM daily_stats
                WHERE date >= :start_date AND date < :end_date
                """),
                {"start_date": start_date, "end_date": end_date},
            ).fetchone()

            if result and result[0]:
                return {
                    "year": year,
                    "month": month,
                    "days": result[0],
                    "total_users": result[1] or 0,
                    "avg_users": round(result[2] or 0, 1),
                    "max_users": result[3] or 0,
                    "min_users": result[4] or 0,
                    "total_requests": result[5] or 0,
                }
            return {"year": year, "month": month, "days": 0, "total_users": 0}
        except Exception:
            return {"year": year, "month": month, "days": 0, "total_users": 0}
        finally:
            s.close()


# 单例
daily_stats_repo = DailyStatsRepository()