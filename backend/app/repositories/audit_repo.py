"""Repository: 审计日志 + 日活统计数据访问。"""
import json
from datetime import datetime, date, timedelta
from typing import Optional

import structlog
from sqlalchemy import text

from app.models import get_session

logger = structlog.get_logger()


class AuditRepository:
    """审计日志 Repository：记录用户访问 + 日活统计。"""

    def __init__(self):
        self.last_error: str = ""

    def log_access(
        self,
        user_id: str,
        user_role: str,
        path: str,
        method: str,
        params: dict,
        status_code: int,
        latency_ms: int,
        trace_id: str,
    ) -> bool:
        """记录用户访问审计日志。"""
        s = get_session()
        try:
            s.execute(
                text("""
                INSERT INTO audit_logs
                (user_id, user_role, path, method, params, status_code, latency_ms, trace_id, gmt_create)
                VALUES (:user_id, :user_role, :path, :method, :params, :status_code, :latency_ms, :trace_id, :gmt_create)
                """),
                {
                    "user_id": user_id,
                    "user_role": user_role,
                    "path": path,
                    "method": method,
                    "params": json.dumps(params) if params else None,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "trace_id": trace_id,
                    "gmt_create": datetime.now(),
                },
            )
            s.commit()
            self.last_error = ""
            return True
        except Exception as e:
            s.rollback()
            self.last_error = str(e)
            logger.debug("audit_log_failed", user_id=user_id, path=path, error=str(e))
            return False
        finally:
            s.close()

    def log_daily_active(self, user_id: str, date_val: date) -> bool:
        """记录日活（每天每个用户只记录一次）。"""
        s = get_session()
        try:
            existing = s.execute(
                text("""
                SELECT 1 FROM daily_active_users
                WHERE user_id = :user_id AND date = :date
                LIMIT 1
                """),
                {"user_id": user_id, "date": date_val},
            ).fetchone()

            if existing:
                return True

            s.execute(
                text("""
                INSERT INTO daily_active_users (user_id, date, gmt_create)
                VALUES (:user_id, :date, :gmt_create)
                """),
                {"user_id": user_id, "date": date_val, "gmt_create": datetime.now()},
            )
            s.commit()
            self.last_error = ""
            return True
        except Exception as e:
            s.rollback()
            self.last_error = str(e)
            logger.debug("daily_active_log_failed", user_id=user_id, error=str(e))
            return False
        finally:
            s.close()

    def get_daily_active_count(self, date_val: date) -> int:
        """获取指定日期的活跃用户数。"""
        s = get_session()
        try:
            result = s.execute(
                text("SELECT COUNT(*) FROM daily_active_users WHERE date = :date"),
                {"date": date_val},
            ).fetchone()
            return result[0] if result else 0
        except Exception:
            return 0
        finally:
            s.close()

    def get_daily_active_trend(self, days: int = 7) -> list[dict]:
        """获取最近 N 天的日活趋势。"""
        s = get_session()
        try:
            results = s.execute(
                text("""
                SELECT date, COUNT(*) as count
                FROM daily_active_users
                WHERE date >= :start_date
                GROUP BY date
                ORDER BY date DESC
                """),
                {"start_date": date.today() - timedelta(days=days)},
            ).fetchall()

            return [{"date": str(r[0]), "count": r[1]} for r in results]
        except Exception:
            return []
        finally:
            s.close()

    def get_user_access_logs(
        self,
        user_id: str,
        page: int = 1,
        size: int = 20,
    ) -> tuple[int, list[dict]]:
        """获取用户的访问日志（分页）。"""
        s = get_session()
        try:
            offset = (page - 1) * size

            total = s.execute(
                text("SELECT COUNT(*) FROM audit_logs WHERE user_id = :user_id"),
                {"user_id": user_id},
            ).fetchone()[0]

            results = s.execute(
                text("""
                SELECT id, user_id, user_role, path, method, status_code, latency_ms, params, trace_id, gmt_create
                FROM audit_logs
                WHERE user_id = :user_id
                ORDER BY gmt_create DESC
                LIMIT :size OFFSET :offset
                """),
                {"user_id": user_id, "size": size, "offset": offset},
            ).fetchall()

            logs = [
                {
                    "id": r[0],
                    "user_id": r[1],
                    "user_role": r[2] or "",
                    "path": r[3],
                    "method": r[4],
                    "status_code": r[5],
                    "latency_ms": r[6],
                    "params": json.loads(r[7]) if r[7] else None,
                    "trace_id": r[8] or "",
                    "gmt_create": str(r[9]),
                }
                for r in results
            ]

            return total, logs
        except Exception:
            return 0, []
        finally:
            s.close()

    def get_all_logs(
        self,
        page: int = 1,
        size: int = 20,
    ) -> tuple[int, list[dict]]:
        """获取所有访问日志（分页）。"""
        s = get_session()
        try:
            offset = (page - 1) * size

            total = s.execute(text("SELECT COUNT(*) FROM audit_logs")).fetchone()[0]

            results = s.execute(
                text("""
                SELECT id, user_id, user_role, path, method, status_code, latency_ms, params, trace_id, gmt_create
                FROM audit_logs
                ORDER BY gmt_create DESC
                LIMIT :size OFFSET :offset
                """),
                {"size": size, "offset": offset},
            ).fetchall()

            logs = [
                {
                    "id": r[0],
                    "user_id": r[1],
                    "user_role": r[2] or "",
                    "path": r[3],
                    "method": r[4],
                    "status_code": r[5],
                    "latency_ms": r[6],
                    "params": json.loads(r[7]) if r[7] else None,
                    "trace_id": r[8] or "",
                    "gmt_create": str(r[9]),
                }
                for r in results
            ]

            return total, logs
        except Exception:
            return 0, []
        finally:
            s.close()


# 单例
audit_repo = AuditRepository()