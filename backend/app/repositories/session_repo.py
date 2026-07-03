"""Repository: MySQL sessions/messages data access + 统计查询."""
from datetime import datetime, timedelta
from typing import Optional

import structlog

from app.models import ChatSession, Message, get_session

logger = structlog.get_logger()


class SessionRepository:
    """MySQL sessions + messages repository."""

    def __init__(self):
        self.last_error: str = ""

    def create_session(self, session_id: str, user_id: str) -> Optional[ChatSession]:
        s = get_session()
        try:
            record = ChatSession(session_id=session_id, user_id=user_id)
            s.add(record)
            s.commit()
            s.refresh(record)
            self.last_error = ""
            return record
        except Exception as e:
            s.rollback()
            self.last_error = str(e)
            logger.exception("session_create_failed", error=self.last_error)
            return None
        finally:
            s.close()

    def add_message(self, session_id: str, query: str, response: str = "",
                    latency_ms: int = 0, is_success: bool = True,
                    error_msg: str = "") -> Optional[Message]:
        s = get_session()
        try:
            record = Message(
                session_id=session_id,
                query=query,
                response=response,
                latency_ms=latency_ms,
                is_success=1 if is_success else 0,
                error_msg=error_msg or None,
            )
            s.add(record)
            # 更新会话消息计数
            s.query(ChatSession).filter(ChatSession.session_id == session_id) \
                .update({"message_count": ChatSession.message_count + 1})
            s.commit()
            self.last_error = ""
            return record
        except Exception as e:
            s.rollback()
            self.last_error = str(e)
            logger.exception("message_add_failed", error=self.last_error)
            return None
        finally:
            s.close()

    @staticmethod
    def stats(window_hours: int = 7 * 24) -> dict:
        """Dashboard 核心指标。"""
        s = get_session()
        try:
            since = datetime.utcnow() - timedelta(hours=window_hours)
            total_sessions = s.query(ChatSession) \
                .filter(ChatSession.gmt_create >= since).count()
            total_messages = s.query(Message) \
                .filter(Message.gmt_create >= since).count()
            active_users = s.query(ChatSession.user_id) \
                .filter(ChatSession.gmt_create >= since).distinct().count()
            session_depth = round(total_messages / total_sessions, 2) if total_sessions > 0 else 0.0
            return {
                "active_users": active_users,
                "sessions": total_sessions,
                "messages": total_messages,
                "session_depth": session_depth,
            }
        finally:
            s.close()

    @staticmethod
    def ai_performance(window_hours: int = 7 * 24) -> dict:
        """AI 性能指标。"""
        s = get_session()
        try:
            since = datetime.utcnow() - timedelta(hours=window_hours)
            rows = s.query(Message).filter(Message.gmt_create >= since).all()
            total = len(rows)
            if total == 0:
                return {"success_rate": 0.0, "avg_latency_ms": 0.0, "p95_latency_ms": 0.0}
            success_count = sum(1 for r in rows if r.is_success)
            latencies = sorted([r.latency_ms or 0 for r in rows])
            p95_idx = int(len(latencies) * 0.95)
            return {
                "success_rate": round(success_count / total * 100, 1),
                "avg_latency_ms": round(sum(latencies) / total, 2),
                "p95_latency_ms": latencies[min(p95_idx, len(latencies) - 1)],
            }
        finally:
            s.close()

    @staticmethod
    def quality_snapshot(days: int = 7) -> list[dict]:
        """近N天质量快照（每天的 错误率/无知识率/慢响应率）。"""
        s = get_session()
        try:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            result = []
            for i in range(days - 1, -1, -1):
                day_start = today - timedelta(days=i)
                day_end = day_start + timedelta(days=1)
                rows = s.query(Message).filter(
                    Message.gmt_create >= day_start,
                    Message.gmt_create < day_end,
                ).all()
                total = len(rows)
                if total == 0:
                    result.append({
                        "date": day_start.strftime("%m/%d"),
                        "error_rate": 0.0,
                        "no_knowledge_rate": 0.0,
                        "slow_rate": 0.0,
                    })
                else:
                    error_rate = sum(1 for r in rows if not r.is_success) / total * 100
                    slow_rate = sum(1 for r in rows if (r.latency_ms or 0) > 10000) / total * 100
                    result.append({
                        "date": day_start.strftime("%m/%d"),
                        "error_rate": round(error_rate, 1),
                        "no_knowledge_rate": 0.0,  # 暂无"无知识"判定来源
                        "slow_rate": round(slow_rate, 1),
                    })
            return result
        finally:
            s.close()

    @staticmethod
    def session_trend(days: int = 7) -> list[dict]:
        """近N天会话趋势。"""
        s = get_session()
        try:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            result = []
            for i in range(days - 1, -1, -1):
                day_start = today - timedelta(days=i)
                day_end = day_start + timedelta(days=1)
                count = s.query(ChatSession).filter(
                    ChatSession.gmt_create >= day_start,
                    ChatSession.gmt_create < day_end,
                ).count()
                result.append({"date": day_start.strftime("%m/%d"), "count": count})
            return result
        finally:
            s.close()

    @staticmethod
    def user_trend(days: int = 7) -> list[dict]:
        """近N天活跃用户趋势。"""
        s = get_session()
        try:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            result = []
            for i in range(days - 1, -1, -1):
                day_start = today - timedelta(days=i)
                day_end = day_start + timedelta(days=1)
                count = s.query(ChatSession.user_id).filter(
                    ChatSession.gmt_create >= day_start,
                    ChatSession.gmt_create < day_end,
                ).distinct().count()
                result.append({"date": day_start.strftime("%m/%d"), "count": count})
            return result
        finally:
            s.close()


# 单例
session_repo = SessionRepository()
