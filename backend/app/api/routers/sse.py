"""SSE 路由：实时推送日活数据 + 写入数据库。"""
import asyncio
import json
import random
from datetime import datetime, date
from typing import Set

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.repositories.daily_stats_repo import daily_stats_repo

logger = structlog.get_logger()
router = APIRouter()

# ============================================================
# 全局状态（模拟用户访问）
# ============================================================

# 当前活跃用户集合（用于计算日活）
_current_active_users: Set[str] = set()
# 累计请求数
_total_requests: int = 0
# SSE 客户端连接
_sse_clients: list = []


class SSEClient:
    """SSE 客户端连接。"""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.connected = True

    async def put(self, data: dict):
        if self.connected:
            await self.queue.put(data)

    def disconnect(self):
        self.connected = False


# ============================================================
# SSE 接口：前端建立长连接
# ============================================================

@router.get("/daily-active")
async def sse_daily_active(request: Request):
    """SSE 接口：前端建立长连接监听日活变化。"""
    client = SSEClient()
    _sse_clients.append(client)
    logger.info("sse_client_connected", clients=len(_sse_clients))

    async def event_stream():
        try:
            # 先发送当前状态（从数据库读取）
            today_stats = daily_stats_repo.get_today()
            initial_data = {
                "type": "initial",
                "date": str(date.today()),
                "active_users": today_stats["active_users"] if today_stats else 0,
                "total_requests": today_stats["total_requests"] if today_stats else 0,
                "timestamp": datetime.now().isoformat(),
            }
            yield f"event: daily_active\ndata: {json.dumps(initial_data)}\n\n"

            while True:
                if await request.is_disconnected():
                    break

                try:
                    data = await asyncio.wait_for(client.queue.get(), timeout=30)
                    yield f"event: daily_active\ndata: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    heartbeat = {"type": "heartbeat", "timestamp": datetime.now().isoformat()}
                    yield f"event: heartbeat\ndata: {json.dumps(heartbeat)}\n\n"

        except asyncio.CancelledError:
            client.disconnect()
        finally:
            if client in _sse_clients:
                _sse_clients.remove(client)
            logger.info("sse_client_disconnected", clients=len(_sse_clients))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ============================================================
# 查询接口（报表）
# ============================================================

@router.get("/today")
async def get_today_stats(request: Request):
    """获取今日日活统计（HTTP接口）。"""
    stats = daily_stats_repo.get_today()
    return {"success": True, "data": stats}


@router.get("/trend")
async def get_trend(days: int = 30, request: Request = None):
    """获取最近 N 天的日活趋势（报表）。"""
    trend = daily_stats_repo.get_trend(days=days)
    return {"success": True, "data": trend}


@router.get("/monthly/{year}/{month}")
async def get_monthly_summary(year: int, month: int, request: Request = None):
    """获取月度汇总（报表）。"""
    summary = daily_stats_repo.get_monthly_summary(year, month)
    return {"success": True, "data": summary}


@router.get("/date/{date_str}")
async def get_stats_by_date(date_str: str, request: Request = None):
    """获取指定日期的统计。"""
    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        return {"success": False, "message": "日期格式错误，请使用 YYYY-MM-DD", "data": None}

    stats = daily_stats_repo.get_stats_by_date(target_date)
    return {"success": True, "data": stats}


# ============================================================
# 手动触发模拟（测试）
# ============================================================

@router.post("/simulate")
async def simulate_daily_active(request: Request):
    """手动触发日活模拟（测试，立即写入数据库）。"""
    # 模拟用户访问
    potential_users = [f"user_{i:04d}" for i in range(1000)]
    requests_count = 600  # 60秒 × 10 QPS

    new_users = 0
    for _ in range(requests_count):
        if random.random() < 0.3:
            user_id = random.choice(potential_users)
            if user_id not in _current_active_users:
                _current_active_users.add(user_id)
                new_users += 1

    _total_requests += requests_count
    active_users = len(_current_active_users)

    # 写入数据库
    daily_stats_repo.update_today(active_users=active_users, total_requests=requests_count)

    # 推送 SSE
    data = {
        "type": "update",
        "date": str(date.today()),
        "active_users": active_users,
        "total_requests": _total_requests,
        "new_users": new_users,
        "timestamp": datetime.now().isoformat(),
    }

    for client in _sse_clients:
        try:
            await client.put(data)
        except Exception:
            pass

    return {
        "success": True,
        "data": {
            "requests": requests_count,
            "new_users": new_users,
            "active_users": active_users,
            "total_requests": _total_requests,
            "pushed_to_clients": len(_sse_clients),
        },
    }


# ============================================================
# 定时任务：每5分钟模拟并写入数据库
# ============================================================

async def start_simulate_task():
    """
    后台定时任务：每5分钟模拟日活数据并写入数据库。
    """
    global _total_requests, _current_active_users

    logger.info("daily_active_task_started_interval", interval_seconds=300)

    while True:
        try:
            # 模拟用户池（1000个潜在用户）
            potential_users = [f"user_{i:04d}" for i in range(1000)]

            # 模拟 3000 次请求（10 QPS × 300秒）
            requests_count = 3000
            new_users = 0

            for _ in range(requests_count):
                # 30% 新用户，70% 老用户重复访问
                if random.random() < 0.3:
                    user_id = random.choice(potential_users)
                    if user_id not in _current_active_users:
                        _current_active_users.add(user_id)
                        new_users += 1
                else:
                    # 老用户重复访问
                    if _current_active_users:
                        user_id = random.choice(list(_current_active_users))

            # 更新累计请求数
            _total_requests += requests_count
            active_users = len(_current_active_users)

            # 写入数据库
            daily_stats_repo.update_today(
                active_users=active_users,
                total_requests=requests_count,
            )

            # 推送 SSE
            data = {
                "type": "update",
                "date": str(date.today()),
                "active_users": active_users,
                "total_requests": _total_requests,
                "new_users": new_users,
                "timestamp": datetime.now().isoformat(),
            }

            disconnected = []
            for client in _sse_clients:
                try:
                    await client.put(data)
                except Exception:
                    disconnected.append(client)

            for client in disconnected:
                if client in _sse_clients:
                    _sse_clients.remove(client)

            logger.info(
                "daily_active_simulated_and_saved",
                requests=requests_count,
                new_users=new_users,
                active_users=active_users,
                total_requests=_total_requests,
                clients=len(_sse_clients),
            )

        except Exception:
            logger.exception("daily_active_simulate_error")

        # 等待5分钟
        await asyncio.sleep(300)