"""日活模拟定时任务 + SSE 推送。

每5分钟：
1. 模拟增加一批日活数据（10 QPS × 5分钟 = 3000 次）
2. 拉取当前日活数据
3. 通过 SSE 推送给前端

伪代码实现，可直接运行测试效果。
"""
import asyncio
import random
import time
from datetime import datetime, date
from typing import Set

import structlog

logger = structlog.get_logger()

# ============================================================
# 模拟数据存储（实际场景用 Redis 或数据库）
# ============================================================

# 当前日活用户集合
_current_active_users: Set[str] = set()
# 上次推送的日活数（用于判断是否有变化）
_last_pushed_count: int = 0
# SSE 客户端连接队列
_sse_clients: list = []

# ============================================================
# 定时任务1：模拟增加日活数据
# ============================================================

async def simulate_daily_active_task():
    """
    定时任务：每5分钟模拟增加一批日活数据。

    设计参数：
    - 10 QPS（每秒10次请求）
    - 5分钟 = 300秒
    - 总请求数 = 10 × 300 = 3000 次

    模拟逻辑：
    1. 生成随机用户ID（模拟不同用户访问）
    2. 用户ID重复率约30%（模拟用户多次访问）
    3. 新用户加入日活集合
    """
    while True:
        logger.info("daily_active_simulate_start", time=datetime.now().isoformat())

        # 模拟 3000 次请求（10 QPS × 300秒）
        # 实际生产中，这些请求分散在5分钟内
        # 这里为了演示，批量生成

        # 生成用户池：假设有1000个潜在用户
        potential_users = [f"user_{i:04d}" for i in range(1000)]

        # 模拟访问：随机选取用户（有重复）
        new_users_count = 0
        for _ in range(3000):
            # 30% 概率是新用户访问，70% 是老用户重复访问
            if random.random() < 0.3:
                user_id = random.choice(potential_users)
                if user_id not in _current_active_users:
                    _current_active_users.add(user_id)
                    new_users_count += 1
            else:
                # 老用户重复访问（从已活跃用户中随机选）
                if _current_active_users:
                    user_id = random.choice(list(_current_active_users))
                else:
                    user_id = random.choice(potential_users)
                    _current_active_users.add(user_id)
                    new_users_count += 1

        logger.info(
            "daily_active_simulate_end",
            total_requests=3000,
            new_users=new_users_count,
            total_active=len(_current_active_users),
        )

        # 5分钟后再次执行
        await asyncio.sleep(300)  # 300秒 = 5分钟


# ============================================================
# 定时任务2：拉取日活数据并推送 SSE
# ============================================================

async def push_daily_active_task():
    """
    定时任务：每5分钟拉取日活数据，判断变化后推送给前端。

    判断逻辑：
    1. 比较当前日活数与上次推送的数
    2. 如果有变化，推送新数据
    3. 如果无变化，可选推送心跳或跳过
    """
    while True:
        current_count = len(_current_active_users)

        # 判断是否有变化
        if current_count != _last_pushed_count:
            # 有变化，推送给所有 SSE 客户端
            data = {
                "type": "daily_active_update",
                "date": str(date.today()),
                "count": current_count,
                "change": current_count - _last_pushed_count,
                "timestamp": datetime.now().isoformat(),
            }

            # 推送给所有连接的客户端
            disconnected = []
            for client in _sse_clients:
                try:
                    await client.put(data)
                except Exception:
                    disconnected.append(client)

            # 移除断开的客户端
            for client in disconnected:
                _sse_clients.remove(client)

            _last_pushed_count = current_count

            logger.info(
                "daily_active_pushed",
                count=current_count,
                change=data["change"],
                clients=len(_sse_clients),
            )
        else:
            # 无变化，推送心跳（保持连接）
            heartbeat = {
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
            }
            for client in _sse_clients:
                try:
                    await client.put(heartbeat)
                except Exception:
                    pass

            logger.debug("daily_active_heartbeat", count=current_count)

        # 5分钟后再次执行
        await asyncio.sleep(300)


# ============================================================
# SSE 接口：前端建立长连接
# ============================================================

class SSEClient:
    """SSE 客户端连接（使用 asyncio.Queue）。"""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.connected = True

    async def put(self, data: dict):
        """推送数据到客户端队列。"""
        if self.connected:
            await self.queue.put(data)

    async def get(self) -> dict:
        """客户端获取数据。"""
        return await self.queue.get()

    def disconnect(self):
        """断开连接。"""
        self.connected = False


async def sse_daily_active_handler():
    """
    SSE 接口伪代码：前端建立长连接监听日活变化。

    实际 FastAPI 实现：
    ```python
    @router.get("/sse/daily-active")
    async def sse_daily_active(request: Request):
        client = SSEClient()
        _sse_clients.append(client)

        async def event_stream():
            try:
                while True:
                    data = await client.get()
                    yield f"data: {json.dumps(data)}\\n\\n"
                    if data.get("type") == "close":
                        break
            except asyncio.CancelledError:
                client.disconnect()
                _sse_clients.remove(client)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
        )
    ```
    """
    client = SSEClient()
    _sse_clients.append(client)

    logger.info("sse_client_connected", clients=len(_sse_clients))

    try:
        while client.connected:
            data = await client.get()
            # 格式化为 SSE 数据格式
            yield f"event: daily_active\ndata: {json.dumps(data)}\n\n"
    except asyncio.CancelledError:
        client.disconnect()
        _sse_clients.remove(client)
        logger.info("sse_client_disconnected")


# ============================================================
# 启动定时任务
# ============================================================

async def start_daily_active_tasks():
    """
    启动所有定时任务。

    在 FastAPI lifespan 中调用：
    ```python
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动后台任务
        asyncio.create_task(start_daily_active_tasks())
        yield
    ```
    """
    # 启动两个定时任务
    task1 = asyncio.create_task(simulate_daily_active_task())
    task2 = asyncio.create_task(push_daily_active_task())

    logger.info("daily_active_tasks_started")

    await asyncio.gather(task1, task2)


# ============================================================
# 测试入口（可直接运行看效果）
# ============================================================

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("日活模拟测试（伪代码演示）")
    print("=" * 60)
    print("设计参数：")
    print("  - QPS: 10")
    print("  - 定时周期: 5分钟")
    print("  - 每周期请求: 3000次")
    print("  - 新用户率: 30%")
    print("=" * 60)

    # 模拟运行（缩短时间便于演示）
    async def quick_test():
        """快速测试（每5秒执行一次，便于观察）。"""

        for round_num in range(1, 6):
            print(f"\n--- 第 {round_num} 轮 ---")
            print(f"时间: {datetime.now().strftime('%H:%M:%S')}")

            # 模拟增加日活
            potential_users = [f"user_{i:04d}" for i in range(1000)]

            requests_count = 50  # 模拟50次请求（缩短）
            new_users = 0
            for _ in range(requests_count):
                if random.random() < 0.3:
                    user_id = random.choice(potential_users)
                    if user_id not in _current_active_users:
                        _current_active_users.add(user_id)
                        new_users += 1

            current_count = len(_current_active_users)

            # 推送 SSE
            if current_count != _last_pushed_count:
                data = {
                    "type": "daily_active_update",
                    "date": str(date.today()),
                    "count": current_count,
                    "change": current_count - _last_pushed_count,
                    "timestamp": datetime.now().isoformat(),
                }
                print(f"推送数据: {json.dumps(data, indent=2)}")
                _last_pushed_count = current_count
            else:
                print(f"无变化，当前日活: {current_count}")

            # 等待5秒
            await asyncio.sleep(5)

        print(f"\n最终日活用户数: {len(_current_active_users)}")
        print(f"模拟用户ID示例: {list(_current_active_users)[:10]}")

    asyncio.run(quick_test())