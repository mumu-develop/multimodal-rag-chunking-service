#!/usr/bin/env python3
"""
RAG 系统压测脚本

测试场景：
1. 搜索接口 - 主流量入口
2. FAQ 搜索 - FAQ 场景

压测指标：
- QPS（每秒请求数）
- 响应时间（P50、P95、P99）
- 错误率

使用方法：
    python scripts/benchmark.py --host http://localhost:8888 --kb-id 1 --duration 60 --concurrency 10
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime

import httpx

# ============================================================
# 测试数据生成
# ============================================================

# 模拟用户查询（覆盖不同场景）
SAMPLE_QUERIES = [
    "如何处理函数执行超时？",
    "数据库连接失败怎么解决？",
    "API 返回 500 错误如何排查？",
    "如何优化查询性能？",
    "系统内存不足如何处理？",
    "日志文件过大怎么清理？",
    "如何配置 HTTPS？",
    "服务启动失败怎么排查？",
    "如何实现分布式锁？",
    "缓存策略有哪些？",
    "消息队列如何选型？",
    "如何设计微服务架构？",
    "容器化部署有什么注意事项？",
    "如何监控服务健康状态？",
    "安全认证有哪些方案？",
    "如何处理并发问题？",
    "数据库索引如何优化？",
    "API 文档如何生成？",
    "如何实现灰度发布？",
    "系统崩溃如何恢复？",
]


def generate_query() -> str:
    """生成随机查询文本"""
    base = random.choice(SAMPLE_QUERIES)
    # 添加随机变化，模拟真实用户查询
    variations = [
        base,
        base + "请详细说明",
        base + "给出具体步骤",
        base.replace("如何", "怎么"),
        base.replace("？", ""),
    ]
    return random.choice(variations)


# ============================================================
# 压测统计
# ============================================================

class BenchmarkStats:
    """压测统计收集器"""

    def __init__(self):
        self.requests = []
        self.errors = []
        self.start_time = None
        self.end_time = None

    def record(self, latency_ms: float, success: bool, error: str = None):
        """记录一次请求"""
        self.requests.append({
            "latency_ms": latency_ms,
            "success": success,
            "error": error,
            "timestamp": time.time(),
        })
        if not success and error:
            self.errors.append(error)

    def report(self) -> dict:
        """生成统计报告"""
        if not self.requests:
            return {"error": "No requests recorded"}

        total = len(self.requests)
        success = sum(1 for r in self.requests if r["success"])
        failed = total - success

        latencies = [r["latency_ms"] for r in self.requests if r["success"]]
        duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0

        # 计算各百分位延迟
        if latencies:
            sorted_latencies = sorted(latencies)
            p50 = sorted_latencies[len(sorted_latencies) // 2]
            p90 = sorted_latencies[int(len(sorted_latencies) * 0.9)]
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
            avg = statistics.mean(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)
        else:
            p50, p90, p95, p99, avg, min_lat, max_lat = 0, 0, 0, 0, 0, 0, 0

        # QPS = 成功请求数 / 总时间
        qps = success / duration if duration > 0 else 0

        return {
            "total_requests": total,
            "success_requests": success,
            "failed_requests": failed,
            "success_rate": f"{(success / total * 100):.2f}%",
            "duration_seconds": f"{duration:.2f}",
            "qps": f"{qps:.2f}",
            "latency_avg_ms": f"{avg:.2f}",
            "latency_p50_ms": f"{p50:.2f}",
            "latency_p90_ms": f"{p90:.2f}",
            "latency_p95_ms": f"{p95:.2f}",
            "latency_p99_ms": f"{p99:.2f}",
            "latency_min_ms": f"{min_lat:.2f}",
            "latency_max_ms": f"{max_lat:.2f}",
            "error_samples": self.errors[:5] if self.errors else [],
        }


# ============================================================
# 搜索接口压测
# ============================================================

async def search_request(
    client: httpx.AsyncClient,
    host: str,
    kb_id: int,
    stats: BenchmarkStats,
    use_rerank: bool = True,
) -> None:
    """执行一次搜索请求"""
    query = generate_query()
    url = f"{host}/api/v1/search"
    payload = {
        "kb_id": str(kb_id),  # kb_id 必须是字符串类型
        "query": query,
        "top_k": 5,
        "use_rerank": use_rerank,
    }

    start = time.time()
    try:
        resp = await client.post(url, json=payload, timeout=30.0)
        latency_ms = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                stats.record(latency_ms, True)
            else:
                stats.record(latency_ms, False, data.get("message", "Unknown error"))
        else:
            stats.record(latency_ms, False, f"HTTP {resp.status_code}")

    except httpx.TimeoutException:
        latency_ms = (time.time() - start) * 1000
        stats.record(latency_ms, False, "Timeout")

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        stats.record(latency_ms, False, str(e))


async def faq_search_request(
    client: httpx.AsyncClient,
    host: str,
    kb_id: int,
    stats: BenchmarkStats,
) -> None:
    """执行一次 FAQ 搜索请求"""
    query = generate_query()
    url = f"{host}/api/v1/faq/search"
    payload = {
        "kb_id": kb_id,
        "query": query,
        "top_k": 5,
        "use_rerank": True,
    }

    start = time.time()
    try:
        resp = await client.post(url, json=payload, timeout=30.0)
        latency_ms = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                stats.record(latency_ms, True)
            else:
                stats.record(latency_ms, False, data.get("message", "Unknown error"))
        else:
            stats.record(latency_ms, False, f"HTTP {resp.status_code}")

    except httpx.TimeoutException:
        latency_ms = (time.time() - start) * 1000
        stats.record(latency_ms, False, "Timeout")

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        stats.record(latency_ms, False, str(e))


# ============================================================
# 并发压测主函数
# ============================================================

async def run_benchmark(
    host: str,
    kb_id: int,
    duration: int,
    concurrency: int,
    scenario: str = "search",
    use_rerank: bool = True,
) -> BenchmarkStats:
    """
    运行并发压测

    参数：
        host: 后端地址
        kb_id: 知识库 ID
        duration: 压测时长（秒）
        concurrency: 并发数
        scenario: 压测场景（search/faq）
    """
    stats = BenchmarkStats()
    stats.start_time = time.time()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 创建并发任务
        tasks = []
        end_time = stats.start_time + duration

        while time.time() < end_time:
            # 保持固定并发数
            for _ in range(concurrency):
                if time.time() >= end_time:
                    break
                if scenario == "faq":
                    task = faq_search_request(client, host, kb_id, stats)
                else:
                    task = search_request(client, host, kb_id, stats, use_rerank)
                tasks.append(task)

            # 执行一批请求
            if tasks:
                await asyncio.gather(*tasks)
                tasks = []

            # 短暂间隔，避免瞬间爆发
            await asyncio.sleep(0.1)

    stats.end_time = time.time()
    return stats


# ============================================================
# 渐进式压测（逐步增加并发）
# ============================================================

async def run_progressive_benchmark(
    host: str,
    kb_id: int,
    max_concurrency: int = 50,
    step: int = 5,
    duration_per_step: int = 30,
    scenario: str = "search",
    use_rerank: bool = True,
) -> list[dict]:
    """
    渐进式压测：逐步增加并发，找出系统极限

    参数：
        host: 后端地址
        kb_id: 知识库 ID
        max_concurrency: 最大并发数
        step: 每步增加的并发数
        duration_per_step: 每步压测时长
        scenario: 压测场景

    返回：
        各并发级别的统计报告
    """
    results = []

    for concurrency in range(step, max_concurrency + 1, step):
        print(f"\n{'='*50}")
        print(f"并发数: {concurrency}")
        print(f"{'='*50}")

        stats = await run_benchmark(
            host=host,
            kb_id=kb_id,
            duration=duration_per_step,
            concurrency=concurrency,
            scenario=scenario,
            use_rerank=use_rerank,
        )

        report = stats.report()
        results.append({
            "concurrency": concurrency,
            **report,
        })

        # 打印当前级别报告
        print(f"总请求: {report['total_requests']}")
        print(f"成功率: {report['success_rate']}")
        print(f"QPS: {report['qps']}")
        print(f"P50 延迟: {report['latency_p50_ms']} ms")
        print(f"P95 延迟: {report['latency_p95_ms']} ms")
        print(f"P99 延迟: {report['latency_p99_ms']} ms")

        # 如果成功率低于 90%，停止压测
        success_rate = float(report['success_rate'].replace('%', ''))
        if success_rate < 90:
            print(f"\n成功率低于 90%，停止压测")
            break

    return results


# ============================================================
# 日活计算
# ============================================================

def calculate_daily_capacity(qps: float) -> dict:
    """
    根据实测 QPS 计算系统日活支撑能力

    假设：
    - 每个活跃用户每天发起 5-10 次搜索请求
    - 系统每天运行 8-12 小时（工作时间）
    - 系统高峰期 QPS 是平均 QPS 的 2-3 倍

    参数：
        qps: 实测稳定 QPS

    返回：
        日活支撑能力估算
    """
    # 不同假设下的计算
    scenarios = []

    # 场景 1：保守估算（高峰期压力大）
    # 每用户 10 次请求，高峰 QPS = 平均 QPS × 3
    avg_qps_1 = qps / 3
    daily_requests_1 = avg_qps_1 * 3600 * 8  # 8 小时工作时间
    daily_active_1 = daily_requests_1 / 10

    scenarios.append({
        "name": "保守估算",
        "假设": "高峰 QPS 3倍，每用户 10 次请求，8 小时工作时间",
        "平均 QPS": f"{avg_qps_1:.2f}",
        "日请求数": f"{daily_requests_1:.0f}",
        "日活支撑": f"{daily_active_1:.0f} 人",
    })

    # 场景 2：中等估算
    avg_qps_2 = qps / 2
    daily_requests_2 = avg_qps_2 * 3600 * 10
    daily_active_2 = daily_requests_2 / 7

    scenarios.append({
        "name": "中等估算",
        "假设": "高峰 QPS 2倍，每用户 7 次请求，10 小时工作时间",
        "平均 QPS": f"{avg_qps_2:.2f}",
        "日请求数": f"{daily_requests_2:.0f}",
        "日活支撑": f"{daily_active_2:.0f} 人",
    })

    # 场景 3：乐观估算（流量均匀分布）
    avg_qps_3 = qps
    daily_requests_3 = avg_qps_3 * 3600 * 12
    daily_active_3 = daily_requests_3 / 5

    scenarios.append({
        "name": "乐观估算",
        "假设": "流量均匀，每用户 5 次请求，12 小时工作时间",
        "平均 QPS": f"{avg_qps_3:.2f}",
        "日请求数": f"{daily_requests_3:.0f}",
        "日活支撑": f"{daily_active_3:.0f} 人",
    })

    return {
        "实测高峰 QPS": f"{qps:.2f}",
        "估算场景": scenarios,
        "建议": "实际部署建议取保守估算，预留 30% 缓冲",
    }


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RAG 系统压测")
    parser.add_argument("--host", default="http://localhost:8888", help="后端地址")
    parser.add_argument("--kb-id", type=int, required=True, help="知识库 ID")
    parser.add_argument("--duration", type=int, default=60, help="压测时长（秒）")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    parser.add_argument("--scenario", choices=["search", "faq"], default="search", help="压测场景")
    parser.add_argument("--no-rerank", action="store_true", help="禁用 rerank（测试纯向量检索性能）")
    parser.add_argument("--progressive", action="store_true", help="渐进式压测")
    parser.add_argument("--max-concurrency", type=int, default=50, help="渐进式压测最大并发")
    parser.add_argument("--step", type=int, default=5, help="渐进式压测步长")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"RAG 系统压测")
    print(f"{'='*60}")
    print(f"后端地址: {args.host}")
    print(f"知识库 ID: {args.kb_id}")
    print(f"压测场景: {args.scenario}")
    print(f"Rerank: {'禁用' if args.no_rerank else '启用'}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    use_rerank = not args.no_rerank

    if args.progressive:
        # 渐进式压测
        results = asyncio.run(
            run_progressive_benchmark(
                host=args.host,
                kb_id=args.kb_id,
                max_concurrency=args.max_concurrency,
                step=args.step,
                duration_per_step=args.duration,
                scenario=args.scenario,
                use_rerank=use_rerank,
            )
        )

        # 最终报告
        print(f"\n{'='*60}")
        print(f"渐进式压测完整报告")
        print(f"{'='*60}")

        # 找出最佳 QPS（成功率 > 95% 的最高并发）
        best_qps = 0
        for r in results:
            success_rate = float(r['success_rate'].replace('%', ''))
            qps = float(r['qps'])
            if success_rate >= 95 and qps > best_qps:
                best_qps = qps

        if best_qps > 0:
            capacity = calculate_daily_capacity(best_qps)
            print(f"\n稳定 QPS: {best_qps:.2f}")
            print(f"\n日活支撑能力估算:")
            for scenario in capacity['估算场景']:
                print(f"\n{scenario['name']}:")
                print(f"  {scenario['假设']}")
                print(f"  日活支撑: {scenario['日活支撑']}")
                print(f"  日请求数: {scenario['日请求数']}")

            print(f"\n{capacity['建议']}")

        # 保存详细报告
        report_file = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n详细报告已保存: {report_file}")

    else:
        # 单次压测
        stats = asyncio.run(
            run_benchmark(
                host=args.host,
                kb_id=args.kb_id,
                duration=args.duration,
                concurrency=args.concurrency,
                scenario=args.scenario,
                use_rerank=use_rerank,
            )
        )

        report = stats.report()

        print(f"\n{'='*60}")
        print(f"压测报告")
        print(f"{'='*60}")
        for key, value in report.items():
            if key != "error_samples":
                print(f"{key}: {value}")

        if report.get("error_samples"):
            print(f"\n错误样本:")
            for err in report["error_samples"]:
                print(f"  - {err}")

        # 日活估算
        if float(report['success_rate'].replace('%', '')) >= 95:
            qps = float(report['qps'])
            capacity = calculate_daily_capacity(qps)
            print(f"\n{'='*60}")
            print(f"日活支撑能力估算")
            print(f"{'='*60}")
            for scenario in capacity['估算场景']:
                print(f"\n{scenario['name']}: {scenario['日活支撑']}")
                print(f"  假设: {scenario['假设']}")
                print(f"  日请求数: {scenario['日请求数']}")


if __name__ == "__main__":
    main()