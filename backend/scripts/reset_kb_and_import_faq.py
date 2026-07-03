#!/usr/bin/env python3
"""
清空知识库数据并导入模拟 FAQ 数据

使用方法：
    python scripts/reset_kb_and_import_faq.py --kb-id 30 --count 500 --host http://localhost:8888
"""
import argparse
import json
import random
import requests
import time


# ============================================================
# 模拟 QA 数据生成
# ============================================================

QA_TEMPLATE = [
    ("如何处理函数执行超时？", "函数执行超时通常由以下原因导致：1. 算法复杂度过高；2. 数据量过大；3. 网络请求阻塞。建议优化算法、增加超时处理机制、使用异步执行。"),
    ("数据库连接失败怎么解决？", "数据库连接失败可能原因：1. 数据库服务未启动；2. 连接参数错误；3. 网络不通；4. 连接数超限。请检查数据库状态、确认连接配置、测试网络连通性。"),
    ("API 返回 500 错误如何排查？", "500 错误是服务器内部错误。排查步骤：1. 查看服务器日志；2. 检查异常堆栈；3. 确认数据库连接；4. 检查内存/CPU 状态。常见原因：空指针、数据库异常、配置错误。"),
    ("如何优化查询性能？", "查询性能优化方法：1. 使用索引；2. 优化 SQL 语句；3. 减少返回字段；4. 使用缓存；5. 分库分表；6. 读写分离。建议先分析慢查询日志。"),
    ("系统内存不足如何处理？", "内存不足处理方案：1. 检查内存泄漏；2. 优化数据结构；3. 使用对象池；4. 增加缓存淘汰策略；5. 调整 JVM/Python 内存参数；6. 扩容服务器内存。"),
    ("日志文件过大怎么清理？", "日志清理方案：1. 使用日志轮转（logrotate）；2. 设置日志级别；3. 定期归档删除；4. 使用日志收集系统（ELK）；5. 配置日志大小上限。"),
    ("如何配置 HTTPS？", "HTTPS 配置步骤：1. 获取 SSL 证书（自签名或 CA 签发）；2. 配置 Nginx/Apache SSL 参数；3. 开启 443 端口；4. 强制 HTTP 重定向 HTTPS；5. 测试证书有效性。"),
    ("服务启动失败怎么排查？", "服务启动失败排查：1. 检查端口占用；2. 查看启动日志；3. 确认配置文件；4. 检查依赖服务（数据库/Redis）；5. 确认权限问题；6. 检查环境变量。"),
    ("如何实现分布式锁？", "分布式锁实现方案：1. Redis SETNX + 过期时间；2. ZooKeeper 临时节点；3. 数据库唯一索引；4. Redisson 等框架。注意：锁超时、死锁、锁释放问题。"),
    ("缓存策略有哪些？", "缓存策略类型：1. 缓存穿透（布隆过滤器）；2. 缓存击穿（热点数据预热）；3. 缓存雪崩（过期时间随机）；4. 缓存更新（定时/主动）；5. 缓存淘汰（LRU/LFU）。"),
    ("消息队列如何选型？", "消息队列选型对比：Kafka（高吞吐、持久化）、RabbitMQ（可靠、功能丰富）、RocketMQ（阿里开源、事务消息）、Redis Stream（轻量）。根据业务需求选择。"),
    ("如何设计微服务架构？", "微服务设计原则：1. 服务拆分粒度；2. API 网关；3. 服务注册发现；4. 配置中心；5. 链路追踪；6. 熔断降级；7. 认证授权；8. 日志监控。"),
    ("容器化部署有什么注意事项？", "容器化注意事项：1. 镜像大小优化；2. 资源限制配置；3. 健康检查；4. 日志输出；5. 配置管理；6. 网络模式；7. 存储卷挂载；8. 安全隔离。"),
    ("如何监控服务健康状态？", "服务监控方案：1. Prometheus + Grafana；2. ELK 日志分析；3. 链路追踪（SkyWalking）；4. 健康检查接口；5. 告警系统；6. 性能指标采集。"),
    ("安全认证有哪些方案？", "认证方案对比：1. JWT（无状态、适合微服务）；2. Session + Cookie（传统 Web）；3. OAuth2（第三方授权）；4. SAML（企业级）；5. API Key（简单场景）。"),
]


def generate_qa_pairs(count: int) -> list[dict]:
    """生成模拟 QA 数据"""
    items = []
    for i in range(count):
        q, a = random.choice(QA_TEMPLATE)
        # 添加变化使数据更真实
        variations = [
            {"question": q, "answer": a},
            {"question": q + "请详细说明", "answer": a + "如有其他问题请联系技术支持。"},
            {"question": q.replace("如何", "怎么"), "answer": a},
            {"question": q.replace("？", ""), "answer": a[:len(a)//2] + "..."},
            {"question": f"关于{q[:10]}的问题", "answer": a + f"（FAQ-{i+1}）"},
        ]
        items.append(random.choice(variations))
    return items


# ============================================================
# API 调用
# ============================================================

def api_get(host: str, path: str) -> dict:
    """GET 请求"""
    resp = requests.get(f"{host}{path}", timeout=30)
    return resp.json()


def api_delete(host: str, path: str) -> dict:
    """DELETE 请求"""
    resp = requests.delete(f"{host}{path}", timeout=30)
    return resp.json()


def api_post(host: str, path: str, data: dict) -> dict:
    """POST 请求"""
    resp = requests.post(f"{host}{path}", json=data, timeout=60)
    return resp.json()


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="清空知识库并导入模拟 FAQ 数据")
    parser.add_argument("--kb-id", type=int, default=30, help="知识库 ID")
    parser.add_argument("--count", type=int, default=500, help="导入 FAQ 数量")
    parser.add_argument("--host", default="http://localhost:8888", help="服务地址")
    parser.add_argument("--skip-delete", action="store_true", help="跳过删除步骤")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"清空知识库并导入 FAQ 数据")
    print(f"{'='*60}")
    print(f"服务地址: {args.host}")
    print(f"知识库 ID: {args.kb_id}")
    print(f"导入数量: {args.count}")
    print(f"{'='*60}\n")

    # 1. 查看现有知识库
    print("[1] 查看现有知识库...")
    kb_list = api_get(args.host, "/api/v1/kb")
    if kb_list.get("success"):
        print(f"    现有知识库数量: {kb_list['data']['total']}")
        for kb in kb_list['data']['result']:
            print(f"    - ID={kb['id']}, Name={kb['name']}, Type={kb['kb_type']}")
    else:
        print(f"    获取失败: {kb_list.get('message')}")
        return

    # 2. 删除知识库（如果存在）
    if not args.skip_delete:
        print(f"\n[2] 删除知识库 kb_id={args.kb_id}...")
        result = api_delete(args.host, f"/api/v1/kb/{args.kb_id}")
        if result.get("success"):
            print(f"    删除成功: {result['data']}")
        else:
            print(f"    删除失败或不存在: {result.get('message')}")
            # 检查是否不存在
            if "不存在" in result.get("message", "") or "Not Found" in result.get("message", ""):
                print("    知识库不存在，继续创建...")
        time.sleep(1)

    # 3. 创建新的 FAQ 知识库
    print(f"\n[3] 创建新的 FAQ 知识库...")
    result = api_post(args.host, "/api/v1/kb", {
        "name": f"压测FAQ库_{args.kb_id}",
        "kb_type": "faq",
        "description": "用于压测的模拟 FAQ 数据",
    })
    if not result.get("success"):
        print(f"    创建失败: {result.get('message')}")
        return

    new_kb_id = result['data']['id']
    collection_name = result['data']['collection_name']
    print(f"    创建成功: kb_id={new_kb_id}, collection={collection_name}")

    # 4. 生成模拟 QA 数据
    print(f"\n[4] 生成 {args.count} 条模拟 QA 数据...")
    qa_items = generate_qa_pairs(args.count)
    print(f"    生成完成")

    # 5. 分批导入（每批 50 条）
    print(f"\n[5] 分批导入 FAQ 数据...")
    batch_size = 50
    total_batches = (args.count + batch_size - 1) // batch_size
    success_count = 0
    failed_count = 0

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, args.count)
        batch_items = qa_items[start_idx:end_idx]

        print(f"    导入批次 {batch_idx + 1}/{total_batches} ({start_idx + 1}-{end_idx})...")
        result = api_post(args.host, "/api/v1/faq/batch", {
            "kb_id": new_kb_id,
            "items": batch_items,
            "creator": "import_script",
        })

        if result.get("success"):
            data = result['data']
            success_count += data['success']
            failed_count += data['failed']
            print(f"    成功: {data['success']}, 失败: {data['failed']}")
        else:
            failed_count += len(batch_items)
            print(f"    批次失败: {result.get('message')}")

        # 避免请求过快
        time.sleep(0.5)

    # 6. 验证导入结果
    print(f"\n[6] 验证导入结果...")
    result = api_get(args.host, f"/api/v1/faq?kb_id={new_kb_id}&size=1")
    if result.get("success"):
        total_faqs = result['data']['total']
        print(f"    FAQ 总数: {total_faqs}")

    # 完成
    print(f"\n{'='*60}")
    print(f"导入完成")
    print(f"{'='*60}")
    print(f"知识库 ID: {new_kb_id}")
    print(f"Collection: {collection_name}")
    print(f"导入成功: {success_count}")
    print(f"导入失败: {failed_count}")
    print(f"\n压测命令:")
    print(f"python scripts/benchmark.py --host {args.host} --kb-id {new_kb_id} --progressive --max-concurrency 50 --step 5 --duration 30")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()