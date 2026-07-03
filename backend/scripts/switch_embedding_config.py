#!/usr/bin/env python3
"""
切换 Embedding/Rerank 配置（保留其他配置不变）

使用方法：
    python scripts/switch_embedding_config.py gpu        # 切换到本地 GPU 服务
    python scripts/switch_embedding_config.py dashscope  # 切换到 DashScope 云服务
    python scripts/switch_embedding_config.py show       # 显示当前配置
"""
import argparse
import re

ENV_FILE = ".env"

# 需要管理的配置项
CONFIG_KEYS = [
    "DASHSCOPE_API_KEY", "EMBED_MODEL_NAME", "EMBED_BASE_URL", "EMBED_API_URL",
    "UNIFIED_VECTOR_DIM", "RERANK_ENABLED", "RERANK_API_URL", "RERANK_MODEL_NAME", "RERANK_TOP_N",
]

# GPU 配置
GPU_CONFIG = {
    "DASHSCOPE_API_KEY": "",
    "EMBED_MODEL_NAME": "Qwen/Qwen3-VL-Embedding-2B",
    "EMBED_BASE_URL": "http://127.0.0.1:6006/embedding/v1",
    "EMBED_API_URL": "http://127.0.0.1:6006/embedding/v1/embeddings",
    "UNIFIED_VECTOR_DIM": "2048",
    "RERANK_ENABLED": "true",
    "RERANK_API_URL": "http://127.0.0.1:6006/reranker/v1/rerank",
    "RERANK_MODEL_NAME": "Qwen/Qwen3-VL-Reranker-2B",
    "RERANK_TOP_N": "50",
}

# DashScope 配置
DASHSCOPE_CONFIG = {
    "DASHSCOPE_API_KEY": "sk-7e7fc775b6ba41dcb7a42028dc2112fc",
    "EMBED_MODEL_NAME": "qwen3-vl-embedding",
    "EMBED_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "EMBED_API_URL": "",
    "UNIFIED_VECTOR_DIM": "2560",
    "RERANK_ENABLED": "true",
    "RERANK_API_URL": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    "RERANK_MODEL_NAME": "qwen3-vl-rerank",
    "RERANK_TOP_N": "50",
}


def read_env() -> list[str]:
    """读取 .env 文件为行列表"""
    with open(ENV_FILE, "r") as f:
        return f.readlines()


def write_env(lines: list[str]):
    """写入 .env 文件"""
    with open(ENV_FILE, "w") as f:
        f.writelines(lines)


def remove_config_lines(lines: list[str]) -> list[str]:
    """删除所有 Embedding/Rerank 相关配置行"""
    result = []
    skip_block = False

    for line in lines:
        stripped = line.strip()

        # 检测配置块注释开始
        if "# Embedding" in stripped or "# Reranking" in stripped or "# Embedding/Rerank" in stripped:
            skip_block = True
            continue

        # 检测配置项
        is_config_line = any(stripped.startswith(f"{key}=") for key in CONFIG_KEYS)

        # 遇到新的非配置注释块，结束跳过
        if skip_block and stripped.startswith("#") and "# Embedding" not in stripped and "# Reranking" not in stripped:
            skip_block = False

        # 非配置块内的配置项也要删除
        if is_config_line:
            continue

        if not skip_block:
            result.append(line)

    return result


def add_config_block(lines: list[str], config: dict, mode: str) -> list[str]:
    """添加配置块"""
    # 确保文件末尾有空行
    while lines and lines[-1].strip() == "":
        lines.pop()

    lines.append("\n")
    lines.append("# ============================================================\n")

    if mode == "gpu":
        lines.append("# Embedding/Rerank 配置（当前使用本地 GPU 服务）\n")
        lines.append("# ============================================================\n\n")
        lines.append("# Embedding (本地 GPU 服务)\n")
    else:
        lines.append("# Embedding/Rerank 配置（当前使用 DashScope 云服务）\n")
        lines.append("# ============================================================\n\n")
        lines.append("# Embedding (DashScope 多模态 API)\n")

    # Embedding 配置
    for key in ["DASHSCOPE_API_KEY", "EMBED_MODEL_NAME", "EMBED_BASE_URL", "EMBED_API_URL", "UNIFIED_VECTOR_DIM"]:
        lines.append(f"{key}={config[key]}\n")

    lines.append("\n")
    if mode == "gpu":
        lines.append("# Reranking (本地 GPU 服务)\n")
    else:
        lines.append("# Reranking (DashScope Rerank API)\n")

    # Rerank 配置
    for key in ["RERANK_ENABLED", "RERANK_API_URL", "RERANK_MODEL_NAME", "RERANK_TOP_N"]:
        lines.append(f"{key}={config[key]}\n")

    return lines


def switch_config(mode: str):
    """切换配置"""
    config = GPU_CONFIG if mode == "gpu" else DASHSCOPE_CONFIG

    lines = read_env()
    lines = remove_config_lines(lines)
    lines = add_config_block(lines, config, mode)
    write_env(lines)

    dim = config["UNIFIED_VECTOR_DIM"]
    embed_model = config["EMBED_MODEL_NAME"]
    rerank_model = config["RERANK_MODEL_NAME"]

    print(f"\n✅ 已切换到 {mode} 模式")
    print(f"   向量维度: {dim}")
    print(f"   Embed 模型: {embed_model}")
    print(f"   Rerank 模型: {rerank_model}")
    print(f"\n⚠️  注意：向量维度变化，需要重新导入数据到新的知识库！\n")


def show_config():
    """显示当前配置"""
    lines = read_env()
    content = "".join(lines)

    print("\n当前 Embedding/Rerank 配置：")
    print("-" * 50)

    for key in CONFIG_KEYS:
        match = re.search(rf"^{key}=(.*)$", content, re.MULTILINE)
        if match:
            value = match.group(1).strip()
            if "API_KEY" in key and value and len(value) > 10:
                value = value[:10] + "..."
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: (未设置)")

    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="切换 Embedding/Rerank 配置")
    parser.add_argument("mode", nargs="?", choices=["gpu", "dashscope", "show"],
                        default="show", help="切换模式 (gpu/dashscope/show)")
    args = parser.parse_args()

    if args.mode == "show":
        show_config()
    else:
        switch_config(args.mode)


if __name__ == "__main__":
    main()