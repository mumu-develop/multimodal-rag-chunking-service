#!/bin/bash
# DeepSeek-OCR 环境一键准备脚本 (不包含 vLLM 服务启动)
# 适用于 AutoDL 等数据盘挂载在 /root/autodl-tmp 的环境

set -e  # 遇到错误即退出

# 定义数据盘路径
DATA_DIR="/root/autodl-tmp"
CONDA_ENV_DIR="$DATA_DIR/conda_envs/deepseek-ocr"
MODEL_DIR="$DATA_DIR/models/DeepSeek-OCR"

echo "============================================"
echo "DeepSeek-OCR 环境准备脚本"
echo "数据盘路径: $DATA_DIR"
echo "Conda 环境路径: $CONDA_ENV_DIR"
echo "模型下载路径: $MODEL_DIR"
echo "============================================"

# 1. 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 Miniconda"
    exit 1
fi
echo "✅ Conda 已安装"

# 2. 清理显存（可选，注释掉，用户可自行执行）
echo "提示: 如果显存被占用，请手动执行 pkill -9 python 或 nvidia-smi 查看"
# 这里不做自动清理，避免误杀

# 3. 创建 conda 环境（如果不存在）
if [ -d "$CONDA_ENV_DIR" ]; then
    echo "Conda 环境已存在: $CONDA_ENV_DIR"
else
    echo "创建 Conda 环境: $CONDA_ENV_DIR (Python 3.12.9)"
    mkdir -p "$DATA_DIR/conda_envs"
    conda create -p "$CONDA_ENV_DIR" python=3.12.9 -y
    echo "✅ Conda 环境创建完成"
fi

# 4. 激活环境并安装依赖
echo "激活 Conda 环境并安装依赖..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_DIR"

# 安装 PyTorch (官方源，必须)
echo "安装 PyTorch 2.6.0 (CUDA 11.8)..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 配置 pip 永久使用清华源
echo "配置 pip 使用清华源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装其他依赖
echo "安装其他依赖 (transformers, vLLM, 工具库)..."
pip install transformers==4.51.1 sentencepiece pillow pymupdf tqdm vllm==0.11.2

echo "✅ 所有依赖安装完成"

# 5. 下载模型（如果未下载）
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    echo "模型已存在: $MODEL_DIR"
else
    echo "下载模型到 $MODEL_DIR (约 7GB，请耐心等待)..."
    mkdir -p "$DATA_DIR/models"
    pip install modelscope
    modelscope download --model deepseek-ai/DeepSeek-OCR --local_dir "$MODEL_DIR"
    echo "✅ 模型下载完成"
fi

# 6. 输出完成信息及启动命令
echo "============================================"
echo "🎉 环境准备完成！"
echo ""
echo "请执行以下命令启动 vLLM 服务（前台运行）:"
echo "============================================"
echo "conda activate $CONDA_ENV_DIR"
echo "vllm serve $MODEL_DIR \\"
echo "  --host 0.0.0.0 \\"
echo "  --port 8000 \\"
echo "  --served-model-name deepseek-ocr \\"
echo "  --gpu-memory-utilization 0.6 \\"
echo "  --trust-remote-code \\"
echo "  --allowed-local-media-path / \\"
echo "  --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \\"
echo "  --no-enable-prefix-caching \\"
echo "  --mm-processor-cache-gb 0"
echo "============================================"
echo ""
echo "提示: 如果显存不足，可降低 --gpu-memory-utilization 到 0.5"
echo "若要后台运行，建议使用 tmux: tmux new -s vllm"