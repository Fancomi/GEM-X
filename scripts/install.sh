#!/bin/bash
# GEM-X 一键安装 (虚拟环境 + torch + 子模块 + GEM/依赖 + SOMA assets)
# 自包含: 仅依赖本仓库自身 (跳过 soma-retargeter SSH 子模块与 --retarget)。
#
# 用法: bash scripts/install.sh [proxy]
#   proxy: baidu (默认, PIP 国内快) | aliyun (HF 快)
#
# 可选环境变量:
#   GEMX_ENV_DIR  虚拟环境路径 (默认见下方; 换机器/换人请改这里或设此变量)
#   CUDA_HOME     CUDA 安装路径 (默认 /usr/local/cuda)
#
# 装完后: (1) bash scripts/download_models.sh  下权重
#         (2) bash scripts/run_demo.sh         跑 taiji.mp4 验证
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_DIR="${GEMX_ENV_DIR:-/root/paddlejob/workspace/env_run/penghaotian/envs/gemx}"
PROXY="${1:-baidu}"

if [ "$PROXY" = "aliyun" ]; then
    export https_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
    export http_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
    PIP_INDEX="https://mirrors.aliyun.com/pypi/simple/"
else
    export https_proxy=http://agent.baidu.com:8188
    export http_proxy=http://agent.baidu.com:8188
    PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple/"
fi
echo "[proxy] $PROXY  PIP_INDEX=$PIP_INDEX"
echo "[paths] REPO_ROOT=$REPO_ROOT  ENV_DIR=$ENV_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "[cuda] CUDA_HOME=$CUDA_HOME"

command -v uv >/dev/null 2>&1 || { echo "[ERROR] 需要 uv, 先装: https://docs.astral.sh/uv/"; exit 1; }

echo "[1/7] 创建虚拟环境 (python 3.10)"
uv venv "$ENV_DIR" --python 3.10 2>/dev/null || true
PYTHON="$ENV_DIR/bin/python"
UV_INSTALL="uv pip install --python $PYTHON --link-mode=copy"

echo "[2/7] PyTorch 2.10.0 + torchvision 0.25.0 (cu126)"
$UV_INSTALL torch==2.10.0+cu126 torchvision==0.25.0+cu126 --index-url https://download.pytorch.org/whl/cu126

echo "[3/7] 初始化子模块 soma + sam-3d-body (跳过 soma-retargeter SSH)"
cd "$REPO_ROOT"
git submodule update --init third_party/soma third_party/sam-3d-body

echo "[4/7] 安装 SOMA body model + 拉 LFS assets"
$UV_INSTALL -e third_party/soma -i "$PIP_INDEX"
( cd third_party/soma && git lfs pull )

echo "[5/7] editable 安装 gem 本仓库"
$UV_INSTALL -e . -i "$PIP_INDEX"

echo "[6/7] SAM-3D-Body 运行时依赖 + detectron2"
$UV_INSTALL -i "$PIP_INDEX" cloudpickle fvcore iopath pycocotools braceexpand roma 'setuptools<75'
$UV_INSTALL 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

echo "[7/7] 链接 SOMA assets -> inputs/soma_assets"
mkdir -p "$REPO_ROOT/inputs"
ln -sfn "$REPO_ROOT/third_party/soma/assets" "$REPO_ROOT/inputs/soma_assets"

echo
echo "============================================================"
echo " 依赖安装完成 (retarget 已跳过, 仅核心 3D pipeline)"
echo " 虚拟环境: $ENV_DIR"
echo
echo " 下一步:"
echo "   1) 下载权重:  bash scripts/download_models.sh $PROXY"
echo "   2) 跑 demo:    bash scripts/run_demo.sh"
echo "============================================================"
