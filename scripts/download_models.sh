#!/bin/bash
# 预拉 GEM-X demo 所需 HF 权重 (nvidia/GEM-X), 复用仓库 gem/utils/hf_utils.py。
# 权重落到本仓库 inputs/ 下, 与代码默认路径一致, 让 demo 跳过运行期下载。
#
# 用法: bash scripts/download_models.sh [proxy]
#   proxy: baidu (默认) | aliyun (HF 快)
#
# 可选环境变量:
#   GEMX_ENV_DIR  虚拟环境路径 (默认见下方; 与 install.sh 保持一致)
#
# 下载清单 (HF repo nvidia/GEM-X):
#   gem_soma.ckpt                    -> inputs/pretrained/
#   vitpose.pth                      -> inputs/checkpoints/vitpose/
#   sam3d_body.ckpt + config.yaml    -> inputs/checkpoints/sam-3d-body-dinov3/
#   mhr_model.pt                     -> inputs/mhr_data/
#   scale_mean.pth + scale_comps.pth -> inputs/soma_data/
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_DIR="${GEMX_ENV_DIR:-/root/paddlejob/workspace/env_run/penghaotian/envs/gemx}"
PROXY="${1:-baidu}"

if [ "$PROXY" = "aliyun" ]; then
    export https_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
    export http_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
else
    export https_proxy=http://agent.baidu.com:8188
    export http_proxy=http://agent.baidu.com:8188
fi
echo "[proxy] $PROXY"
echo "[paths] REPO_ROOT=$REPO_ROOT  ENV_DIR=$ENV_DIR"

PYTHON="$ENV_DIR/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] 虚拟环境不存在: $ENV_DIR"
    echo "        先运行 bash scripts/install.sh, 或设 GEMX_ENV_DIR 指向已有环境。"
    exit 1
fi

cd "$REPO_ROOT"
echo "[download] 调用 gem/utils/hf_utils.py 预拉权重 (已存在则跳过)..."
"$PYTHON" - <<'PY'
from gem.utils.hf_utils import (
    download_checkpoint,
    download_vitpose_checkpoint,
    download_sam3d_checkpoint,
    download_mhr_model,
    download_soma_data,
)

print("[1/5] gem_soma.ckpt ...");        print("  ->", download_checkpoint())
print("[2/5] vitpose.pth ...");          print("  ->", download_vitpose_checkpoint())
print("[3/5] sam3d_body.ckpt ...");      print("  ->", download_sam3d_checkpoint())
print("[4/5] mhr_model.pt ...");         print("  ->", download_mhr_model())
print("[5/5] soma scale data ...");      print("  ->", download_soma_data())
print("[OK] 全部权重就位")
PY

echo
echo "[ckpt] 当前 inputs/ 结构:"
find "$REPO_ROOT/inputs" -maxdepth 2 \( -type f -o -type l \) -printf '  %p\n' 2>/dev/null | sort
echo
echo "============================================================"
echo " 权重下载完成。下一步: bash scripts/run_demo.sh"
echo "============================================================"
