#!/bin/bash
# 跑 GEM-X 全 3D pipeline demo (demo_soma.py), 默认用 taiji.mp4 验证。
# 自包含: 仅依赖本仓库自身。静态相机模式 (-s), VO/SLAM 未装。
#
# 用法: bash scripts/run_demo.sh [video_path] [gpu_id]
#   video_path: 输入视频 (默认 taiji.mp4)
#   gpu_id:     CUDA 设备号 (默认 0)
#
# 可选环境变量:
#   GEMX_ENV_DIR  虚拟环境路径 (默认见下方; 与 install.sh 保持一致)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_DIR="${GEMX_ENV_DIR:-/root/paddlejob/workspace/env_run/penghaotian/envs/gemx}"

VIDEO="${1:-/root/paddlejob/workspace/env_run/penghaotian/datas/Test/taiji.mp4}"
GPU_ID="${2:-0}"

PYTHON="$ENV_DIR/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] 虚拟环境不存在: $ENV_DIR"
    echo "        先运行 bash scripts/install.sh, 或设 GEMX_ENV_DIR 指向已有环境。"
    exit 1
fi
if [ ! -f "$VIDEO" ]; then
    echo "[ERROR] 视频不存在: $VIDEO"
    exit 1
fi

# headless 离屏渲染 (open3d OffscreenRenderer 需要 EGL)
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
# CUDA 运行期 lib 可见 (cu126 wheel 自带; 保险设一下 nvcc 路径)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

cd "$REPO_ROOT"
VIDEO_STEM="$(basename "$VIDEO")"
VIDEO_STEM="${VIDEO_STEM%.*}"
echo "[run] GPU=$GPU_ID  video=$VIDEO  static_cam=on"
echo "[run] 输出将落在 outputs/$VIDEO_STEM/"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$REPO_ROOT/scripts/demo/demo_soma.py" \
    --video "$VIDEO" -s --output_root "$REPO_ROOT/outputs"

OUT="$REPO_ROOT/outputs/$VIDEO_STEM/${VIDEO_STEM}_3_incam_global_horiz.mp4"
echo
if [ -s "$OUT" ]; then
    echo "============================================================"
    echo " [OK] demo 完成, 结果视频: $OUT ($(du -h "$OUT" | cut -f1))"
    echo " 其他输出见 outputs/$VIDEO_STEM/"
    echo "============================================================"
else
    echo "[ERROR] 未生成结果视频: $OUT (检查上面日志)"
    exit 1
fi
