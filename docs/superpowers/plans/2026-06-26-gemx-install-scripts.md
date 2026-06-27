# GEM-X 安装脚本实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 GEM-X 编写三个自包含 bash 脚本 (install / download_models / run_demo) + README, 仿 GVHMR 风格, 让本机开箱即用跑通 demo_soma.py 并用 taiji.mp4 端到端验证。

**Architecture:** 三个 bash 脚本放在 `GEM-X/scripts/` 下, 路径由 `BASH_SOURCE` 自动推导, 关键路径用环境变量覆盖。install 建外部 venv (`envs/gemx`) + 装 torch/子模块/GEM/detectron2 + symlink SOMA assets; download_models 复用仓库 `gem/utils/hf_utils.py` 预拉权重; run_demo 跑 demo_soma.py -s。

**Tech Stack:** bash, uv (Python 包管理), Python 3.10, PyTorch 2.10.0+cu126, HuggingFace Hub, git submodule + git-lfs, detectron2。

**测试说明:** 这是脚本类任务, 无 pytest 框架。每个脚本的 "测试" = 真实执行该脚本并检查产出/退出码。验证命令与预期产出在每个 Task 中明确给出。

---

## 关键事实 (实现时直接用, 勿再查)

- 仓库根: `/root/paddlejob/workspace/env_run/penghaotian/sport_project/GEM-X`
- 测试视频: `/root/paddlejob/workspace/env_run/penghaotian/datas/Test/taiji.mp4` (已确认存在, 2.6M)
- venv 默认路径: `/root/paddlejob/workspace/env_run/penghaotian/envs/gemx` (可被 `GEMX_ENV_DIR` 覆盖)
- 本机已具备: `uv` 0.11.7 (`/root/.local/bin/uv`), `nvcc` 12.9, `git-lfs` 3.0.2, 驱动 550 (CUDA 12.4), 8×H800
- baidu 代理: `http://agent.baidu.com:8188` (实测可走通 pytorch/HF); PIP 源 `https://pypi.tuna.tsinghua.edu.cn/simple/`
- aliyun 代理: `http://njxg-banqian20230721-sousuo00230.njxg:3231/`; PIP 源 `https://mirrors.aliyun.com/pypi/simple/`
- 子模块 (`.gitmodules`): soma=`https://github.com/NVlabs/SOMA-X.git`, sam-3d-body=`https://github.com/facebookresearch/sam-3d-body.git`, soma-retargeter=SSH (**跳过**)
- SOMA Python 包名: `from soma import SOMALayer` (pip 包标识 `py-soma-x`)
- SomaLayer `data_root="inputs/soma_assets"` → symlink 到 `third_party/soma/assets`
- HF repo: `nvidia/GEM-X`; 权重函数见 `gem/utils/hf_utils.py`
- demo 入口: `python scripts/demo/demo_soma.py --video <v> -s --output_root outputs`
- demo 输出目录: `outputs/<video_stem>/`; 终判文件 `outputs/taiji/taiji_3_incam_global_horiz.mp4`
- GEM 依赖 (搬自 `scripts/install_env.sh`): `cloudpickle fvcore iopath pycocotools braceexpand roma 'setuptools<75'` + detectron2 `git+https://github.com/facebookresearch/detectron2.git@a1ce2f9 --no-build-isolation --no-deps`

## Task 1: 编写 install.sh

**Files:**
- Create: `scripts/install.sh`

- [ ] **Step 1: 写完整脚本**

```bash
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
```

- [ ] **Step 2: 赋可执行权限**

Run: `chmod +x scripts/install.sh`
Expected: 无输出, 退出码 0

- [ ] **Step 3: 语法检查**

Run: `bash -n scripts/install.sh && echo OK`
Expected: 输出 `OK` (无语法错误)

- [ ] **Step 4: 提交**

```bash
git add scripts/install.sh
git commit -m "add GEM-X install.sh (env + torch + submodules + deps)"
```

## Task 2: 编写 download_models.sh

**Files:**
- Create: `scripts/download_models.sh`

- [ ] **Step 1: 写完整脚本**

```bash
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
```

- [ ] **Step 2: 赋可执行权限**

Run: `chmod +x scripts/download_models.sh`
Expected: 无输出, 退出码 0

- [ ] **Step 3: 语法检查**

Run: `bash -n scripts/download_models.sh && echo OK`
Expected: 输出 `OK`

- [ ] **Step 4: 提交**

```bash
git add scripts/download_models.sh
git commit -m "add GEM-X download_models.sh (prefetch HF weights via hf_utils)"
```

## Task 3: 编写 run_demo.sh

**Files:**
- Create: `scripts/run_demo.sh`

- [ ] **Step 1: 写完整脚本**

```bash
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
VIDEO_STEM="$(basename "$VIDEO")"; VIDEO_STEM="${VIDEO_STEM%.*}"
echo "[run] GPU=$GPU_ID  video=$VIDEO  static_cam=on"
echo "[run] 输出将落在 outputs/$VIDEO_STEM/"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/demo/demo_soma.py \
    --video "$VIDEO" -s --output_root outputs

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
```

- [ ] **Step 2: 赋可执行权限**

Run: `chmod +x scripts/run_demo.sh`
Expected: 无输出, 退出码 0

- [ ] **Step 3: 语法检查**

Run: `bash -n scripts/run_demo.sh && echo OK`
Expected: 输出 `OK`

- [ ] **Step 4: 提交**

```bash
git add scripts/run_demo.sh
git commit -m "add GEM-X run_demo.sh (run demo_soma.py on taiji.mp4)"
```

## Task 4: 编写 scripts README

**Files:**
- Create: `scripts/README_install.md`

(注: 仓库 `scripts/` 下无现成 README, 命名 `README_install.md` 避免与潜在通用 README 混淆, 专述这三个脚本。)

- [ ] **Step 1: 写文档**

````markdown
# GEM-X 安装与 Demo 脚本 (scripts/)

本目录的三个脚本提供 GEM-X **全 3D pipeline demo** 的开箱即用流程:
安装环境、下载权重、跑 demo。脚本**自包含** (仅依赖本仓库自身, 跳过 soma-retargeter
SSH 子模块与 `--retarget`), 路径从脚本位置自动推导。

## 本机环境快照 (已验证)

| 项 | 值 |
| --- | --- |
| OS | Linux 5.15 (Ubuntu) |
| GPU | NVIDIA H800 80GB ×8 (demo 默认单卡) |
| Python | 3.10 (venv) |
| PyTorch | 2.10.0+cu126 (驱动 550/CUDA 12.4, 靠 CUDA 次版本兼容跑 cu126) |
| uv | 0.11.7 |
| 虚拟环境 | `/root/paddlejob/workspace/env_run/penghaotian/envs/gemx` |

## 前置条件

- **Linux + NVIDIA GPU**。
- **uv**: 建 venv 与装包。先装: https://docs.astral.sh/uv/ (脚本检查, 缺则报错)。
- **git-lfs**: 拉 SOMA body model assets。`apt install git-lfs` 后 `git lfs install`。
- **内网代理**: 脚本内置百度(默认)/阿里两档, 用于走通 pytorch/HuggingFace。

## 三步流程

```bash
# (1) 安装: 建 venv + torch + 子模块(soma/sam-3d-body) + GEM/依赖 + detectron2 + SOMA assets
#     proxy 可选 baidu(默认, PIP 快) | aliyun(HF 快)
bash scripts/install.sh baidu

# (2) 下权重: 从 HF nvidia/GEM-X 预拉 ckpt/vitpose/sam3d/mhr/soma_data 到 inputs/
bash scripts/download_models.sh baidu

# (3) 跑 demo: 默认用 taiji.mp4, 单卡 GPU 0, 静态相机
bash scripts/run_demo.sh
#   自定义: bash scripts/run_demo.sh /path/to/video.mp4 1
```

成功后结果视频在 `outputs/<video_stem>/<video_stem>_3_incam_global_horiz.mp4`。

## 换机器/换人要改的地方

| 文件 | 项 | 当前默认值 | 覆盖方式 |
| --- | --- | --- | --- |
| 三个脚本 | `ENV_DIR` | `/root/paddlejob/.../envs/gemx` | 设 `GEMX_ENV_DIR=/your/env` |
| `install.sh` / `download_models.sh` | 代理 URL | 百度/阿里内网代理 | 改脚本内 `http(s)_proxy` 段 |
| `run_demo.sh` | 默认视频 | `taiji.mp4` | 命令行第 1 个参数 |
| `run_demo.sh` | GPU | `0` | 命令行第 2 个参数 |

## 目录内容

| 文件 | 作用 |
| --- | --- |
| `install.sh` | 建 venv + torch + 子模块 + GEM/依赖 + detectron2 + SOMA assets symlink |
| `download_models.sh` | 复用 `gem/utils/hf_utils.py` 预拉 HF 权重到 `inputs/` |
| `run_demo.sh` | 跑 `scripts/demo/demo_soma.py -s` 出网格叠加视频 |

## 故障排查

| 现象 | 处理 |
| --- | --- |
| `ModuleNotFoundError: gem` | venv 未装好, 重跑 `install.sh` |
| `from soma import` 失败 | `third_party/soma` 未 `git lfs pull` 或未 `uv pip install -e` |
| OpenGL/EGL 报错 | 确认 `run_demo.sh` 已设 `PYOPENGL_PLATFORM=egl`; 机器需有 EGL 库 |
| `torch.cuda.is_available()` 为 False | cu126 与驱动不兼容, 回退 cu124 wheel (改 `install.sh` 步骤 2) |
| HF 下载慢/失败 | 换 `aliyun` 代理重跑 `download_models.sh aliyun` (断点续传) |
````

- [ ] **Step 2: 提交**

```bash
git add scripts/README_install.md
git commit -m "add GEM-X scripts README (install/download/demo workflow)"
```

## Task 5: 端到端执行 install.sh

**Files:** 无 (运行已写好的脚本)

- [ ] **Step 1: 运行 install.sh (后台, 耗时长)**

Run: `bash scripts/install.sh baidu`
Expected: 7 步全部打印, 最后出现 "依赖安装完成" 横幅, 退出码 0。
失败处理: 若 cu126 torch 安装/导入失败, 暂停回报用户, 提议回退 cu124。

- [ ] **Step 2: 验证 venv 与 torch CUDA**

Run:
```bash
/root/paddlejob/workspace/env_run/penghaotian/envs/gemx/bin/python -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
Expected: 打印 `torch 2.10.0+cu126 cuda True`

- [ ] **Step 3: 验证 soma 包与 assets symlink**

Run:
```bash
/root/paddlejob/workspace/env_run/penghaotian/envs/gemx/bin/python -c "from soma import SOMALayer; print('soma OK')"
ls -l inputs/soma_assets
```
Expected: 打印 `soma OK`; symlink 指向 `third_party/soma/assets` 且目录非空

- [ ] **Step 4: 验证 gem 与 detectron2**

Run:
```bash
/root/paddlejob/workspace/env_run/penghaotian/envs/gemx/bin/python -c "import gem, detectron2; print('gem+d2 OK')"
```
Expected: 打印 `gem+d2 OK`

## Task 6: 端到端执行 download_models.sh + run_demo.sh

**Files:** 无 (运行已写好的脚本)

- [ ] **Step 1: 下载权重**

Run: `bash scripts/download_models.sh baidu`
Expected: 5 项逐一打印路径, 最后 "权重下载完成"。`inputs/` 下出现
`pretrained/gem_soma.ckpt`、`checkpoints/vitpose/vitpose.pth`、
`checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt`、`mhr_data/mhr_model.pt`、
`soma_data/scale_mean.pth`、`soma_data/scale_comps.pth`。

- [ ] **Step 2: 跑 demo (后台, 耗时长)**

Run: `bash scripts/run_demo.sh`
Expected: demo 跑完 (检测→2D关键点→SAM3D特征→GEM推理→渲染), 最后打印
"[OK] demo 完成, 结果视频: .../outputs/taiji/taiji_3_incam_global_horiz.mp4"。

- [ ] **Step 3: 确认产出**

Run: `ls -lh outputs/taiji/`
Expected: 存在非空的 `taiji_3_incam_global_horiz.mp4`、`taiji_1_incam.mp4`、
`taiji_2_global.mp4`、`0_kp2d77_overlay.mp4`。

- [ ] **Step 4: 最终提交 (若无代码改动则跳过)**

仅当 Task 5/6 过程中对脚本做了修正才提交:
```bash
git add scripts/
git commit -m "fix GEM-X scripts after end-to-end verification"
```



