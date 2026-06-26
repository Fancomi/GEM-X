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
