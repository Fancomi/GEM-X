# GEM-X 一键安装脚本设计 (仿 GVHMR)

## 目标

参考 `GVHMR/scripts/install.sh` 的风格，为 GEM-X 编写一套自包含的安装/下载/验证脚本，
让用户在本机 (Linux + NVIDIA H800, 驱动 550/CUDA 12.4) 上开箱即用地跑通
`demo_soma.py` 全 3D pipeline。最终用 `/root/paddlejob/workspace/env_run/penghaotian/datas/Test/taiji.mp4`
做端到端验证, 产出网格叠加视频。

## 已确认决策

| 项 | 决策 |
| --- | --- |
| 虚拟环境位置 | 外部 `GEMX_ENV_DIR` (默认 `/root/paddlejob/workspace/env_run/penghaotian/envs/gemx`), 仿 GVHMR |
| PyTorch 版本 | pin `torch==2.10.0+cu126` / `torchvision==0.25.0+cu126` (按 `requirements.txt`) |
| retarget | 跳过 (soma-retargeter 走 SSH 子模块, 本机拉不到, 且仅 `--retarget` 用到) |
| 验证 | 装完自动跑 taiji.mp4 (触发权重下载) |
| 脚本数量 | 三脚本 (install / download_models / run_demo), 仿 GVHMR |
| 权重下载 | 独立 `download_models.sh` 预拉, 复用仓库自带 `gem/utils/hf_utils.py` 的 hf_hub_download |
| 代理 | 内置百度(默认)/阿里两档, 仿 GVHMR |

## 架构总览

在 GEM-X 仓库 `scripts/` 下新增三个脚本 + 一份 README, 路径从脚本位置 (`BASH_SOURCE`)
自动推导, 关键路径可用环境变量覆盖。

```
scripts/
├── install.sh           # 建 env + torch + 子模块 + GEM/依赖 + SOMA assets symlink
├── download_models.sh   # 用 hf_hub_download 预拉 nvidia/GEM-X 全部权重到 inputs/
├── run_demo.sh          # 跑 demo_soma.py -s 验证 taiji.mp4
└── README.md            # 三步流程说明 + 换机器须改项 + 故障排查
```

不改动 GVHMR 仓库, 不改动 GEM-X 现有代码 (仅新增脚本与文档)。

## 脚本 1: install.sh

用法: `bash scripts/install.sh [baidu|aliyun]`

环境变量:
- `GEMX_ENV_DIR` 虚拟环境路径 (默认 `/root/paddlejob/workspace/env_run/penghaotian/envs/gemx`)
- `CUDA_HOME` (默认 `/usr/local/cuda`)

步骤:
1. 代理设置 (baidu: PIP 用清华源 + 百度内网 proxy; aliyun: HF 用阿里源 + 阿里内网 proxy) + CUDA 环境 (PATH/LD_LIBRARY_PATH); 检查 `uv` 可用
2. `uv venv $GEMX_ENV_DIR --python 3.10` (setup.cfg 要求 ≥3.10, 与本机 GVHMR/Dockerfile 一致)
3. PyTorch: `uv pip install torch==2.10.0+cu126 torchvision==0.25.0+cu126 --index-url https://download.pytorch.org/whl/cu126`
4. 子模块: `git submodule update --init third_party/soma third_party/sam-3d-body` (**跳过** soma-retargeter SSH 子模块)
5. SOMA: `uv pip install -e third_party/soma` → `cd third_party/soma && git lfs pull` (拉 body model assets)
6. GEM 本体 + SAM-3D-Body 运行时依赖 (搬自 `scripts/install_env.sh`):
   - `uv pip install -e .`
   - `uv pip install cloudpickle fvcore iopath pycocotools braceexpand roma 'setuptools<75'`
   - detectron2: `uv pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps`
7. `ln -sf third_party/soma/assets inputs/soma_assets` (SomaLayer 的 `data_root` 指向此处)

注意: 第 6 步 detectron2 用 `--no-build-isolation`, 依赖 torch 已装好 (步骤 3 已保证顺序)。

## 脚本 2: download_models.sh

用法: `bash scripts/download_models.sh [baidu|aliyun]`

设置代理后, 在 venv 内 `python -c` 调用仓库自带 `gem/utils/hf_utils.py` 的各 `download_*`
函数, 预拉 `nvidia/GEM-X` 全部权重到 `inputs/` 对应目录 (与代码默认路径完全一致, 让 demo 跳过下载):

| 文件 | 落地目录 | hf_utils 函数 |
| --- | --- | --- |
| `gem_soma.ckpt` | `inputs/pretrained/` | `download_checkpoint()` |
| `vitpose.pth` | `inputs/checkpoints/vitpose/` | `download_vitpose_checkpoint()` |
| `sam3d_body.ckpt` + `model_config.yaml` | `inputs/checkpoints/sam-3d-body-dinov3/` | `download_sam3d_checkpoint()` |
| `mhr_model.pt` | `inputs/mhr_data/` | `download_mhr_model()` |
| `scale_mean.pth` + `scale_comps.pth` | `inputs/soma_data/` | `download_soma_data()` |

`hf_hub_download` 自带本地缓存检测 (文件已存在则跳过), 天然支持断点重试。
YOLOX ONNX 由 `gem/utils/yolox_detector.py` 自动下到 `~/.cache/rtmlib` (openmmlab 源, 非 HF),
脚本设好代理让它能下, 不单独管理。

## 脚本 3: run_demo.sh

用法: `bash scripts/run_demo.sh [video_path]`

默认 `video_path=/root/paddlejob/workspace/env_run/penghaotian/datas/Test/taiji.mp4`

- 设 headless 渲染环境 `PYOPENGL_PLATFORM=egl`, `EGL_PLATFORM=surfaceless`
- `CUDA_VISIBLE_DEVICES=0` (H800, 单卡)
- 在 venv 内跑 `python scripts/demo/demo_soma.py --video <video> -s --output_root outputs`
  - 用 `-s` (static_cam): VO/SLAM 未装, demo 本就 fallback 静态相机, 显式加 `-s` 避免无谓告警
- 输出落 `outputs/taiji/`, 检查 `taiji_3_incam_global_horiz.mp4` 是否生成

## 验证与风险

- **CUDA 兼容**: 驱动 550 (CUDA 12.4), cu126 wheel 靠 CUDA Minor Version Compatibility 在 R550 上可跑
  (本机 sam3d env 的 cu124 已验证 GPU 可用)。若 cu126 实测加载失败 (`torch.cuda.is_available()` 为假
  或报 CUDA 版本错), 回退方案: 改用 cu124 wheel (`torch==2.6.0+cu124` 等)。
- **首次下载耗时**: ckpt + sam3d + vitpose + mhr 合计数 GB, 时间较长; `hf_hub_download` 缓存可断点重试。
- **代理已实测**: baidu 代理可走通 `download.pytorch.org` 与 `huggingface.co` (返回 200)。

## 端到端验证流程

`install.sh` 跑通后, 依次执行:
1. `bash scripts/download_models.sh baidu` — 预拉全部权重
2. `bash scripts/run_demo.sh` — 跑 taiji.mp4

成功判据: `outputs/taiji/taiji_3_incam_global_horiz.mp4` 生成且非空。

