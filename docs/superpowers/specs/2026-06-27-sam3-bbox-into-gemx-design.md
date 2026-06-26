# 用 sam3 bbox 驱动 GEM-X 人体预测 — 设计

**日期:** 2026-06-27
**状态:** 已确认 (用户批准设计, 待写实现计划)

## 目标

把 sam3 离线检测的 bbox 喂进 GEM-X 的 `demo_soma` 全 3D pipeline, 跳过 GEM-X 自带的
YOLOX 检测, 复用 sam3 的检测结果跑人体预测, 输出 incam + global 可视化叠加视频。
仿 `MoGe/sam3dbody/run_duomo.py` / `run_sam3d_body.py` 的「喂 boxes, 跳过自带检测」
同款套路, 数据沿用 `datas/cos_videos`。

**本次范围 (用户确认):**
- 只要可视化视频 (incam + global), 不导出 3D 参数。
- 先单视频验证 (`10m_2024_12_04_20241204_092631`), 跑通肉眼确认后再谈批量。
- 保留原始帧率 (50fps), 不走 demo 默认的 30fps 重编码。

## 核心思路

GEM-X `scripts/demo/demo_soma.py` 的 `run_preprocess` 有缓存机制:

```python
if not Path(paths.bbx).exists():
    _run_human_detection(cfg, paths, L, W, H)   # YOLOX + ByteTrack
bbx_xys = torch.load(paths.bbx)["bbx_xys"]
```

**只要 `bbx.pt` 已存在, YOLOX 检测就被天然跳过。** 因此无需改 demo 源码——
在调用 pipeline 前, 把 sam3 的 bbox 预先转换并写成 `bbx.pt` 即可。这正是
run_duomo「喂 boxes、跳过自带检测」的注入思路, 只是注入点换成 GEM-X 的缓存文件。

## 架构

### 新增文件
- `scripts/run_gemx_sam3.py` — 薄封装驱动脚本。

不重写 pipeline、不修改官方 `demo_soma.py`。脚本 `import` 复用 demo 的函数:
`_build_cfg`、`run_preprocess`、`load_data_dict`、`render_incam`、
`render_global_o3d`、`render_2d_keypoints`、`resolve_ckpt_path`。

### 数据流

```
detect/<vid>.json  (sam3 检测, 131 帧, bbox=[x1,y1,x2,y2])
  │
  ├─(1) 裁视频到检测覆盖帧段 [first, last]，保留原始 50fps（不重编码成 30fps）
  │         → 临时输入视频 trimmed.mp4 (帧数 == bbox 行数)
  │
  ├─(2) bbox(xyxy) → torch tensor (L,4)
  │         → smooth_bbx_xyxy(window=5) + clamp 到图像边界
  │         → get_bbx_xys_from_xyxy(base_enlarge=1.2) 得 bbx_xys (L,3)
  │         → torch.save({"bbx_xyxy","bbx_xys"}, paths.bbx)
  │
  ├─(3) run_preprocess(cfg)
  │         bbx.pt 已存在 → 跳过 YOLOX
  │         → VitPose(pose_type="soma") 提 2D 关键点 → vitpose.pt
  │         → static_cam → 单位矩阵相机轨迹 camera.pt
  │         → SAM3DBExtractor 提特征 → vit_features.pt
  │
  ├─(4) render_2d_keypoints → 0_kp2d77_overlay.mp4
  │
  ├─(5) load_data_dict(cfg) → model.predict → hpe_results.pt
  │
  └─(6) render_incam + render_global_o3d
            → merge_videos_horizontal
            → outputs/sam3_gemx/<vid>/<vid>_3_incam_global_horiz.mp4
```

## 两个关键对齐点

### 1. 帧率 — 绕过 30fps 重编码
demo 的 `main()` 会先调 `_copy_video_if_needed(cfg)`, 内部用
`get_writer(dst, fps=30, ...)` **强制把输入重编码为 30fps**。本批跳水视频为
50fps/240 帧, 重编码后帧数变化, 会让按原始帧索引的 sam3 bbox 错位。

**做法:** 不调用 `main()`, 自写驱动直接把 `cfg.video_path` 指向「原始 50fps 裁出的视频」,
跳过 `_copy_video_if_needed`。GEM-X 全程按 `get_video_lwh(video_path)` 拿帧数对齐,
只要视频帧数 == bbx 行数就一致, 输出视频沿用源 fps (50)。

### 2. 检测覆盖范围 — 只跑有人帧
sam3 对目标视频只检到前 **131 帧** (frame_idx 0~130, 连续无空洞),
后 109 帧人已入水/出画无检测。

**做法:** 只跑检测覆盖的帧段——裁出 `[first_idx, last_idx]` 区间画面 + 对应行 bbox,
严格一一对应。不对无检测的尾段做填充延伸。

## 帧对齐与缺帧策略

- 读 detect json, 取所有 `frame_idx` 的 min/max 作为覆盖帧段 `[lo, hi]`。
- 裁出视频帧 `[lo, hi]` (含两端), 长度 `N = hi-lo+1`。
- 构造长度 N 的 bbox 序列, 索引对齐 `frame_idx - lo`。
- **覆盖段内若有空洞** (本视频无, 通用化考虑): 线性插值补齐 (对齐 run_duomo 策略:
  `interpolate(linear) + bfill + ffill`), 但**不向覆盖段外的尾段延伸**。

## 输出

| 产物 | 路径 |
| --- | --- |
| 横向拼接结果 | `outputs/sam3_gemx/<vid>/<vid>_3_incam_global_horiz.mp4` |
| incam 叠加 | `outputs/sam3_gemx/<vid>/<vid>_1_incam.mp4` |
| global 视角 | `outputs/sam3_gemx/<vid>/<vid>_2_global.mp4` |
| 2D 关键点叠加 | `outputs/sam3_gemx/<vid>/0_kp2d77_overlay.mp4` |
| 中间缓存 | `outputs/sam3_gemx/<vid>/preprocess/{bbx,vitpose,camera,vit_features}.pt` |

输出视频帧率 = 源视频帧率 (50fps)。

## 运行接口 (对齐 run_sam3d_body / run_duomo)

```bash
python scripts/run_gemx_sam3.py --videos 10m_2024_12_04_20241204_092631 --gpu 0
```

- `--videos`: 视频 stem (无扩展名), 可传多个 (本次先单个)。
- `--gpu`: 物理 GPU id (默认 0), 内部 `CUDA_VISIBLE_DEVICES` 重映射到 cuda:0。
- 静态相机 (等价 demo 的 `-s`), 单卡。
- 环境变量: `PYOPENGL_PLATFORM=egl`、`EGL_PLATFORM=surfaceless` (open3d 离屏渲染)。

## 边界处理

| 情况 | 处理 |
| --- | --- |
| detect json 不存在 | 跳过该视频, 打印 FAIL 原因 |
| 视频文件不存在 | 跳过该视频, 打印 FAIL 原因 |
| detect json `frames` 为空 | 跳过, 打印 FAIL 原因 |
| 视频帧数 < 覆盖段末帧 | 跳过, 打印帧数不匹配 |
| 覆盖段内 bbox 空洞 | 线性插值补齐 (不向段外延伸) |
| bbox 越界 | clamp 到 [0, W-1] / [0, H-1] |

## 关键事实 (实现时直接用)

- 仓库根: `/root/paddlejob/workspace/env_run/penghaotian/sport_project/GEM-X`
- 虚拟环境: `/root/paddlejob/workspace/env_run/penghaotian/envs/gemx` (已装好, taiji demo 已端到端跑通)
- 数据根: `/root/paddlejob/workspace/env_run/penghaotian/datas/cos_videos`
  - 检测: `detect/<vid>.json` (字段: frame_idx, bbox[xyxy], mask_rle, cy, cx, area)
  - 视频: `videos/<vid>.mp4`
- 目标视频 `10m_2024_12_04_20241204_092631`: 240 帧 / 50fps / 1080×1920; sam3 检到 131 帧 (0~130)
- demo 入口函数在 `scripts/demo/demo_soma.py`; 配置 `configs/demo_soma.yaml`
- bbx.pt 格式: `{"bbx_xyxy": (L,4), "bbx_xys": (L,3)}`
- 工具函数: `gem/utils/geo_transform.py:get_bbx_xys_from_xyxy`、
  `gem/utils/kp2d_utils.py:smooth_bbx_xyxy`、
  `gem/utils/video_io_utils.py:{get_video_lwh, read_video_np, save_video, get_writer, merge_videos_horizontal}`

## 测试说明

脚本类任务, 无 pytest 框架。"测试" = 真实执行脚本检查产出/退出码:
单视频跑通后, `outputs/sam3_gemx/<vid>/<vid>_3_incam_global_horiz.mp4` 存在且非空,
帧数 == 检测覆盖帧数, 肉眼确认网格叠加合理。
