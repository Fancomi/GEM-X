# sam3 bbox → GEM-X 人体预测 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 写一个 `scripts/run_gemx_sam3.py`, 把 sam3 离线检测 bbox 注入 GEM-X 的 demo_soma pipeline (跳过 YOLOX), 保留原始 50fps、只跑检测覆盖帧, 输出 incam+global 可视化叠加视频。

**Architecture:** 利用 GEM-X `run_preprocess` 的「bbx.pt 已存在则跳过检测」缓存机制注入 sam3 bbox; 自写薄驱动复用 demo_soma 的 `_build_cfg/run_preprocess/load_data_dict/render_incam/render_global_o3d/render_2d_keypoints`, 绕过 `_copy_video_if_needed` 的 30fps 重编码, 用原始 fps 裁出检测覆盖帧段的视频做输入。

**Tech Stack:** Python 3.10, PyTorch, hydra, pandas, opencv, GEM-X (editable 安装), open3d 离屏渲染 (EGL)。

**测试说明:** 脚本类任务, 无 pytest 框架。纯逻辑函数 (`boxes_from_sam3_json`) 用真实 detect json 做断言式自测; 整体 pipeline 的"测试" = 真实单视频执行并检查产出/退出码 (沿用本仓库 install 脚本计划的同款做法)。

---

## 关键事实 (实现时直接用, 勿再查)

- 仓库根: `/root/paddlejob/workspace/env_run/penghaotian/sport_project/GEM-X`
- venv: `/root/paddlejob/workspace/env_run/penghaotian/envs/gemx/bin/python` (taiji demo 已端到端跑通, 所有权重已缓存)
- 数据根: `/root/paddlejob/workspace/env_run/penghaotian/datas/cos_videos`
  - 检测 json: `detect/<vid>.json`, 字段 `frame_idx, bbox([x1,y1,x2,y2]), mask_rle, cy, cx, area`
  - 视频: `videos/<vid>.mp4`
- 目标视频: `10m_2024_12_04_20241204_092631` → 240 帧 / 50fps / 1080×1920; sam3 检到 131 帧 (frame_idx 0~130, 连续无空洞)
- demo: `scripts/demo/demo_soma.py` (可 import; 模块顶层只 setdefault EGL 环境变量 + patch torch.load, 不初始化 CUDA)
- demo 复用函数: `_build_cfg(args)`、`run_preprocess(cfg)`、`load_data_dict(cfg)`、`render_incam(cfg,fps)`、`render_global_o3d(cfg,fps)`、`render_2d_keypoints(...)`、`resolve_ckpt_path(cfg)`
- `_build_cfg` 需要的 args 属性: `video, output_root, static_cam, verbose, render_mhr, ckpt, exp, sam3d_ckpt_path, sam3d_mhr_path`
- cfg.paths: `bbx, vitpose, vit_features, slam, hpe_results, incam_video, global_video, incam_global_horiz_video`; output_dir=`<output_root>/<video_stem>`, preprocess_dir=`<output_dir>/preprocess`
- bbx.pt 格式: `{"bbx_xyxy": (L,4) float, "bbx_xys": (L,3) float}`
- 工具: `gem/utils/geo_transform.py:get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2)`、`gem/utils/kp2d_utils.py:smooth_bbx_xyxy(bbx_xyxy, window=5)`、`gem/utils/video_io_utils.py:{get_video_lwh, read_video_np, save_video, merge_videos_horizontal}`
- `read_video_np(path, start_frame, end_frame)`: 返回帧 `[start, end)`, 断言 `len==end-start`
- demo `_run_human_detection` 的 bbx 处理顺序 (照抄): smooth_bbx_xyxy → clamp 到 [0,W-1]/[0,H-1] → get_bbx_xys_from_xyxy(base_enlarge=1.2)
- 输出根: `outputs/sam3_gemx/<vid>/`; 终判文件 `outputs/sam3_gemx/<vid>/<vid>_3_incam_global_horiz.mp4`

## Task 1: 创建脚本骨架 + `boxes_from_sam3_json` 解析

**Files:**
- Create: `scripts/run_gemx_sam3.py`

> 顶层**不**导入 torch / demo_soma (需在设置 `CUDA_VISIBLE_DEVICES` 后才导入)。
> 顶层只用标准库 + numpy + pandas。`boxes_from_sam3_json` 返回 numpy, 不碰 torch。

- [ ] **Step 1: 写骨架 + 解析函数**

```python
#!/usr/bin/env python3
"""
用 sam3 离线检测 bbox 驱动 GEM-X demo_soma 全 3D pipeline (跳过 YOLOX)。
对照 MoGe/sam3dbody/run_duomo.py 的"喂 boxes、跳过自带检测"套路。

用法:
  python scripts/run_gemx_sam3.py --videos 10m_2024_12_04_20241204_092631 --gpu 0

注入点: GEM-X run_preprocess 在 bbx.pt 已存在时跳过检测。本脚本预先把 sam3
bbox 写成 bbx.pt, 并绕过 demo 的 30fps 重编码, 用原始 fps 裁出检测覆盖帧段。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = '/root/paddlejob/workspace/env_run/penghaotian/datas/cos_videos'
DETECT_DIR = f'{DATA_ROOT}/detect'
VIDEO_DIR = f'{DATA_ROOT}/videos'


def boxes_from_sam3_json(json_path):
    """读 sam3 detect json -> (boxes_xyxy, lo, hi, src_w, src_h)。

    boxes_xyxy: (N,4) float32 numpy, N=hi-lo+1, 行索引对齐 frame_idx-lo。
    只覆盖检测帧段 [lo,hi]; 段内空洞线性插值 (NOT 向段外尾帧延伸)。
    """
    with open(json_path) as f:
        det = json.load(f)
    frames = det['frames']
    if not frames:
        raise ValueError('no_frames')
    vi = det['video_info']
    src_w, src_h = int(vi['width']), int(vi['height'])
    rows = [(fd['frame_idx'], *fd['bbox']) for fd in frames]
    df = pd.DataFrame(rows, columns=['frame', 'x1', 'y1', 'x2', 'y2']).set_index('frame')
    lo, hi = int(df.index.min()), int(df.index.max())
    df = df.reindex(range(lo, hi + 1))
    df = df.interpolate(method='linear').bfill().ffill()
    boxes = df.values.astype(np.float32)  # (N,4)
    return boxes, lo, hi, src_w, src_h
```

- [ ] **Step 2: 运行解析自测 (真实 json)**

Run:
```bash
cd /root/paddlejob/workspace/env_run/penghaotian/sport_project/GEM-X
/root/paddlejob/workspace/env_run/penghaotian/envs/gemx/bin/python - <<'PY'
import sys; sys.path.insert(0, 'scripts')
from run_gemx_sam3 import boxes_from_sam3_json
b, lo, hi, w, h = boxes_from_sam3_json(
    '/root/paddlejob/workspace/env_run/penghaotian/datas/cos_videos/detect/10m_2024_12_04_20241204_092631.json')
assert lo == 0 and hi == 130, (lo, hi)
assert b.shape == (131, 4), b.shape
assert (w, h) == (1080, 1920), (w, h)
assert not (b != b).any(), 'has NaN'
print('OK boxes', b.shape, 'frame_seg', lo, hi, 'src', w, h)
PY
```
Expected: `OK boxes (131, 4) frame_seg 0 130 src 1080 1920`

- [ ] **Step 3: 提交**

```bash
git add scripts/run_gemx_sam3.py
git commit -m "add run_gemx_sam3 skeleton + sam3 bbox parser"
```

## Task 2: `write_bbx_pt` + `trim_video` + 对齐自测

**Files:**
- Modify: `scripts/run_gemx_sam3.py`

- [ ] **Step 1: 追加两个函数 (在 `boxes_from_sam3_json` 之后)**

```python
def write_bbx_pt(boxes_xyxy_np, W, H, out_path):
    """照抄 demo _run_human_detection: smooth -> clamp -> xys, 存 bbx.pt。"""
    import torch
    from gem.utils.kp2d_utils import smooth_bbx_xyxy
    from gem.utils.geo_transform import get_bbx_xys_from_xyxy

    bbx_xyxy = torch.from_numpy(np.asarray(boxes_xyxy_np)).float()
    bbx_xyxy = smooth_bbx_xyxy(bbx_xyxy, window=5)
    bbx_xyxy[:, [0, 2]] = bbx_xyxy[:, [0, 2]].clamp(0, W - 1)
    bbx_xyxy[:, [1, 3]] = bbx_xyxy[:, [1, 3]].clamp(0, H - 1)
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'bbx_xyxy': bbx_xyxy, 'bbx_xys': bbx_xys}, out_path)
    return bbx_xyxy.shape[0]


def trim_video(src, lo, hi, dst):
    """裁出帧段 [lo,hi] (含两端), 保留源 fps 写到 dst。返回 (N, fps)。"""
    import cv2
    from gem.utils.video_io_utils import read_video_np, save_video

    fps = cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FPS) or 30.0
    frames = read_video_np(str(src), start_frame=lo, end_frame=hi + 1)  # [lo, hi+1)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, str(dst), fps=int(round(fps)), crf=23)
    return len(frames), float(fps)
```

- [ ] **Step 2: 运行对齐自测 (裁视频帧数 == bbx 行数)**

Run:
```bash
cd /root/paddlejob/workspace/env_run/penghaotian/sport_project/GEM-X
/root/paddlejob/workspace/env_run/penghaotian/envs/gemx/bin/python - <<'PY'
import sys; sys.path.insert(0, 'scripts')
from run_gemx_sam3 import boxes_from_sam3_json, write_bbx_pt, trim_video
import torch, tempfile, os
vid = '10m_2024_12_04_20241204_092631'
dj = f'/root/paddlejob/workspace/env_run/penghaotian/datas/cos_videos/detect/{vid}.json'
mp4 = f'/root/paddlejob/workspace/env_run/penghaotian/datas/cos_videos/videos/{vid}.mp4'
b, lo, hi, w, h = boxes_from_sam3_json(dj)
td = tempfile.mkdtemp()
bbx_path = os.path.join(td, 'bbx.pt')
n_bbx = write_bbx_pt(b, w, h, bbx_path)
d = torch.load(bbx_path)
assert d['bbx_xyxy'].shape == (131, 4), d['bbx_xyxy'].shape
assert d['bbx_xys'].shape == (131, 3), d['bbx_xys'].shape
n_trim, fps = trim_video(mp4, lo, hi, os.path.join(td, 'in.mp4'))
assert n_trim == n_bbx == 131, (n_trim, n_bbx)
assert abs(fps - 50.0) < 1e-3, fps
print('OK bbx', d['bbx_xyxy'].shape, d['bbx_xys'].shape, '| trim', n_trim, 'fps', fps)
PY
```
Expected: `OK bbx torch.Size([131, 4]) torch.Size([131, 3]) | trim 131 fps 50.0`

- [ ] **Step 3: 提交**

```bash
git add scripts/run_gemx_sam3.py
git commit -m "add write_bbx_pt + trim_video (50fps preserved, frame-aligned)"
```

## Task 3: 主驱动 `process_video` + `main` (注入 + 复用 demo pipeline)

**Files:**
- Modify: `scripts/run_gemx_sam3.py`

> 关键: `process_video` 内才 import torch/hydra/demo_soma (要在 main 设好
> `CUDA_VISIBLE_DEVICES`/EGL 之后)。`_build_cfg` 用 trimmed 视频的 stem 推出
> `output_dir=<out_root>/<vid>`; 把 bbx.pt 写在 `run_preprocess` 之前 → YOLOX 跳过;
> 全程跳过 demo 的 `_copy_video_if_needed` (30fps 重编码)。

- [ ] **Step 1: 追加 `process_video` (在 `trim_video` 之后)**

```python
def process_video(vid, out_root):
    import torch
    import hydra
    from types import SimpleNamespace
    from gem.utils.net_utils import detach_to_cpu
    from gem.utils.video_io_utils import merge_videos_horizontal
    import demo_soma as D

    detect_json = Path(DETECT_DIR) / f'{vid}.json'
    src_video = Path(VIDEO_DIR) / f'{vid}.mp4'
    if not detect_json.exists():
        return False, 'no_detect_json', None
    if not src_video.exists():
        return False, 'no_video', None

    boxes, lo, hi, W, H = boxes_from_sam3_json(str(detect_json))
    out_dir = Path(out_root) / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    trimmed = out_dir / f'{vid}.mp4'
    n_trim, fps = trim_video(src_video, lo, hi, trimmed)
    if n_trim != boxes.shape[0]:
        return False, f'frame_mismatch(trim={n_trim},bbx={boxes.shape[0]})', None
    fps_i = int(round(fps))

    args = SimpleNamespace(
        video=str(trimmed), output_root=str(out_root), static_cam=True,
        verbose=False, render_mhr=False, ckpt=None,
        exp='gem_soma_regression', sam3d_ckpt_path=None, sam3d_mhr_path=None)
    cfg = D._build_cfg(args)  # cfg.video_path == trimmed (源 fps); 不调 _copy_video_if_needed

    write_bbx_pt(boxes, W, H, cfg.paths.bbx)  # 注入 -> run_preprocess 跳过 YOLOX
    D.run_preprocess(cfg)
    D.render_2d_keypoints(
        video_path=cfg.video_path, vitpose_path=cfg.paths.vitpose,
        bbx_path=cfg.paths.bbx,
        output_path=str(Path(cfg.output_dir) / '0_kp2d77_overlay.mp4'), fps=fps_i)

    data = D.load_data_dict(cfg)
    if not Path(cfg.paths.hpe_results).exists():
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(D.resolve_ckpt_path(cfg))
        model = model.eval().cuda()
        pred = model.predict(data, static_cam=cfg.static_cam, postproc=True)
        torch.save(detach_to_cpu(pred), cfg.paths.hpe_results)

    D.render_incam(cfg, fps=fps_i)
    D.render_global_o3d(cfg, fps=fps_i)
    merge_videos_horizontal(
        [cfg.paths.incam_video, cfg.paths.global_video],
        cfg.paths.incam_global_horiz_video)
    return True, 'ok', {'frames': n_trim, 'fps': fps,
                        'out': cfg.paths.incam_global_horiz_video}
```

- [ ] **Step 2: 追加 `main` + `__main__` (文件末尾)**

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', nargs='+', required=True,
                        help='video stems (无扩展名), 与 run_sam3d_body 用法一致')
    parser.add_argument('--gpu', type=int, default=0, help='物理 GPU id')
    parser.add_argument('--out_root', default=str(REPO_ROOT / 'outputs' / 'sam3_gemx'),
                        help='输出根 (默认 GEM-X/outputs/sam3_gemx)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('EGL_PLATFORM', 'surfaceless')
    os.chdir(REPO_ROOT)  # SomaLayer 用相对路径 inputs/soma_assets
    sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'demo'))

    passed, failed = 0, 0
    for i, vid in enumerate(args.videos):
        t0 = time.time()
        try:
            ok, reason, info = process_video(vid, args.out_root)
        except Exception as e:
            import traceback; traceback.print_exc()
            ok, reason, info = False, f'EXC:{e}', None
        if ok:
            passed += 1
            status = f"PASS ({info['frames']}f @ {info['fps']:.0f}fps, {time.time()-t0:.0f}s)\n        -> {info['out']}"
        else:
            failed += 1
            status = f'FAIL:{reason}'
        print(f'[{i+1}/{len(args.videos)}] {vid} -> {status}', flush=True)
    print(f'Done! pass={passed} fail={failed}', flush=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: 语法检查 + 提交**

Run:
```bash
cd /root/paddlejob/workspace/env_run/penghaotian/sport_project/GEM-X
/root/paddlejob/workspace/env_run/penghaotian/envs/gemx/bin/python -c "import ast; ast.parse(open('scripts/run_gemx_sam3.py').read()); print('OK')"
```
Expected: `OK`

```bash
git add scripts/run_gemx_sam3.py
git commit -m "add process_video + main (inject bbx.pt, reuse demo pipeline)"
```

## Task 4: 端到端单视频验证

**Files:** 无 (运行已写好的脚本)

- [ ] **Step 1: 跑单视频 (耗时长, 建议后台)**

Run:
```bash
cd /root/paddlejob/workspace/env_run/penghaotian/sport_project/GEM-X
/root/paddlejob/workspace/env_run/penghaotian/envs/gemx/bin/python \
  scripts/run_gemx_sam3.py --videos 10m_2024_12_04_20241204_092631 --gpu 0
```
Expected: 末尾打印 `[1/1] 10m_2024_12_04_20241204_092631 -> PASS (131f @ 50fps, ...s)` 与 `Done! pass=1 fail=0`。
失败处理: 若 SomaLayer 找不到 `inputs/soma_assets` → 确认 `os.chdir(REPO_ROOT)` 生效;
若 open3d EGL 报错 → 确认 EGL 环境变量已设。先诊断根因, 勿盲改。

- [ ] **Step 2: 确认产出**

Run:
```bash
ls -lh /root/paddlejob/workspace/env_run/penghaotian/sport_project/GEM-X/outputs/sam3_gemx/10m_2024_12_04_20241204_092631/
```
Expected: 存在非空的 `10m_2024_12_04_20241204_092631_3_incam_global_horiz.mp4`、
`*_1_incam.mp4`、`*_2_global.mp4`、`0_kp2d77_overlay.mp4`。

- [ ] **Step 3: 校验帧数对齐 (输出 == 检测覆盖帧)**

Run:
```bash
/root/paddlejob/workspace/env_run/penghaotian/envs/gemx/bin/python - <<'PY'
import cv2
p='/root/paddlejob/workspace/env_run/penghaotian/sport_project/GEM-X/outputs/sam3_gemx/10m_2024_12_04_20241204_092631/10m_2024_12_04_20241204_092631_1_incam.mp4'
c=cv2.VideoCapture(p)
n=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); fps=c.get(cv2.CAP_PROP_FPS)
print('incam frames', n, 'fps', round(fps))
assert n == 131, n
print('OK frame-aligned (131)')
PY
```
Expected: `incam frames 131 fps 50` 与 `OK frame-aligned (131)`

- [ ] **Step 4: 最终提交 (仅当 Task 4 过程中改了脚本)**

```bash
git add scripts/run_gemx_sam3.py
git commit -m "fix run_gemx_sam3 after end-to-end verification"
```
