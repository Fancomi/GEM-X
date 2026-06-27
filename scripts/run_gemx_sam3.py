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
        raise ValueError(f'no frames in {json_path}')
    vi = det['video_info']
    src_w, src_h = int(vi['width']), int(vi['height'])
    rows = [(fd['frame_idx'], *fd['bbox']) for fd in frames]
    df = pd.DataFrame(rows, columns=['frame', 'x1', 'y1', 'x2', 'y2']).set_index('frame')
    assert len(df) == len(df.index.unique()), f'duplicate frame_idx in {json_path}'
    lo, hi = int(df.index.min()), int(df.index.max())
    df = df.reindex(range(lo, hi + 1))
    df = df.interpolate(method='linear').bfill().ffill()
    assert not df.isnull().values.any(), f'NaN after interpolation in {json_path}'
    boxes = df.values.astype(np.float32)  # (N,4)
    return boxes, lo, hi, src_w, src_h


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

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    frames = read_video_np(str(src), start_frame=lo, end_frame=hi + 1)  # [lo, hi+1)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, str(dst), fps=int(round(fps)), crf=23)
    return len(frames), float(fps)


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
