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
