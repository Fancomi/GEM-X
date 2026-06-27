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
