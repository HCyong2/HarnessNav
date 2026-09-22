#!/usr/bin/env python
"""三值栅格可视化：UNKNOWN 暗、FREE 灰、OCC 白。"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "test", "harness"))

from harness.protocol import dump_json
from nav.occupancy import CELL_FREE, CELL_OCC, CELL_UNKNOWN, EVEN_PANO_INDICES, OccupancyMap, save_rgb
from boot import boot_env

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="三值栅格可视化")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default=os.path.join(HERE, "output"))
    p.add_argument("--gpu-id", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    """扫一圈后画栅格 BEV。"""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env, config = boot_env(seed=args.seed, gpu_id=args.gpu_id)
    try:
        occ = OccupancyMap.from_config(config)
        occ.scan_around(env, rotate_times=12, look_down_floor=True)
        bev = occ.render_bev_from_env(env, sector_labels=EVEN_PANO_INDICES,
                                      paint_grid=True, color_mode="bw", point_size=1)
        save_rgb(os.path.join(args.out, "grid_bev.png"), bev)
        g = occ.grid
        counts = {
            "unknown": int(np.sum(g == CELL_UNKNOWN)) if g is not None else 0,
            "free": int(np.sum(g == CELL_FREE)) if g is not None else 0,
            "occ": int(np.sum(g == CELL_OCC)) if g is not None else 0,
            "inflate": int(np.sum(occ.inflate > 0)) if occ.inflate is not None else 0,
            "shape": None if g is None else list(g.shape),
        }
        dump_json(os.path.join(args.out, "meta.json"), counts)
        print(counts)
        print(args.out)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
