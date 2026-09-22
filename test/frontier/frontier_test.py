#!/usr/bin/env python
"""环视后把前沿画在 BEV 与第一视角上。"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "test", "harness"))

from harness.overlay import draw_annotated, face_best_frontier_view
from harness.protocol import dump_json
from nav.goto import body_yaw_env
from nav.occupancy import EVEN_PANO_INDICES, OccupancyMap, save_rgb
from nav.transform import habitat_camera_intrinsic
from boot import boot_env

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="前沿提取可视化")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default=os.path.join(HERE, "output"))
    p.add_argument("--gpu-id", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    """扫一圈（含低头扫地），对准有前沿的扇区后落盘。"""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env, config = boot_env(seed=args.seed, gpu_id=args.gpu_id)
    try:
        occ = OccupancyMap.from_config(config)
        yaw0 = body_yaw_env(env)
        occ.scan_around(env, rotate_times=12, look_down_floor=True)
        xyz = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
        frontiers = occ.extract_frontiers(agent_xyz=xyz, pf=env.sim.pathfinder)
        intrinsic = habitat_camera_intrinsic(config)
        rgb, cands, pano_id = face_best_frontier_view(env, frontiers, intrinsic, yaw0)
        bev = occ.render_bev_from_env(env, sector_labels=EVEN_PANO_INDICES,
                                      frontiers=frontiers, paint_grid=False)
        save_rgb(os.path.join(args.out, "bev.png"), bev)
        ego = draw_annotated(rgb, "frontier", cands)
        save_rgb(os.path.join(args.out, "ego.png"), ego)
        dump_json(os.path.join(args.out, "meta.json"), {
            "n_frontiers": len(frontiers), "frontiers": frontiers, "in_view": cands,
            "ego_pano_id": pano_id,
        })
        print(f"frontiers={len(frontiers)} in_view={len(cands)} pano={pano_id}")
        print(args.out)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
