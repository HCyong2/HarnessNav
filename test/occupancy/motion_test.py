#!/usr/bin/env python
"""10 个 episode：扫圈后 occupancy A* 跟最近前沿，统计 miss / 原地转 / 前进。"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "test", "harness"))

from harness.protocol import dump_json
from nav.goto import body_position, pursue_occupancy
from nav.occupancy import EVEN_PANO_INDICES, OccupancyMap, save_rgb
from boot import boot_env

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="occupancy 跟随 10 epi")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--out", default=os.path.join(HERE, "output_motion"))
    return p.parse_args(argv)


def main(argv=None):
    """连跑若干集，每集跟一个前沿。"""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env, config = boot_env(seed=args.seed, gpu_id=args.gpu_id, episodes=max(args.episodes, 1))
    rows = []
    try:
        for epi in range(args.episodes):
            if epi > 0:
                env.reset()
            occ = OccupancyMap.from_config(config)
            occ.scan_around(env, rotate_times=12, look_down_floor=True)
            xyz = body_position(env)
            frontiers = occ.extract_frontiers(agent_xyz=xyz)
            if not frontiers:
                rows.append({"epi": epi, "n_frontiers": 0, "status": "no_frontier",
                             "steps": 0, "n_forward": 0, "n_turn": 0, "dist_m": 0.0})
                print(rows[-1])
                continue
            goal = min(
                frontiers,
                key=lambda fr: float(fr["geodesic_m"]) if fr.get("geodesic_m") is not None
                else 1e9)
            start = body_position(env)
            out = pursue_occupancy(env, occ, goal["xyz"], max_steps=args.steps,
                                   success_dist=0.35)
            end = body_position(env)
            moved = float(np.hypot(end[0] - start[0], end[2] - start[2]))
            n_fwd = int(out.get("n_forward") or 0)
            n_turn = int(out.get("n_turn") or 0)
            status = "ok"
            if out["steps"] == 0 or (n_fwd == 0 and n_turn == 0):
                status = "miss"
            elif n_fwd == 0 and n_turn > 3:
                status = "spin"
            rows.append({
                "epi": epi,
                "n_frontiers": len(frontiers),
                "goal": goal.get("fid"),
                "status": status,
                "steps": out["steps"],
                "n_forward": n_fwd,
                "n_turn": n_turn,
                "arrived": out["arrived"],
                "blocked": out["blocked"],
                "dist_m": out["dist_moved_m"],
                "disp_m": moved,
            })
            bev = occ.render_bev_from_env(env, sector_labels=EVEN_PANO_INDICES,
                                          frontiers=frontiers, paint_grid=True,
                                          path_xyz=out.get("path_xz"))
            save_rgb(os.path.join(args.out, f"epi{epi}_bev.png"), bev)
            print(rows[-1])
    finally:
        env.close()
    n_bad = sum(1 for r in rows if r["status"] in ("miss", "spin", "no_frontier"))
    dump_json(os.path.join(args.out, "meta.json"), {"rows": rows, "n_bad": n_bad})
    print(f"bad={n_bad}/{len(rows)}")
    print(args.out)
    return 0 if n_bad <= max(1, args.episodes // 10) else 1


if __name__ == "__main__":
    sys.exit(main())
