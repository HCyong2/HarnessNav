#!/usr/bin/env python
"""Mover 输入图：semantic mask+id 或画面内前沿。"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "test", "harness"))

from harness.overlay import (draw_annotated, face_best_frontier_view,
                             semantic_candidates)
from harness.protocol import dump_json
from nav.goto import body_yaw_env
from nav.occupancy import OccupancyMap, clean_depth, save_rgb
from nav.transform import habitat_camera_intrinsic
from boot import GLEE_TEST_QUERIES, boot_env, find_glee_view, maybe_glee

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="Mover 标注可视化")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default=os.path.join(HERE, "output"))
    p.add_argument("--mode", choices=("semantic", "frontier"), default="semantic")
    p.add_argument("--text", default="", help="semantic 优先查询词；空则试 door/bed/window")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def main(argv=None):
    """画一张 Ego annotated。"""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env, config = boot_env(seed=args.seed, gpu_id=args.gpu_id)
    backend = maybe_glee(device=args.device) if args.mode == "semantic" else None
    try:
        occ = OccupancyMap.from_config(config)
        yaw0 = body_yaw_env(env)
        occ.scan_around(env, rotate_times=12, look_down_floor=True)
        spec = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
        intrinsic = habitat_camera_intrinsic(config)
        kept = None
        pano_id = None
        query = args.text.strip() or None
        xyz = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
        if args.mode == "semantic":
            queries = [query] if query else list(GLEE_TEST_QUERIES)
            rgb, query, result, pano_id = find_glee_view(env, backend, queries, node_yaw=yaw0)
            depth = clean_depth(env.sim.get_sensor_observations()["depth"],
                                float(spec.min_depth), float(spec.max_depth))
            cands, kept = semantic_candidates(result, depth, query or "obj")
        else:
            fr = occ.extract_frontiers(xyz, env.sim.pathfinder)
            rgb, cands, pano_id = face_best_frontier_view(env, fr, intrinsic, yaw0)
        ego = draw_annotated(rgb, args.mode, cands, kept)
        save_rgb(os.path.join(args.out, "ego_annotated.png"), ego)
        dump_json(os.path.join(args.out, "meta.json"), {
            "mode": args.mode, "query": query, "pano_id": pano_id, "candidates": cands,
        })
        print(f"mode={args.mode} query={query} pano={pano_id} candidates={len(cands)}")
        print(args.out)
    finally:
        if backend is not None:
            backend.close()
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
