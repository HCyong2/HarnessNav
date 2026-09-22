#!/usr/bin/env python
"""Recall：两处扫描后取回节点 0 的图，与当前 FOV 并排。"""

import argparse
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "test", "harness"))

from harness.loop import Harness
from harness.protocol import dump_json
from harness.skills.recall import run_recall
from nav.occupancy import EVEN_PANO_INDICES, HAB_FORWARD, HAB_RIGHT, save_rgb, to_rgb_uint8
from boot import boot_env

HERE = os.path.dirname(os.path.abspath(__file__))
STUCK = 1e-4


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="Recall skill 可视化")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default=os.path.join(HERE, "recall", "output"))
    p.add_argument("--gpu-id", type=int, default=0)
    return p.parse_args(argv)


def walk(env, n=8):
    """前进若干步，卡住就右转。"""
    moved = 0
    turns = 0
    while moved < n and turns < 12 and not env.episode_over:
        before = np.asarray(env.sim.get_agent_state().position)
        env.step(HAB_FORWARD)
        after = np.asarray(env.sim.get_agent_state().position)
        if np.linalg.norm(after - before) > STUCK:
            moved += 1
            turns = 0
        else:
            env.step(HAB_RIGHT)
            turns += 1


def main(argv=None):
    """Scan、走开再 Scan，Recall 节点 0。"""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env, config = boot_env(seed=args.seed, gpu_id=args.gpu_id)
    try:
        hns = Harness(env, config, args.out, backend=None)
        hns.scan_node()
        walk(env, 10)
        hns.scan_node()
        now = to_rgb_uint8(env.sim.get_sensor_observations()["rgb"])
        rec = run_recall(hns.graph, 0, pano_id=0, query="where was the opening")
        old = rec["images"][0] if rec.get("images") else None
        save_rgb(os.path.join(args.out, "current_fov.png"), now)
        if old is not None:
            save_rgb(os.path.join(args.out, "recall_n0_dir0.png"), old)
            hgt = min(now.shape[0], old.shape[0])
            w = min(now.shape[1], old.shape[1])
            strip = np.concatenate(
                [cv2.resize(now, (w, hgt)), cv2.resize(old, (w, hgt))], axis=1)
            save_rgb(os.path.join(args.out, "recall_side_by_side.png"), strip)
        dump_json(os.path.join(args.out, "meta.json"),
                  {k: v for k, v in rec.items() if k != "images"})
        nodes, edges = hns.graph.overlays()
        xyz = env.sim.get_agent_state().position
        bev = hns.occ.render_bev_from_env(
            env, sector_labels=EVEN_PANO_INDICES, node_overlays=nodes, edges=edges,
            frontiers=hns.occ.extract_frontiers(xyz, env.sim.pathfinder))
        save_rgb(os.path.join(args.out, "recall_bev.png"), bev)
        print(f"nodes={len(hns.graph.nodes)}")
        print(args.out)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
