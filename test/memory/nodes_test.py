#!/usr/bin/env python
"""节点图：2～3 个 ScanNode 的 BEV + history.json。"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "test", "harness"))

from harness.loop import Harness
from harness.protocol import dump_json
from nav.occupancy import EVEN_PANO_INDICES, HAB_FORWARD, HAB_RIGHT, save_rgb
from boot import boot_env

HERE = os.path.dirname(os.path.abspath(__file__))
STUCK = 1e-4


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="节点图可视化")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default=os.path.join(HERE, "nodes", "output"))
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--scans", type=int, default=3)
    return p.parse_args(argv)


def walk(env, n=10):
    """前进若干步。"""
    moved, turns = 0, 0
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
    """多次扫描并写出 history。"""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env, config = boot_env(seed=args.seed, gpu_id=args.gpu_id)
    try:
        hns = Harness(env, config, args.out, backend=None)
        for i in range(args.scans):
            hns.scan_node()
            if i + 1 < args.scans:
                walk(env, 10)
        xyz = env.sim.get_agent_state().position
        fr = hns.occ.extract_frontiers(xyz, env.sim.pathfinder)
        nodes, edges = hns.graph.overlays()
        bev = hns.occ.render_bev_from_env(env, sector_labels=EVEN_PANO_INDICES,
                                          node_overlays=nodes, edges=edges, frontiers=fr)
        save_rgb(os.path.join(args.out, "nodes_bev.png"), bev)
        hist = hns.graph.history(hns.current_id)
        dump_json(os.path.join(args.out, "history.json"), hist)
        dump_json(os.path.join(args.out, "meta.json"), {
            "n_nodes": len(hns.graph.nodes), "n_edges": len(hns.graph.edges),
            "current_id": hns.current_id,
        })
        print(f"nodes={len(hns.graph.nodes)} edges={len(hns.graph.edges)}")
        print(args.out)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
