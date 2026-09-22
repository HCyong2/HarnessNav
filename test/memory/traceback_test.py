#!/usr/bin/env python
"""TraceBack：从第二节点走回节点 0，BEV 画轨迹。"""

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
    p = argparse.ArgumentParser(description="TraceBack 可视化")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default=os.path.join(HERE, "traceback", "output"))
    p.add_argument("--gpu-id", type=int, default=0)
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
    """两处扫描后回溯到 0。"""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env, config = boot_env(seed=args.seed, gpu_id=args.gpu_id)
    try:
        hns = Harness(env, config, args.out, backend=None)
        hns.scan_node()
        walk(env, 12)
        hns.scan_node()
        before = np.asarray(env.sim.get_agent_state().position)
        result = hns.do_traceback(0)
        after = np.asarray(env.sim.get_agent_state().position)
        nodes, edges = hns.graph.overlays()
        bev = hns.occ.render_bev_from_env(env, sector_labels=EVEN_PANO_INDICES,
                                          node_overlays=nodes, edges=edges,
                                          frontiers=hns.occ.extract_frontiers(after, env.sim.pathfinder))
        save_rgb(os.path.join(args.out, "traceback_bev.png"), bev)
        dump_json(os.path.join(args.out, "meta.json"), {
            "result": result, "start_xyz": before.tolist(), "end_xyz": after.tolist(),
            "target_xyz": hns.graph.nodes[0]["xyz"], "target_yaw": hns.graph.nodes[0]["yaw"],
        })
        print(result)
        print(args.out)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
