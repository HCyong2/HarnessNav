#!/usr/bin/env python
"""Verify：视点 0、侧移/原地、回 home；BEV 上画 home，确认不新建 ScanNode。"""

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
from harness.skills.verify import run_verify
from nav.occupancy import EVEN_PANO_INDICES, save_rgb
from boot import boot_env, maybe_glee

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="Verify 可视化")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default=os.path.join(HERE, "verify", "output"))
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--text", default="door",
                    help="核实目标词，默认 door；可改 bed / window")
    return p.parse_args(argv)


def main(argv=None):
    """Scan 一次后 Verify，对比节点数。"""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env, config = boot_env(seed=args.seed, gpu_id=args.gpu_id)
    backend = maybe_glee(device=args.device)
    try:
        hns = Harness(env, config, args.out, goal=args.text, backend=backend)
        ctx = hns.scan_node()
        n_before = len(hns.graph.nodes)
        out = run_verify(env, hns.occ, hns.min_depth, hns.max_depth,
                         args.text, 0)
        n_after = len(hns.graph.nodes)
        for i, img in enumerate(out.get("views") or []):
            save_rgb(os.path.join(args.out, f"verify_view{i}.png"), img)
        nodes, edges = hns.graph.overlays()
        bev = hns.occ.render_bev_from_env(env, sector_labels=EVEN_PANO_INDICES,
                                          node_overlays=nodes, edges=edges)
        save_rgb(os.path.join(args.out, "verify_bev.png"), bev)
        dump_json(os.path.join(args.out, "meta.json"), {
            "nodes_before": n_before, "nodes_after": n_after,
            "result": {k: v for k, v in out.items() if k != "views"},
            "scan_node": ctx["node_id"],
        })
        print(f"nodes {n_before}->{n_after} consistency={out.get('consistency')}")
        print(args.out)
    finally:
        if backend is not None:
            backend.close()
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
