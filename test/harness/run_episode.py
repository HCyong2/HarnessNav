#!/usr/bin/env python
"""P0 端到端：脚本 Planner/Mover 跑若干 ScanNode。"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "test", "harness"))

from harness.loop import Harness
from boot import boot_env, maybe_glee

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "output")


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="Harness P0 episode")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-scans", type=int, default=3)
    p.add_argument("--no-glee", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    """跑一集脚本导航。"""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env, config = boot_env(seed=args.seed, gpu_id=args.gpu_id)
    backend = None if args.no_glee else maybe_glee(device=args.device)
    try:
        goal = str(getattr(env.current_episode, "object_category", "chair"))
        hns = Harness(env, config, args.out, goal=goal, backend=backend)
        summary = hns.run(max_scans=args.max_scans)
        print(summary)
        print(args.out)
    finally:
        if backend is not None:
            backend.close()
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
