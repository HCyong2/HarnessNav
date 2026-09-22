#!/usr/bin/env python
"""Depth Skill：在实例框上写 ``id 深度``。"""

import argparse
import os
import sys

import cv2

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "test", "harness"))

from harness.protocol import dump_json
from harness.skills.depth import run_depth
from nav.occupancy import OccupancyMap, save_rgb, to_rgb_uint8
from perception.render import overlay_instances
from boot import GLEE_TEST_QUERIES, boot_env, find_glee_view, maybe_glee

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="Depth skill 可视化")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default=os.path.join(HERE, "depth", "output"))
    p.add_argument("--text", default="", help="优先查询词；空则依次试 door/bed/window")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def main(argv=None):
    """环视找得到分割的朝向，再量深度。"""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env, config = boot_env(seed=args.seed, gpu_id=args.gpu_id)
    backend = maybe_glee(device=args.device)
    queries = [args.text] if args.text.strip() else list(GLEE_TEST_QUERIES)
    try:
        occ = OccupancyMap.from_config(config)
        occ.scan_around(env, rotate_times=12, look_down_floor=True)
        rgb, query, result, pano_id = find_glee_view(env, backend, queries)
        sensors = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
        out = {"ok": False, "instances": [], "query": query, "pano_id": pano_id}
        canvas = to_rgb_uint8(rgb)
        if query:
            out = run_depth(env, backend, query, None, float(sensors.min_depth),
                            float(sensors.max_depth), rgb=canvas)
            out["query"] = query
            out["pano_id"] = pano_id
            if len(result) > 0:
                canvas = overlay_instances(canvas, result)
            for inst in out.get("instances") or []:
                u, v = inst["uv"]
                cv2.putText(canvas, f"{inst['id']} {inst['depth_m']:.2f}m",
                            (u, max(20, v - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2, cv2.LINE_AA)
        save_rgb(os.path.join(args.out, "depth_anno.png"), canvas)
        dump_json(os.path.join(args.out, "meta.json"), out)
        print(out)
        print(args.out)
    finally:
        if backend is not None:
            backend.close()
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
