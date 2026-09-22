#!/usr/bin/env python
"""Look Skill：平视 / 低头 / 抬头三张并排。"""

import argparse
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "test", "harness"))

from harness.protocol import dump_json
from harness.skills.look import restore_pitch, run_look
from nav.occupancy import save_rgb, to_rgb_uint8
from boot import boot_env

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="Look skill 可视化")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default=os.path.join(HERE, "look", "output"))
    p.add_argument("--gpu-id", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    """落盘 look_level / look_down / look_up。"""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env, config = boot_env(seed=args.seed, gpu_id=args.gpu_id)
    try:
        pitch = 0
        rgb0 = to_rgb_uint8(env.sim.get_sensor_observations()["rgb"])
        save_rgb(os.path.join(args.out, "look_level.png"), rgb0)

        out_down, pitch = run_look(env, "down", pitch)
        rgb1 = to_rgb_uint8(env.sim.get_sensor_observations()["rgb"])
        save_rgb(os.path.join(args.out, "look_down.png"), rgb1)

        pitch = restore_pitch(env, pitch)
        out_up, pitch = run_look(env, "up", pitch)
        rgb2 = to_rgb_uint8(env.sim.get_sensor_observations()["rgb"])
        save_rgb(os.path.join(args.out, "look_up.png"), rgb2)
        restore_pitch(env, pitch)

        h = min(rgb0.shape[0], rgb1.shape[0], rgb2.shape[0])
        w = min(rgb0.shape[1], rgb1.shape[1], rgb2.shape[1])
        strip = np.concatenate([cv2.resize(f, (w, h)) for f in (rgb0, rgb1, rgb2)], axis=1)
        save_rgb(os.path.join(args.out, "look_strip.png"), strip)
        dump_json(os.path.join(args.out, "meta.json"), {
            "look_down": out_down, "look_up": out_up,
        })
        print(args.out)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
