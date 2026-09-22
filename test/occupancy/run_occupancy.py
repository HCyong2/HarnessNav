#!/usr/bin/env python
"""在 HM3D 场景里环视建图，保存全景与点云 BEV。

流程
----
a. 在出生点环视一圈，输出全景图和 BEV。
b. 前进若干步（走不动就先转身），再环视一圈，再次输出。

BEV 会保留两次扫描之间的历史轨迹：历史环视点为蓝色空心节点并标从 0 起的蓝序号，
当前所在点只画红三角。节点之间蓝线相连。

用法
----
    python test/occupancy/run_occupancy.py
    python test/occupancy/run_occupancy.py --bev-mode bw --pano-even
    python test/occupancy/run_occupancy.py --no-merge --pano-indices 0,3,6,9
"""

import argparse
import os
import sys

import habitat
import numpy as np
from habitat.config.read_write import read_write

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nav.occupancy import (  # noqa: E402
    DEFAULT_PANO_INDICES,
    EVEN_PANO_INDICES,
    HAB_FORWARD,
    HAB_RIGHT,
    OccupancyMap,
    save_panorama,
    save_rgb,
)
from play2nav.config import check_split, play_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, "out")
STUCK_EPS_M = 1e-4


def parse_indices(text):
    """把 ``1,3,5`` 这样的字符串解析成整数元组。

    Args:
        text (str): 逗号分隔的序号。

    Returns:
        tuple: 整数序号。
    """
    parts = [p.strip() for p in str(text).split(",") if p.strip() != ""]
    return tuple(int(p) for p in parts)


def parse_args(argv=None):
    """解析命令行。"""
    parser = argparse.ArgumentParser(description="环视扫描并保存全景 / BEV 占用图")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--stage", default="val")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--rotate-times", type=int, default=12, help="环视张数，每张 30°")
    parser.add_argument("--bev-mode", choices=("color", "bw", "both"), default="both",
                        help="BEV 点云：彩色、黑白占用，或两种都存（默认 both）")
    parser.add_argument("--pano-even", action="store_true",
                        help="保存偶数朝向 0,2,4,6,8,10（0 为未转动的当前观测）")
    parser.add_argument("--pano-indices", default=None,
                        help="覆盖要保存的全景序号，逗号分隔，如 1,3,5,7,9,11")
    parser.add_argument("--no-merge", action="store_true",
                        help="全景分别保存；默认拼成一张")
    parser.add_argument("--no-look-down", action="store_true",
                        help="环视时不低头扫地面")
    parser.add_argument("--move-steps", type=int, default=12,
                        help="两次扫描之间尝试前进的步数")
    parser.add_argument("--voxel-size", type=float, default=0.10)
    return parser.parse_args(argv)


def dump_scan(occ, env, out_dir, tag, images, indices, merge, rotate_times, bev_mode):
    """把一次环视的全景和 BEV 落到磁盘。

    Args:
        occ (OccupancyMap): 占用图。
        env: ``habitat.Env``。
        out_dir (str): 输出目录。
        tag (str): 文件名前缀，如 ``scan0``。
        images (list): 环视 RGB。
        indices (tuple): 要保存的全景序号。
        merge (bool): 是否拼成一张。
        rotate_times (int): 一圈朝向数。
        bev_mode (str): ``color`` / ``bw`` / ``both``。

    Returns:
        list: 写出的路径。
    """
    paths = save_panorama(images, out_dir, f"{tag}_pano", indices=indices, merge=merge)
    modes = ("color", "bw") if bev_mode == "both" else (bev_mode,)
    for mode in modes:
        bev = occ.render_bev_from_env(env, color_mode=mode, resolution=0.02, point_size=3,
                                      sector_labels=indices, rotate_times=rotate_times)
        bev_path = os.path.join(out_dir, f"{tag}_bev_{mode}.png")
        save_rgb(bev_path, bev)
        paths.append(bev_path)
    n_pts = len(occ.global_pcd.points)
    print(f"[{tag}] pano frames={len(images)} pano_files={len(paths) - len(modes)}  "
          f"pcd={n_pts}  scan_nodes={len(occ.scan_nodes)}  "
          f"path={len(occ.path_positions)}")
    for p in paths:
        print(f"         {p}")
    return paths


def walk_forward(env, occ, n_steps):
    """尝试前进；被挡住就右转再试。

    Args:
        env: ``habitat.Env``。
        occ (OccupancyMap): 用来记下轨迹。
        n_steps (int): 希望成功前进的步数。

    Returns:
        int: 实际产生位移的前进步数。
    """
    moved = 0
    turns = 0
    attempts = 0
    max_attempts = n_steps * 4
    while moved < n_steps and attempts < max_attempts and not env.episode_over:
        attempts += 1
        before = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
        env.step(HAB_FORWARD)
        after = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
        occ.record_body(after, min_step=0.0)
        occ.integrate_from_env(env)
        if np.linalg.norm(after - before) > STUCK_EPS_M:
            moved += 1
            turns = 0
            continue
        env.step(HAB_RIGHT)
        occ.record_body(np.asarray(env.sim.get_agent_state().position, dtype=np.float64),
                        min_step=0.0)
        occ.integrate_from_env(env)
        turns += 1
        if turns >= 12:
            print("  walk: 转满一圈仍走不动，停止移动")
            break
    print(f"  walk: moved {moved}/{n_steps} forward step(s)")
    return moved


def main(argv=None):
    """建环境，扫两圈，落盘。

    Returns:
        int: 退出码。
    """
    args = parse_args(argv)
    if args.pano_indices is not None:
        indices = parse_indices(args.pano_indices)
    elif args.pano_even:
        indices = EVEN_PANO_INDICES
    else:
        indices = DEFAULT_PANO_INDICES
    merge = not args.no_merge
    os.makedirs(args.out, exist_ok=True)

    config, fixed, yaml_success = play_config(
        stage=args.stage, episodes=1, seed=args.seed, gpu_id=args.gpu_id)
    problems = check_split(args.stage, config)
    if problems:
        raise SystemExit("数据路径对不上:\n  " + "\n  ".join(problems))
    with read_write(config):
        config.habitat.environment.iterator_options.shuffle = False

    env = habitat.Env(config)
    env.reset()
    occ = OccupancyMap.from_config(config, voxel_size=args.voxel_size)
    occ.record_body(env.sim.get_agent_state().position, min_step=0.0)

    scene = os.path.basename(env.current_episode.scene_id).split(".")[0]
    print(f"scene      : {scene}  ep {env.current_episode.episode_id}")
    print(f"pano idx   : {indices}  merge={merge}")
    print(f"bev mode   : {args.bev_mode}")
    print(f"output     : {args.out}")
    print(f"success yaml: {yaml_success:.2f} (本脚本不发 stop)\n")

    try:
        print("# a. 出生点环视")
        images0 = occ.scan_around(env, rotate_times=args.rotate_times,
                                  look_down_floor=not args.no_look_down)
        dump_scan(occ, env, args.out, "scan0", images0, indices, merge,
                  args.rotate_times, args.bev_mode)

        print("# b. 移动后再环视")
        walk_forward(env, occ, args.move_steps)
        images1 = occ.scan_around(env, rotate_times=args.rotate_times,
                                  look_down_floor=not args.no_look_down)
        dump_scan(occ, env, args.out, "scan1", images1, indices, merge,
                  args.rotate_times, args.bev_mode)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
