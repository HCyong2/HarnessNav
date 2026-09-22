#!/usr/bin/env python
"""M0：在 Habitat HM3D 场景中选中一个语义目标，抬升到 3D，吸附到 navmesh，用 habitat 原生代码导航。

两种模式：

* **交互模式（默认）** —— 每轮输入一个目标词，用该文本分割当前帧，把最佳实例的中心像素
  反投影成 3D 点、吸附到 navmesh，再让 habitat 的跟随器走过去。
* **``--oracle``** —— 目标直接取 episode 标注的 view point，不分割、不输入，一路走到成功
  或超步，并把 habitat 原生的 ``spl`` / ``success`` / ``soft_spl`` 在多个 episode 上取平均。

可行走面取自场景预烘焙的 navmesh（``sim.pathfinder``），路径用 ``find_path``，离散动作由
``sim.make_greedy_follower`` 生成，指标取自 ``env.get_metrics()``。

保存（``--no-save`` 可关闭），``n`` 为累计已执行的移动数：

* ``<stem>_<n>.png``      原始观测
* ``<stem>_<n>_seg.png``  分割叠加图（仅交互模式）
* ``<stem>_<n>_map.png``  navmesh 俯视地图 + 轨迹 + 测地路径 + 目标标记

用法
-----
    python test/pix2move/run_pix2move.py                  # 交互
    python test/pix2move/run_pix2move.py --seed 3
    python test/pix2move/run_pix2move.py --oracle --episodes 20   # 跑 baseline 拿 SPL
"""

import argparse
import gc
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness import quiet_sim  # noqa: F401  须在 habitat 之前

import cv2
import habitat
import numpy as np
import torch
from habitat.config.read_write import read_write
from habitat.utils.visualizations.maps import colorize_topdown_map, draw_agent

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, "obs")

from nav.goto import (  # noqa: E402
    goal_from_mask,
    make_follower,
    oracle_goal_world,
    pursue,
    snap_goal_to_navmesh,
)
from nav.habitat_config import hm3d_config  # noqa: E402
from nav.occupancy import (  # noqa: E402
    HAB_FORWARD,
    HAB_LEFT,
    HAB_LOOK_DOWN,
    HAB_LOOK_UP,
    HAB_RIGHT,
    HAB_STOP,
    clean_depth,
    sensor_pose,
)
from nav.transform import (  # noqa: E402
    habitat_camera_intrinsic,
    habitat_rotation,
    habitat_translation,
    mapper_local_to_world,
)
from perception import mask_center_pixel  # noqa: E402
from perception.render import save_image, overlay_instances, draw_info_bar  # noqa: E402

# 交互模式下的手动微调指令，一条指令执行一个动作（转身 30° / 前进 0.25 m）。
# 这些词必须在送去做分割之前拦截：GLEE 是开放词表，阈值调低时什么词都能匹配出一个实例。
MANUAL_ACTIONS = {
    "left": HAB_LEFT,
    "right": HAB_RIGHT,
    "forward": HAB_FORWARD,
    "up": HAB_LOOK_UP,
    "down": HAB_LOOK_DOWN,
}

# 叠加在俯视地图上（RGB）。
TRAJ_COLOR = (255, 0, 255)          # 品红：agent 走过的地方
PATH_COLOR = (0, 200, 255)          # 橙黄：navmesh 上的测地路径（接下来要走的）
GOAL_COLOR = (255, 0, 0)            # 红色空心圆环：吸附到 navmesh 后的目标点
RAW_GOAL_COLOR = (255, 230, 0)      # 黄色实心点：反投影得到的原始点（在物体表面上）


# --------------------------------------------------------------------------- #
# 仿真器与相机
# --------------------------------------------------------------------------- #

def build_backend(name: str, args):
    """按名字构造一个分割后端。

    后端模块在这里按需导入，跑 ``--oracle`` 或只用其中一个后端时不必付出另一个后端的
    导入开销，因此这几处 import 有意不提到文件顶部。

    Args:
        name (str): ``glee`` 或 ``gdino_sam``。
        args: argparse 的解析结果，提供各后端所需的阈值与设备参数。

    Returns:
        SegBackend: 对应后端的实例。

    Raises:
        ValueError: ``name`` 不是已知的后端名。
    """
    if name == "glee":
        from perception.glee_backend import GLEEBackend

        return GLEEBackend(device=args.device, score_threshold=args.glee_threshold,
                           num_inst_select=args.max_instances)
    if name == "gdino_sam":
        from perception.gdino_sam_backend import GDinoSAMBackend

        return GDinoSAMBackend(device=args.device, sam_type=args.sam_type,
                               box_threshold=args.box_threshold,
                               text_threshold=args.text_threshold,
                               max_instances=args.max_instances)
    raise ValueError(f"unknown backend: {name}")


def sensor_specs(config):
    """从环境配置读传感器量程与视场角，不写死。

    Returns:
        tuple: ``(min_depth, max_depth, hfov_deg)``。
    """
    sensors = config.habitat.simulator.agents.main_agent.sim_sensors
    spec = getattr(sensors, "depth_sensor", None) or sensors.rgb_sensor
    return float(spec.min_depth), float(spec.max_depth), float(spec.hfov)


# --------------------------------------------------------------------------- #
# navmesh 俯视地图
# --------------------------------------------------------------------------- #

def topdown_geometry(env):
    """取俯视图的 ``(origin_xz, (row_span_m, col_span_m))``，源自 pathfinder 的边界。

    行列约定与 habitat 自己的 ``maps.to_grid(pos[2], pos[0], ...)`` 一致——注意那个函数
    的第一个参数是 habitat 的 **z**（形参却叫 ``realworld_x``），所以**行跨 z、列跨 x**。
    每轴跨度直接取 ``get_bounds``，不用 ``calculate_meters_per_pixel``：后者只返回一个标量
    （两轴取 min），只有正方形场景才同时正确。
    """
    lower_bound, upper_bound = env.sim.pathfinder.get_bounds()
    origin = (float(lower_bound[0]), float(lower_bound[2]))
    spans = (abs(float(upper_bound[2]) - float(lower_bound[2])),   # 行跨 z
             abs(float(upper_bound[0]) - float(lower_bound[0])))   # 列跨 x
    return origin, spans


def world_xz_to_pixel(point_xz, origin, spans, shape):
    """habitat 世界 ``(x, z)`` -> 俯视地图上的 ``(row, col)``。"""
    x, z = float(point_xz[0]), float(point_xz[1])
    return (int(round((z - origin[1]) * shape[0] / spans[0])),
            int(round((x - origin[0]) * shape[1] / spans[1])))


def build_topdown_image(env):
    """取 habitat 原生的 top_down_map measurement，上色后返回。

    Returns:
        tuple: ``(image_rgb, info)``，``info`` 里带雾区与 agent 位置等信息。

    Raises:
        RuntimeError: 配置里没有 top_down_map measurement。
    """
    info = env.get_metrics().get("top_down_map")
    if info is None:
        raise RuntimeError("top_down_map measurement 不在 config 里，无法绘制俯视地图")
    return colorize_topdown_map(info["map"], info["fog_of_war_mask"]), info


def draw_topdown_trajectory(canvas, origin, spans, trajectory_xz, path_xz=None,
                            raw_goal_xz=None, snapped_goal_xz=None):
    """把走过的轨迹、navmesh 测地路径和目标标记叠加到俯视地图上。

    坐标都是 habitat 世界 ``(x, z)``，与局部原点取在哪无关。标记尺寸按地图尺寸缩放而不是
    写死像素：habitat 的地图约一千像素宽，几像素的点和圆环在这个尺度下看不见，或者被
    agent 精灵盖住。

    Args:
        canvas (np.ndarray): 俯视地图，原地绘制。
        origin (tuple): 地图左上角对应的世界 ``(x, z)``。
        spans (tuple): ``(行跨度, 列跨度)``，米。
        trajectory_xz: agent 走过的轨迹，``(x, z)`` 序列。
        path_xz: navmesh 测地路径，``(x, z)`` 序列。
        raw_goal_xz: 反投影得到的原始目标点，``(x, z)``。
        snapped_goal_xz: 吸附后的导航目标点，``(x, z)``。

    Returns:
        np.ndarray: 地图与图例纵向拼接后的图像。
    """
    h, w = canvas.shape[:2]
    unit = max(min(h, w), 1)
    goal_radius = max(unit // 40, 10)
    goal_thick = max(goal_radius // 4, 2)
    raw_radius = max(unit // 90, 4)

    def to_px(point_xz):
        """世界 ``(x, z)`` -> 像素 ``(row, col)``；落在地图外时返回 ``None``。"""
        if point_xz is None:
            return None
        rc = world_xz_to_pixel(point_xz, origin, spans, (h, w))
        return rc if (0 <= rc[0] < h and 0 <= rc[1] < w) else None

    def polyline(points_xz, color, thickness):
        """把一串世界 ``(x, z)`` 点连成折线，画到地图外或跳变过大的段会被略过。"""
        px = [p for p in (to_px(p) for p in points_xz) if p is not None]
        for a, b in zip(px[:-1], px[1:]):
            # 比整张地图还长的跳变是瞬移，不是行走。
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 4 * max(h, w):
                continue
            cv2.line(canvas, (a[1], a[0]), (b[1], b[0]), color, thickness, cv2.LINE_AA)

    polyline(trajectory_xz, TRAJ_COLOR, 2)
    if path_xz is not None and len(path_xz):
        polyline(path_xz, PATH_COLOR, 2)

    raw_px, goal_px = to_px(raw_goal_xz), to_px(snapped_goal_xz)
    if raw_px is not None:
        cv2.circle(canvas, (raw_px[1], raw_px[0]), raw_radius, RAW_GOAL_COLOR, -1, cv2.LINE_AA)
    if goal_px is not None:
        cv2.circle(canvas, (goal_px[1], goal_px[0]), goal_radius, GOAL_COLOR,
                   goal_thick, cv2.LINE_AA)

    entries = [("agent path", TRAJ_COLOR)]
    if path_xz is not None and len(path_xz):
        entries.append(("navmesh path", PATH_COLOR))
    if raw_px is not None:
        entries.append(("goal target (on object)", RAW_GOAL_COLOR))
    if goal_px is not None:
        entries.append(("GOAL: heading here", GOAL_COLOR))

    legend = np.full((8 + 20 * len(entries), w, 3), 24, dtype=np.uint8)
    for i, (text, color) in enumerate(entries):
        y = 18 + i * 20
        cv2.circle(legend, (16, y - 5), 6, color, -1, cv2.LINE_AA)
        cv2.putText(legend, text, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([canvas, legend])


# --------------------------------------------------------------------------- #
# 保存与指标
# --------------------------------------------------------------------------- #

def to_rgb(obs):
    """观测 rgb 取成 (H, W, 3) uint8（habitat 可能返回 RGBA）。"""
    rgb = np.asarray(obs["rgb"])
    return np.ascontiguousarray(rgb[:, :, :3]).astype(np.uint8)


METRIC_KEYS = ("distance_to_goal", "success", "spl", "soft_spl")


def print_metrics(env, prefix="  "):
    """打印 habitat 原生指标。

    ``distance_to_goal`` / ``success`` 衡量的是 **episode 标注的目标**（view point），
    不是本脚本选中的像素目标；交互模式下它是一把独立的尺子，说明 agent 离真正的目标还有
    多远。``spl`` / ``soft_spl`` 在 agent 调用 stop 之前恒为 0，这是 habitat 的定义
    （``SPL = success · L / max(L, travelled)``），不是 bug。

    Returns:
        dict: ``env.get_metrics()`` 的结果。
    """
    m = env.get_metrics()
    parts = [f"{k}={m[k]:.3f}" for k in METRIC_KEYS if k in m]
    coll = m.get("collisions")
    if isinstance(coll, dict) and "count" in coll:
        parts.append(f"collisions={coll['count']}")
    print(f"{prefix}habitat: " + "  ".join(parts))
    return m


def save_observation(out_dir, stem, move_count, obs):
    """保存原始观测，并返回其 RGB 数组。

    Returns:
        np.ndarray: ``(H, W, 3)`` RGB uint8。
    """
    rgb = to_rgb(obs)
    save_image(os.path.join(out_dir, f"{stem}_{move_count}.png"), rgb)
    return rgb


def save_segmentation(out_dir, stem, move_count, rgb, result, header):
    """保存分割叠加图（mask + box + 中心像素）。"""
    canvas = overlay_instances(rgb, result)
    return save_image(os.path.join(out_dir, f"{stem}_{move_count}_seg.png"),
                      draw_info_bar(canvas, header))


def save_topdown_map(out_dir, stem, move_count, env, origin, spans, trajectory,
                     path_xz=None, raw_goal_xz=None, snapped_goal_xz=None, header=None):
    """保存俯视地图：agent 图标 + 轨迹 + 测地路径 + 目标标记。"""
    canvas, info = build_topdown_image(env)
    radius = max(min(canvas.shape[:2]) // 32, 4)
    for i in range(len(info["agent_map_coord"])):
        draw_agent(canvas, tuple(info["agent_map_coord"][i]),
                   float(info["agent_angle"][i]), radius)
    canvas = draw_topdown_trajectory(canvas, origin, spans, trajectory, path_xz=path_xz,
                                     raw_goal_xz=raw_goal_xz, snapped_goal_xz=snapped_goal_xz)
    if header:
        canvas = draw_info_bar(canvas, header)
    return save_image(os.path.join(out_dir, f"{stem}_{move_count}_map.png"), canvas)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析结果。
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="val", help="HM3D dataset split")
    parser.add_argument("--episodes", type=int, default=1, help="how many episodes to sample")
    parser.add_argument("--seed", type=int, default=None, help="dataset seed")
    parser.add_argument("--oracle", action="store_true",
                        help="aim at the episode's annotated view point (no segmentation) "
                             "and average SPL over --episodes")
    parser.add_argument("--backend", default="glee", choices=["glee", "gdino_sam"],
                        help="segmentation backend (glee is the default; gdino_sam is kept for ablations)")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="directory for observation images")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-actions", type=int, default=5,
                        help="actions to execute per iteration (interactive mode)")
    parser.add_argument("--goal-radius", type=float, default=0.25,
                        help="distance at which the follower declares arrival (m)")
    parser.add_argument("--success-distance", type=float, default=1.0,
                        help="habitat success threshold in metres; the HM3D ObjectNav "
                             "standard is 1.0, this repo's YAML says 0.25")
    parser.add_argument("--max-offset", type=float, default=2.5,
                        help="max snap distance in metres before a goal is rejected")
    parser.add_argument("--max-detour-m", type=float, default=3.0,
                        help="近距（直线 < 0.5 m）允许的最大绝对绕路量，米 (cross-wall guard)")
    parser.add_argument("--max-detour-ratio", type=float, default=4.0,
                        help="近距之外允许的最大 测地/直线 比 (cross-wall guard)")
    parser.add_argument("--no-save", action="store_true", help="do not write any images")
    parser.add_argument("--sam-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--glee-threshold", type=float, default=0.3)
    parser.add_argument("--max-instances", type=int, default=15)
    return parser.parse_args(argv)


def main(argv=None):
    """建立环境并进入所选模式。

    Returns:
        int: 进程退出码。
    """
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    config = hm3d_config(stage=args.stage, episodes=args.episodes)
    yaml_success = float(config.habitat.task.measurements.success.success_distance)
    with read_write(config):
        if args.seed is not None:
            config.habitat.seed = args.seed
        # 本仓库 YAML 给的是 0.25 m，HM3D ObjectNav 的标准判据是 1.0 m。用 0.25 m 时几乎
        # 不可能 stop 在那么小的范围内，success/SPL 会恒为 0，指标失去意义。
        config.habitat.task.measurements.success.success_distance = args.success_distance

    env = habitat.Env(config)
    # ``habitat.Env(config)`` 只构造、不会 reset；不 reset 就 step 会直接断言失败，
    # 而 current_episode 也是 reset 之后才有效。
    env.reset()
    pf = env.sim.pathfinder
    if not pf.is_loaded:
        raise SystemExit("navmesh 没有加载，无法使用 habitat 原生导航")

    intrinsic = habitat_camera_intrinsic(config)
    d_min, d_max, _ = sensor_specs(config)
    origin, spans = topdown_geometry(env)

    print(f"scene      : {os.path.basename(env.current_episode.scene_id).split('.')[0]}"
          f"  ep {env.current_episode.episode_id}")
    print(f"category   : {getattr(env.current_episode, 'object_category', 'n/a')}"
          f"  ({len(env.current_episode.goals)} annotated goal(s))")
    print(f"navmesh    : loaded, x[{origin[0]:.1f}, {origin[0] + spans[1]:.1f}] "
          f"z[{origin[1]:.1f}, {origin[1] + spans[0]:.1f}] m")
    print(f"intrinsics : fx=fy={intrinsic[0][0]:.1f} cx={intrinsic[0][2]:.1f} "
          f"cy={intrinsic[1][2]:.1f}")
    print(f"depth      : {d_min:.1f}-{d_max:.1f} m; {d_min:.1f} and {d_max:.1f} are no-hit sentinels")
    print(f"success    : within {args.success_distance:.2f} m of a view point "
          f"(YAML says {yaml_success:.2f}; HM3D ObjectNav standard is 1.00)")
    print(f"follower   : goal_radius={args.goal_radius:.2f} m, stop_key={HAB_STOP}")
    print(f"output     : {args.out}\n")

    max_steps = int(config.habitat.environment.max_episode_steps)
    backend = None
    try:
        if args.oracle:
            return run_oracle(env, args, max_steps)
        print(f"loading backend [{args.backend}] ...")
        backend = build_backend(args.backend, args)
        # 交互模式下不 reset，跟随器建一次就够。
        return run_interactive(env, pf, make_follower(env, args.goal_radius), args,
                               intrinsic, d_min, d_max, backend, max_steps)
    finally:
        if backend is not None:
            backend.close()
        gc.collect()
        torch.cuda.empty_cache()
        env.close()


def run_oracle(env, args, max_steps):
    """``--oracle`` 模式：瞄准标注的 view point 走到位，统计 SPL / success。

    Returns:
        int: 进程退出码，0 表示成功。
    """
    totals = {k: 0.0 for k in METRIC_KEYS}
    done, move_count = 0, 0

    while done < args.episodes:
        # 每集都重新取 pathfinder 与跟随器：``env.reset()`` 换场景时会连 navmesh 一起换掉，
        # 上一集留下的句柄要么在回答旧场景的查询、要么直接失效，都不能跨 reset 复用。
        pf, follower = env.sim.pathfinder, make_follower(env, args.goal_radius)

        stem = (f"{os.path.basename(env.current_episode.scene_id).split('.')[0]}"
                f"_{env.current_episode.episode_id}")
        print(f"--- {stem}  ({len(env.current_episode.goals)} goal(s), "
              f"category {getattr(env.current_episode, 'object_category', 'n/a')}) ---")

        goal_world = oracle_goal_world(env)
        if goal_world is None:
            print("  这个 episode 没有标注目标，跳过\n")
            done += 1
            env.reset()
            continue

        agent_world = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
        # guard=False：view point 是数据集给的标准答案，本来就该在 navmesh 上，拿"跨墙
        # 检查"去否决标注没有道理。这里只吸附 + 报告测地/欧氏，不拦。
        snapped, info = snap_goal_to_navmesh(
            pf, goal_world, agent_world, max_offset=args.max_offset,
            max_detour_m=args.max_detour_m, max_detour_ratio=args.max_detour_ratio,
            guard=False)
        if info["reject"]:
            print(f"  注意：{info['reject']}")
        if info["fallback"]:
            print(f"       目标不可达，只朝它推进到 {info['advance']:.0%} 处")
        if info["euclid"] is not None:
            print(f"  goal  : euclid={info['euclid']:.2f} m  "
                  f"geodesic={info['geodesic']:.2f} m  ratio={info['ratio']:.2f}")
        # 标注的 view point 连 find_path 都走不通，这个 episode 就是无解的，别再往下推。
        target = snapped if snapped is not None else goal_world

        steps, arrived = pursue(env, follower, target, max_steps)
        move_count += steps
        m = print_metrics(env)
        for k in METRIC_KEYS:
            totals[k] += float(m.get(k, 0.0))
        done += 1
        print(f"  ep {env.current_episode.episode_id}: {steps} step(s), "
              f"arrived={arrived}, episode_over={env.episode_over}\n")

        if done < args.episodes:
            env.reset()

    print("=" * 62)
    print(f"oracle over {done} episode(s), {move_count} move(s) total")
    for k in METRIC_KEYS:
        print(f"  mean {k:<16}: {totals[k] / max(done, 1):.4f}")
    return 0


def run_interactive(env, pf, follower, args, intrinsic, d_min, d_max, backend, max_steps):
    """交互模式：提示输入目标词 -> 分割 -> 像素 -> navmesh -> 跟着走。

    Returns:
        int: 进程退出码，0 表示正常结束。
    """
    origin, spans = topdown_geometry(env)
    obs = env.sim.get_sensor_observations()
    cam_pos = np.asarray(sensor_pose(env)[0], dtype=np.float64)
    origin_local = habitat_translation(cam_pos)
    trajectory = [(float(cam_pos[0]), float(cam_pos[2]))]
    move_count, saved_index = 0, -1

    while True:
        # 一个移动序号只落盘一次：不推进仿真器的轮次（提示词打错、分割没匹配到）不该
        # 用同样的内容重写同一张图。
        rgb = to_rgb(obs)
        if move_count != saved_index and not args.no_save:
            save_image(os.path.join(args.out, f"{stem_of(env)}_{move_count}.png"), rgb)
            save_topdown_map(args.out, stem_of(env), move_count, env, origin, spans, trajectory,
                             header=[f"{stem_of(env)}_{move_count}  |  {move_count} move(s)"])
            saved_index = move_count
        print(f"--- frame {stem_of(env)}_{move_count}.png ({move_count} move(s)) ---")
        print_metrics(env)

        try:
            target = input("target | left|right|forward|up|down | stop=quit: ").strip()
        except EOFError:
            print("\n[stdin closed] finishing.")
            break
        if not target:
            continue
        if target.lower() == "stop":
            print("stopped by user.")
            break

        # 手动微调必须在分割**之前**拦截，否则会被 GLEE 当成一个物体名匹配掉。
        nudge = MANUAL_ACTIONS.get(target.lower())
        if nudge is not None:
            obs = env.step(nudge)
            move_count += 1
            wp = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
            trajectory.append((float(wp[0]), float(wp[2])))
            print(f"  manual: {target} -> action {nudge} "
                  f"(1 step, {move_count} move(s) so far)\n")
            if env.episode_over:
                print("episode is over; stopping.")
                break
            continue

        result = backend.segment(rgb, target)
        if len(result) == 0:
            print(f"  no instance matched '{target}' -- try another phrase\n")
            continue
        best = result.top(1)
        mask = best.masks[0]

        depth = clean_depth(obs["depth"], d_min, d_max)
        center_px = mask_center_pixel(mask, depth)
        cam_pos = np.asarray(sensor_pose(env)[0], dtype=np.float64)
        cam_rot = habitat_rotation(sensor_pose(env)[1])
        pos_local = habitat_translation(cam_pos) - origin_local

        goal_local3 = goal_from_mask(depth, mask, intrinsic, pos_local, cam_rot, center_px)
        if goal_local3 is None:
            print("  the instance has no valid depth -- cannot localise it\n")
            continue

        # 原点必须传 origin_local（mapper 顺序），不能传 habitat 世界顺序的起点：
        # mapper_local_to_world 只置换点、不置换原点，传错会让标记静默落到地图之外。
        goal_world = mapper_local_to_world(goal_local3, origin_local)
        if abs(float(goal_world[2]) - float(origin_local[1])) > 30.0:
            print(f"  WARNING: goal world z={goal_world[2]:.2f} 离 agent 的 "
                  f"z={origin_local[1]:.2f} 太远，坐标系很可能搞混了")

        agent_world = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
        print(f"  instance      : {best.labels[0]} score={best.scores[0]:.3f} "
              f"size={int(mask.sum())}px  ({best.time_ms:.0f} ms)")
        print(f"  centre pixel  : {center_px}")
        print(f"  goal (world)  : x={goal_world[0]:+.2f} y={goal_world[1]:+.2f} "
              f"z={goal_world[2]:+.2f}")

        snapped, info = snap_goal_to_navmesh(
            pf, goal_world, agent_world, max_offset=args.max_offset,
            max_detour_m=args.max_detour_m, max_detour_ratio=args.max_detour_ratio)
        if snapped is None:
            print(f"  REJECTED: {info['reject']}")
            print(f"            (吸附位移 {info['offset']:.2f} m)\n")
            continue
        print(f"  snap          : offset={info['offset']:.2f} m  "
              f"euclid={info['euclid']:.2f} m  geodesic={info['geodesic']:.2f} m  "
              f"detour={info['excess']:.2f} m  ratio={info['ratio']:.2f}")
        if info["fallback"]:
            # 目标在另一个 navmesh 小岛上，走不到；这里只朝它靠近。
            print(f"  partial       : {info['reject']}")
            print(f"                  朝目标推进到 {info['advance']:.0%} 处就停"
                  f"（目标本身不可达，navmesh 分片）")
        elif info["euclid"] < 0.4:
            print("                  已经站在目标附近了")

        if not args.no_save:
            canvas = overlay_instances(rgb, result)
            save_image(os.path.join(args.out, f"{stem_of(env)}_{move_count}_seg.png"),
                       draw_info_bar(canvas, [
                           f"{stem_of(env)}_{move_count}  |  backend: {args.backend}"
                           f"  |  prompt: \"{target}\"",
                           f"Goal {best.labels[0]} {best.scores[0]:.2f} | centre px {center_px}"
                           f" | snap offset {info['offset']:.2f} m | detour {info['excess']:.2f} m",
                           f"goal_radius {args.goal_radius:.2f} m"
                           f"  |  navmesh geodesic {info['geodesic']:.2f} m",
                       ]))
            save_topdown_map(args.out, stem_of(env), move_count, env, origin, spans, trajectory,
                             path_xz=info["points"],
                             raw_goal_xz=(goal_world[0], goal_world[2]),
                             snapped_goal_xz=(snapped[0], snapped[2]),
                             header=[f"{stem_of(env)}_{move_count}  |  prompt: \"{target}\"  |  "
                                     f"geodesic {info['geodesic']:.2f} m  |  "
                                     f"detour {info['excess']:.2f} m"])

        before = info["geodesic"]
        # stop_at_goal=False：交互时走到位就停，但**不结束 episode**，否则到一次目标
        # 整个会话就没了。
        steps, arrived = pursue(env, follower, snapped, args.max_actions, stop_at_goal=False)
        move_count += steps
        obs = env.sim.get_sensor_observations()

        wp = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
        trajectory.append((float(wp[0]), float(wp[2])))
        _, after_info = snap_goal_to_navmesh(
            pf, goal_world, wp, max_offset=args.max_offset,
            max_detour_m=args.max_detour_m, max_detour_ratio=args.max_detour_ratio)
        after = after_info["geodesic"]
        after_txt = f"{after:.2f}" if after is not None else "?"
        print(f"  executed {steps} action(s); {move_count} move(s) so far. "
              f"navmesh distance to goal {before:.2f} -> {after_txt} m"
              f"{'  (arrived)' if arrived else ''}\n")

        if env.episode_over:
            print("episode is over; stopping.")
            break

    print(f"\nfinished. {move_count} move(s); images in {args.out}")
    return 0


def stem_of(env):
    """``<scene>_<episode_id>``，图名用。"""
    return (f"{os.path.basename(env.current_episode.scene_id).split('.')[0]}"
            f"_{env.current_episode.episode_id}")


if __name__ == "__main__":
    sys.exit(main())
