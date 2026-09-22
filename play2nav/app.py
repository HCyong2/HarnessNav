#!/usr/bin/env python
"""在 HM3D 场景里用键盘驾驶 agent，按 p 结束本集并落盘。

用法::

    python play2nav/app.py
    python play2nav/app.py --host 127.0.0.1
    python play2nav/app.py --headless --scripted "w,w,w,a,a,p" --selfcheck

键位：``w`` 前进 / ``a`` 左转 / ``d`` 右转 / ``q`` 抬头 / ``e`` 低头 / ``p`` 结束本集。
"""

import argparse
import gc
import os
import signal
import sys
import time
from collections import Counter

import habitat
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("HABITAT_LAB_LOG", "40")

from play2nav import recorder as rec                     # noqa: E402
from play2nav import topdown as td                       # noqa: E402
from play2nav.config import check_split, play_config, summary_lines  # noqa: E402
from play2nav.keystate import KEY_ACTION                 # noqa: E402
from play2nav.web import Play2NavServer                  # noqa: E402

# habitat 默认动作编号，与 config 里 actions 声明顺序一致。
HAB_STOP, HAB_FORWARD, HAB_LEFT, HAB_RIGHT = 0, 1, 2, 3
HAB_LOOK_UP, HAB_LOOK_DOWN = 4, 5

ACTION_IDS = {
    "move_forward": HAB_FORWARD,
    "turn_left": HAB_LEFT,
    "turn_right": HAB_RIGHT,
    "look_up": HAB_LOOK_UP,
    "look_down": HAB_LOOK_DOWN,
    "stop": HAB_STOP,
}
ACTION_NAMES = {v: k for k, v in ACTION_IDS.items()}

# sim 循环 tick 间隔。动作闸门靠时间戳，tick 越细长按节奏越准。
TICK_S = 0.02

# 前进后机身坐标几乎不变即视为卡住。
STUCK_EPS_M = 1e-4

DEFAULT_SCRIPT = "w,w,w,a,a,w,w,w,p"
DEFAULT_OUT_ROOT = "/DATA_HDD/hc/play2nav_outputs"


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

def to_rgb(obs):
    """把观测里的 rgb 取成 ``(H, W, 3)`` uint8。

    Args:
        obs (dict): habitat 观测。

    Returns:
        np.ndarray: RGB 图。
    """
    rgb = np.asarray(obs["rgb"])
    return np.ascontiguousarray(rgb[:, :, :3]).astype(np.uint8)


def jsonable(value):
    """把 numpy 标量/数组转成可 ``json.dump`` 的类型。

    Args:
        value: 任意嵌套结构。

    Returns:
        只含 Python 基本类型的对象。
    """
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def parse_script(text):
    """把 ``"w,w,a,p"`` 解析成动作名序列。

    Args:
        text (str): 逗号分隔的键位或动作名。

    Returns:
        list: habitat 动作名列表。

    Raises:
        ValueError: 空串或无法识别的 token。
    """
    tokens = [t.strip() for t in str(text).split(",") if t.strip()]
    if not tokens:
        raise ValueError("--scripted 是空的")
    out = []
    for token in tokens:
        key = token.lower()
        if key == "p":
            out.append("stop")
        elif key in KEY_ACTION:
            out.append(KEY_ACTION[key])
        elif key in ACTION_IDS:
            out.append(key)
        else:
            raise ValueError(
                f"--scripted 里的 '{token}' 不认识；可用："
                f"{sorted(KEY_ACTION) + ['p']} 或 {sorted(ACTION_IDS)}")
    return out


# --------------------------------------------------------------------------- #
# 一集的全部状态
# --------------------------------------------------------------------------- #

class Navigator:
    """持有 ``env`` 与当前 episode 状态，负责步进与落盘。

    ``start_episode`` 每次 ``env.reset()`` 后都会重取 pathfinder 与地图几何。
    """

    def __init__(self, env, args, out_root, max_episode_steps):
        """构造导航器。

        Args:
            env: habitat 环境。
            args: 命令行参数。
            out_root (str): 产物根目录。
            max_episode_steps (int): 单集最大步数。
        """
        self.env = env
        self.args = args
        self.out_root = out_root
        # 显式传进来，不去摸 env._config 的内部结构（habitat 各版本层级不一致）。
        self.max_steps = int(max_episode_steps)
        self.session_records = []
        self.episode_index = 0
        # 已出现过的 (scene, episode_id)，用来识别迭代器绕回。
        self.seen_episodes = set()
        self.repeated = False

        # 每集的字段，先给一份初值，避免 ``start_episode`` 抛异常时属性不存在。
        self.pf = None
        self.navmesh_ok = False
        self.origin = (0.0, 0.0)
        self.spans = (1.0, 1.0)
        self.scene_id = "?"
        self.episode_id = -1
        self.category = "n/a"
        self.obs = None
        self.rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        self.step_idx = 0
        self.actions = []
        self.path_length = 0.0
        self.trajectory = []
        self.episode_over = True
        self.finished = True
        self.terminated_by = None
        self.frame_seq = 0          # 俯视图帧号，在 ``_record_step`` 里递增
        self.emit_seq = 0           # 第一视角帧号，在 ``_emit_frame`` 里递增
        self.start_distance = float("nan")
        self.shortest_path_length = float("nan")
        self.wall_time_s = 0.0
        self.last_pos = None
        self.goal_xyz = None
        self.goal_ref_xz = None
        self.goal_ref_distance = float("nan")
        self.path_xz = None
        self.recorder = None
        self.episode_dir = None
        self.last_topdown = None          # 最近一帧俯视画布，网页直接编码复用
        self._topdown_base = None         # 底图上色缓存，按 ``topdown_refresh`` 间隔重算
        self._topdown_base_at = 0.0
        self.topdown_refresh = 0.0
        self.on_frame = None              # 第一视角出帧回调；headless 为 None
        self.last_action = None
        self.last_delta_m = 0.0
        self.last_collided = False
        self.last_stuck = False
        self.hint_logged = False          # 卡住提示是否已写入事件流
        self.last_error = None            # 换集失败等信息，随快照持续下发

    # -- 生命周期 --------------------------------------------------------- #

    def start_episode(self):
        """``env.reset()`` 并对新 episode 重新取全部句柄。

        Returns:
            str: 本集输出目录。
        """
        t0 = time.time()
        self.env.reset()

        # reset 之后重取：旧句柄会指向上一集的 navmesh。
        self.pf = self.env.sim.pathfinder
        self.navmesh_ok = bool(self.pf.is_loaded)
        self.origin, self.spans = td.topdown_geometry(self.env)

        episode = self.env.current_episode
        self.scene_id = os.path.basename(episode.scene_id).split(".")[0]
        self.episode_id = int(getattr(episode, "episode_id", -1))
        self.category = str(getattr(episode, "object_category", "n/a"))

        key = (str(episode.scene_id), str(getattr(episode, "episode_id", "")))
        self.repeated = key in self.seen_episodes
        self.seen_episodes.add(key)

        self.goal_xyz = None
        goals = getattr(episode, "goals", None)
        if goals:
            try:
                self.goal_xyz = [float(v) for v in goals[0].position]
            except Exception:
                self.goal_xyz = None

        self.obs = self.env.sim.get_sensor_observations()
        self.rgb = to_rgb(self.obs)

        self.step_idx = 0
        self.actions = []
        self.path_length = 0.0
        self.last_pos = self._agent_pos()
        self.trajectory = [self._xz(self.last_pos)]
        self._topdown_base = None
        self._topdown_base_at = 0.0
        self.last_action = None
        self.last_delta_m = 0.0
        self.last_collided = False
        self.last_stuck = False
        self.hint_logged = False
        self.episode_over = False
        self.finished = False
        self.terminated_by = None
        self.start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        self.wall_time_s = 0.0
        self.wall_t0 = t0

        metrics = self.env.get_metrics()
        self.start_distance = float(metrics.get("distance_to_goal", float("nan")))
        self.shortest_path_length = self._episode_shortest_path(episode)

        # 离 agent 最近的 goal view point，口径与 ``distance_to_goal`` 一致。
        self.goal_ref_xz = None
        self.goal_ref_distance = float("nan")
        if self.navmesh_ok and goals:
            try:
                self.goal_ref_xz, self.goal_ref_distance = td.closest_view_point(
                    self.env, self.last_pos, goals[0].view_points, episode)
            except Exception:
                self.goal_ref_xz = None

        # 本集最优路线只在换集时算一次，之后每帧复用；跨岛不可达则为 None。
        self.path_xz = None
        end_xyz = self.goal_ref_xz if self.goal_ref_xz is not None else self.goal_xyz
        if self.navmesh_ok and end_xyz is not None:
            self.path_xz = td.navmesh_path_xz(self.pf, self.last_pos, end_xyz)

        self.episode_dir = rec.make_episode_dir(
            self.out_root, self.scene_id, self.episode_id, self.category)
        self.recorder = rec.EpisodeRecorder(
            self.episode_dir, fps=self.args.video_fps,
            topdown_video_size=self.args.topdown_video_size,
            enabled=not self.args.no_save)
        self._record_step()
        self._emit_frame()

        self.episode_index += 1
        self.last_error = None
        return self.episode_dir

    def _episode_shortest_path(self, episode):
        """episode 标注里的最优路径长度（SPL 的分母）。"""
        info = getattr(episode, "info", None) or {}
        for key in ("shortest_path_length", "geodesic_distance"):
            value = info.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                return float(value)
        return self.start_distance

    # -- 位姿 ------------------------------------------------------------- #

    def _agent_pos(self):
        return np.asarray(self.env.sim.get_agent_state().position, dtype=np.float64)

    @staticmethod
    def _xz(pos):
        return (float(pos[0]), float(pos[2]))

    # -- 自检 ------------------------------------------------------------- #

    def verify_episode(self):
        """检查本集句柄与地图几何是否属于当前场景。

        Returns:
            list: 问题描述；空列表表示通过。
        """
        problems = []
        if not self.navmesh_ok:
            return ["navmesh 未加载"]

        pf = self.env.sim.pathfinder
        if self.pf is not pf:
            problems.append("pathfinder 句柄与 env.sim.pathfinder 不是同一对象")

        fresh_origin, fresh_spans = td.topdown_geometry(self.env)
        if not (np.allclose(fresh_origin, self.origin, atol=1e-6)
                and np.allclose(fresh_spans, self.spans, atol=1e-6)):
            problems.append(f"地图几何缓存已过期：缓存 origin={self.origin} spans={self.spans}"
                            f" vs 当前 origin={fresh_origin} spans={fresh_spans}")

        if self.trajectory and not td.scene_bounds_ok(pf, self.trajectory):
            lower, upper = pf.get_bounds()
            problems.append(
                f"轨迹越出本场景 navmesh 边界 x[{lower[0]:.1f},{upper[0]:.1f}] "
                f"z[{lower[2]:.1f},{upper[2]:.1f}]：起点 {self.trajectory[0]} "
                f"终点 {self.trajectory[-1]}")
        return problems

    # -- 步进 ------------------------------------------------------------- #

    def can_step(self):
        return (not self.finished) and (not self.episode_over) and self.navmesh_ok

    def step(self, action_id):
        """发一个动作。先推第一视角，再记俯视图与录像。

        Args:
            action_id (int): habitat 动作编号。

        Returns:
            bool: 实际发出动作为 True。
        """
        if not self.can_step():
            return False

        prev = self.last_pos
        self.obs = self.env.step(int(action_id))
        self.rgb = to_rgb(self.obs)

        self.step_idx += 1
        self.last_action = ACTION_NAMES.get(int(action_id), str(action_id))
        self.actions.append(self.last_action)

        pos = self._agent_pos()
        self.last_delta_m = float(np.linalg.norm(pos - prev))
        self.path_length += self.last_delta_m
        self.last_pos = pos
        self.trajectory.append(self._xz(pos))

        self.last_stuck = (self.last_action == "move_forward"
                           and self.last_delta_m <= STUCK_EPS_M)

        collisions = self._metrics().get("collisions")
        self.last_collided = bool(collisions.get("is_collision", False)) \
            if isinstance(collisions, dict) else False

        self._emit_frame()
        self._record_step()
        self.wall_time_s = time.time() - self.wall_t0

        if self.env.episode_over:
            self.episode_over = True
            self.terminated_by = "user_stop" if int(action_id) == HAB_STOP else "max_steps"
        return True

    def _emit_frame(self):
        """递增 ``emit_seq`` 并通知订阅者。无订阅者时为空操作。"""
        self.emit_seq += 1
        if self.on_frame is not None:
            self.on_frame()

    def stuck_hint(self):
        """前进被挡住时返回提示文案；否则 ``None``。

        Returns:
            str or None: 卡住提示。
        """
        if not self.can_step():
            return None
        if not self.last_stuck:
            return None
        return ("⚠ 碰到障碍物，无法移动（本步位移 0.000 m）。"
                "出生点顶着家具是常见情况，按住 a 或 d 转身，再按 w。")

    def _record_step(self):
        """生成俯视画布、递增 ``frame_seq``、写入录像。"""
        canvas = self.topdown_canvas()
        self.last_topdown = canvas
        self.frame_seq += 1
        if self.recorder is not None:
            self.recorder.add_step(self.rgb, canvas, caption=self.caption())

    # -- 画面 ------------------------------------------------------------- #

    def topdown_canvas(self, now=None, force=False):
        """当前俯视图：底图 + agent 图标 + 轨迹。

        Args:
            now (float, optional): 单调时钟；默认 ``time.monotonic()``。
            force (bool): 为 True 时忽略缓存、立刻重算底图。

        Returns:
            np.ndarray or None: RGB 画布。
        """
        now = time.monotonic() if now is None else now
        try:
            info = td.topdown_info(self.env)
            if (force or self._topdown_base is None or self.topdown_refresh <= 0
                    or (now - self._topdown_base_at) >= self.topdown_refresh):
                self._topdown_base = td.colorize_topdown(info)
                self._topdown_base_at = now
            canvas = self._topdown_base.copy()
        except Exception:
            return None
        td.draw_agents(canvas, info)
        start = self.trajectory[0] if self.trajectory else None
        return td.draw_trajectory(canvas, self.origin, self.spans, self.trajectory,
                                  start_xz=start, path_xz=self.path_xz)

    def caption(self, final=False):
        """俯视图顶部的英文信息条。过程中不显示 ``distance_to_goal``。

        Args:
            final (bool): 是否写入结算指标。

        Returns:
            list: 文本行。
        """
        lines = [f"{self.scene_id}  ep {self.episode_id}  |  target: {self.category}"
                 f"  |  step {self.step_idx}"]
        if final or self.episode_over:
            m = self._metrics()
            lines.append(
                f"success={float(m.get('success', float('nan'))):.3f}"
                f"  spl={float(m.get('spl', float('nan'))):.3f}"
                f"  soft_spl={float(m.get('soft_spl', float('nan'))):.3f}"
                f"  dist={float(m.get('distance_to_goal', float('nan'))):.3f} m"
                f"  ({self.terminated_by})")
        return lines

    def _metrics(self):
        try:
            return self.env.get_metrics()
        except Exception:
            return {}

    # -- 收尾 ------------------------------------------------------------- #

    def finish(self, terminated_by=None):
        """结束本集并落盘。幂等。

        Args:
            terminated_by (str, optional): 结束原因。

        Returns:
            dict or None: 本集记录；已结束过则为 None。
        """
        if self.finished:
            return None
        self.finished = True
        self.episode_over = True
        if terminated_by:
            self.terminated_by = terminated_by
        if not self.terminated_by:
            self.terminated_by = "quit_early"

        self.wall_time_s = max(self.wall_time_s, time.time() - self.wall_t0)
        metrics = self._metrics()
        record = self._build_record(metrics)

        canvas = self.topdown_canvas(force=True)
        if canvas is not None:
            canvas = td.draw_caption(canvas, self.caption(final=True))
        self.last_topdown = canvas
        self.frame_seq += 1
        if self.recorder is not None:
            self.recorder.finalize(record, trajectory_canvas=canvas)

        self.session_records.append(record)
        rec.write_session_summary(self.out_root, self.session_records)
        return record

    def _build_record(self, metrics):
        collisions = metrics.get("collisions")
        if isinstance(collisions, dict):
            collisions = {"count": int(collisions.get("count", 0)),
                          "is_collision": bool(collisions.get("is_collision", False))}
        else:
            collisions = None

        return jsonable({
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "object_category": self.category,
            "stage": self.args.stage,
            "seed": self.args.seed,
            "start_time": getattr(self, "start_time_str", ""),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_time_s": round(self.wall_time_s, 3),
            "num_steps": self.step_idx,
            "max_episode_steps": self.max_steps,
            "actions": list(self.actions),
            "action_counts": dict(Counter(self.actions)),
            "path_length_m": round(self.path_length, 4),
            "start_distance_to_goal": round(self.start_distance, 4),
            "shortest_path_length": round(self.shortest_path_length, 4),
            "success": metrics.get("success"),
            "spl": metrics.get("spl"),
            "soft_spl": metrics.get("soft_spl"),
            "distance_to_goal": metrics.get("distance_to_goal"),
            "collisions": collisions,
            "terminated_by": self.terminated_by,
            "success_distance": self.args.success_distance,
            "navmesh_loaded": self.navmesh_ok,
            "episode_dir": self.episode_dir,
        })

    def close(self):
        if self.recorder is not None:
            self.recorder.close()


# --------------------------------------------------------------------------- #
# 无显示器模式：走完全相同的 step/record 路径，只是用脚本代替键盘
# --------------------------------------------------------------------------- #

def run_headless(nav, args):
    script = parse_script(args.scripted)
    print(f"script     : {args.scripted}")
    print(f"episodes   : {args.episodes}\n")

    exit_code = 0
    previous_scene = None
    for index in range(args.episodes):
        episode_dir = nav.start_episode()
        print(f"--- episode {index + 1}/{args.episodes} ---")
        print(f"  scene    : {nav.scene_id}  ep {nav.episode_id}  target: {nav.category}")
        if nav.repeated:
            print(f"  note     : 本集重复（已跑完 --episodes {args.episodes} 集，"
                  f"迭代器绕回开头）—— 加大 --episodes 才有新场景")
        if index and previous_scene == nav.scene_id:
            print("  note     : 与上一集同场景，本集无法验证跨场景句柄是否失效")
        previous_scene = nav.scene_id
        print(f"  navmesh  : {'loaded' if nav.navmesh_ok else 'MISSING'}"
              f"  bounds x[{nav.origin[0]:.1f}, {nav.origin[0] + nav.spans[1]:.1f}]"
              f" z[{nav.origin[1]:.1f}, {nav.origin[1] + nav.spans[0]:.1f}] m")
        print(f"  goal ref : 最近 view point 测地 {nav.goal_ref_distance:.4f} m"
              f"  vs habitat distance_to_goal {nav.start_distance:.4f} m"
              f"  -> {'一致' if abs(nav.goal_ref_distance - nav.start_distance) < 0.01 else '不一致！'}")
        print(f"  path     : " + (f"navmesh 最优路线 {len(nav.path_xz)} 点"
                                 if nav.path_xz else "不可达（目标不在同一 navmesh 小岛）"))
        print(f"  output   : {episode_dir}")
        if not nav.navmesh_ok:
            print("  navmesh 没加载，跳过本集（无 navmesh 就没有 topdown 与可达性）")
            nav.finish("no_navmesh")
            exit_code = 1
            continue

        if args.selfcheck:
            ok, worst, checked = td.selfcheck(nav.env)
            print(f"  selfcheck: world_xz_to_pixel vs maps.to_grid -> "
                  f"{'OK' if ok else 'MISMATCH'} (max diff {worst}px, {checked} points)")
            if not ok:
                exit_code = 1

        for action in script:
            if action == "stop":
                nav.step(HAB_STOP)
                break
            if not nav.step(ACTION_IDS[action]):
                break
            if nav.episode_over:
                break

        record = nav.finish()
        problems = nav.verify_episode()
        if problems:
            exit_code = 1
            print("  verify   : FAILED")
            for problem in problems:
                print(f"    - {problem}")
        else:
            print("  verify   : OK（句柄 / 几何 / 轨迹均属本场景）")
        if record:
            print(f"  steps    : {record['num_steps']}  path {record['path_length_m']:.2f} m"
                  f"  ({record['terminated_by']})")
            print(f"  metrics  : success={_fmt(record.get('success'))}"
                  f"  spl={_fmt(record.get('spl'))}"
                  f"  soft_spl={_fmt(record.get('soft_spl'))}"
                  f"  distance_to_goal={_fmt(record.get('distance_to_goal'))}")
            print(f"  artifacts: rgb.mp4 / topdown.mp4 / trajectory.png / metrics.json\n")

    _print_session_summary(nav)
    return exit_code


def _fmt(value):
    if isinstance(value, (int, float)) and np.isfinite(value):
        return f"{float(value):.4f}"
    return "n/a"


def _print_session_summary(nav):
    if not nav.session_records:
        return
    print("=== session ===")
    for record in nav.session_records:
        print(f"  {record['scene_id']} ep {record['episode_id']:<4} "
              f"{record['object_category']:<14} success={_fmt(record.get('success'))} "
              f"spl={_fmt(record.get('spl'))} steps={record['num_steps']}")
    summary = rec.write_session_summary(nav.out_root, nav.session_records)
    if summary:
        print(f"  summary -> {summary}")


# --------------------------------------------------------------------------- #
# 浏览器交互模式：sim 循环在主线程，HTTP 服务在 daemon 线程
# --------------------------------------------------------------------------- #

def run_web(nav, args):
    """浏览器界面：sim 循环在主线程，HTTP 服务在 daemon 线程。"""
    server = Play2NavServer(host=args.host, port=args.port,
                            jpeg_quality=args.jpeg_quality,
                            display_scale=args.display_scale,
                            topdown_size=args.topdown_video_size,
                            hold_delay=args.hold_delay)
    server.publish_placeholder()
    server.start()
    print(f"[web] 打开 {server.url()}")
    print(f"[web] 监听 {args.host}:{server.port}")
    if args.host in ("0.0.0.0", "::"):
        print("[web] 注意：监听所有网卡，同一网络内任何人都能打开此页并操控机器人（无登录）")
    print(f"[web] w/a/d/q/e 轻触走一步、按住满 {args.hold_delay:.1f}s 转持续移动；"
          f"p 结束并保存，Esc 退出")
    print(f"[web] 画面 第一视角 {args.display_scale:g}x / JPEG q{args.jpeg_quality}，"
          f"俯视图上限 {args.fps} Hz；卡的话调小 --display-scale 或 --jpeg-quality")
    sys.stdout.flush()

    try:
        _web_loop(nav, args, server)
    finally:
        server.stop()
        if not nav.finished:
            nav.finish("quit_early")
        _print_session_summary(nav)


class _TopdownGate:
    """俯视图限速闸门：窗口内的新帧推迟到窗口结束再发，不丢弃。"""

    def __init__(self, interval):
        self.interval = max(float(interval), 0.0)
        self.published_seq = -1
        self.published_at = 0.0

    def due(self, now, frame_seq):
        """这一帧是否该发了：还没发过，且距上次发布已够一个窗口。"""
        return frame_seq != self.published_seq and (now - self.published_at) >= self.interval

    def mark(self, now, frame_seq):
        self.published_seq = frame_seq
        self.published_at = now


def _web_loop(nav, args, server):
    """sim 循环：取意图、发动作、发布快照。不持锁访问 ``env``。"""
    nav.topdown_refresh = args.topdown_refresh
    published = {"rgb": -1, "topdown": -1}

    def push_rgb():
        """编码并发布第一视角；失败则保留旧帧号，下一 tick 再试。"""
        try:
            server.publish_rgb(nav.rgb)
        except Exception as exc:                      # noqa: BLE001
            print(f"[frame] 第一视角编码/发布失败：{type(exc).__name__}: {exc}")
            return
        published["rgb"] = nav.emit_seq

    nav.on_frame = push_rgb
    gate = _TopdownGate(1.0 / max(args.fps, 1))
    last_action = last_key_log = 0.0

    def load(prefix):
        """换一集。失败不抛：服务还活着，页面会提示并让人再点一次。"""
        try:
            nav.start_episode()
        except Exception as exc:
            nav.last_error = f"加载场景失败：{exc}"
            print(f"[{prefix}] {nav.last_error}")
            return False
        print(f"[{prefix}] {nav.scene_id} ep {nav.episode_id} target={nav.category} "
              f"-> {nav.episode_dir}")
        if nav.repeated:
            print(f"[{prefix}] 本集与之前某集重复：--episodes {args.episodes} 集已跑完，"
                  f"迭代器绕回开头了；加大 --episodes 才能继续拿新场景")
        return True

    def on_stop():
        """``p``：真的发出 habitat 的 stop（指标是在 stop 那一刻判定的），然后结算落盘。"""
        if nav.finished:
            return
        if not nav.episode_over:
            nav.step(HAB_STOP)
        record = nav.finish("user_stop")
        if record:
            print(f"[stop] steps={record['num_steps']} "
                  f"success={_fmt(record.get('success'))} "
                  f"spl={_fmt(record.get('spl'))} -> {record['episode_dir']}")

    load("web")
    while True:
        now = time.monotonic()
        server.housekeeping(now)

        pending = server.take_intents()
        if pending["quit"]:
            if not nav.finished:
                nav.finish("quit_early")
            break
        if pending["stop"]:
            on_stop()
        if pending["next"]:
            if not nav.finished:
                nav.finish("quit_early")
            load("next")

        if nav.can_step() and (now - last_action) >= args.action_interval:
            key = server.active_key(now)
            if key:
                last_action = now
                nav.step(ACTION_IDS[KEY_ACTION[key]])
        elif nav.episode_over and not nav.finished:
            nav.finish(nav.terminated_by or "max_steps")

        if args.key_debug and (now - last_key_log) > 0.5:
            last_key_log = now
            print(f"[keys] {server.key_debug(now)}")

        if gate.due(now, nav.frame_seq):
            server.publish_topdown(nav.last_topdown)
            gate.mark(now, nav.frame_seq)
            published["topdown"] = nav.frame_seq
        if published["rgb"] != nav.emit_seq:
            push_rgb()

        server.publish_snapshot(**_snapshot_fields(nav, args, server, published))
        time.sleep(TICK_S)


def _snapshot_fields(nav, args, server=None, published=None):
    """把 navigator 状态打成只含基本类型的 dict，供 ``/state`` 读取。

    必须在 sim 线程调用。

    Returns:
        dict: 快照字段。
    """
    if nav.last_error:
        phase = "error"
    elif not nav.navmesh_ok:
        phase = "no_navmesh"
    elif not nav.finished:
        phase = "navigating"
    else:
        phase = "finished"

    hint = nav.stuck_hint()
    fields = {
        "phase": phase,
        "scene_id": nav.scene_id,
        "episode_id": nav.episode_id,
        "category": nav.category,
        "step_idx": nav.step_idx,
        "max_steps": nav.max_steps,
        "navmesh_ok": bool(nav.navmesh_ok),
        "frame_seq": nav.frame_seq,
        "emit_seq": nav.emit_seq,
        "pub_rgb_seq": (published or {}).get("rgb", -1),
        "pub_topdown_seq": (published or {}).get("topdown", -1),
        "status_lines": _status_lines(nav, args),
        "terminated_by": nav.terminated_by,
        "episode_dir": nav.episode_dir,
        "last_error": nav.last_error,
        "repeated": bool(nav.repeated),
        "stuck_hint": hint,
        "last_action": nav.last_action,
        "last_delta_m": round(nav.last_delta_m, 4),
    }
    if server is not None:
        fields["hold_delay"] = server.hold_delay
        fields["input_live"] = _input_live(nav, server, args)
        fields["input_log"] = server.input_log()
        fields["stream"] = server.stream_stats()
        if bool(hint) != bool(nav.hint_logged):
            if hint:
                server.note(hint, "warn")
            nav.hint_logged = bool(hint)
    return fields


def _input_live(nav, server, args):
    """「输入检测」面板顶上那几行实时状态：按键收到了没有、够不够门槛、上一动结果如何。"""
    held, waiting, elapsed, pending = server.key_status()
    lines = []
    if waiting is not None:
        lines.append(f"按住 {waiting} … {elapsed:.1f}s / {server.hold_delay:.1f}s "
                     f"（轻触那一步已出，到点转持续移动）")
    elif held:
        key = held[0]
        lines.append(f"按住 {key} → {KEY_ACTION.get(key, '?')} 持续输出中")
    elif pending:
        lines.append(f"{pending[0]} 轻触的一步已收下，待发（sim 线程正在执行上一步）")
    else:
        lines.append(f"没有按键按住（轻触 = 一步；按住满 {server.hold_delay:.1f}s = 持续移动）")

    if nav.can_step() and nav.last_action:
        moved = "" if nav.last_action != "move_forward" else f"，位移 {nav.last_delta_m:.3f} m"
        flag = "（有接触）" if nav.last_collided else ""
        lines.append(f"上一步: {nav.last_action}{moved}{flag}")

    hint = nav.stuck_hint()
    if hint:
        lines.append(hint)

    if not nav.can_step():
        lines.append("环境当前不接受动作：" + _why_cannot_step(nav))
    return lines


def _why_cannot_step(nav):
    """「环境为什么不动」的一句话。用户按了键没反应时，这是最想知道的那句。"""
    if nav.last_error:
        return "加载失败，请点「进入下一个场景」重试"
    if not nav.navmesh_ok:
        return "本场景没有 navmesh，无法导航"
    if nav.finished:
        return f"本集已结束（{nav.terminated_by}），点「进入下一个场景」开始下一集"
    if nav.episode_over:
        return "本集已到步数上限"
    return "未知原因"


def _status_lines(nav, args):
    """侧边栏「状态」栏的文本。"""
    if nav.last_error:
        return [nav.last_error, "", "再点一次「进入下一个场景」重试。"]
    if not nav.navmesh_ok:
        return ["navmesh 未加载！", "本集无法导航，", "请点「进入下一个场景」"]
    if nav.can_step():
        lines = ["导航中",
                 f"轻触 w/a/d/q/e 走一步，按住满 {args.hold_delay:.0f}s 转持续移动"
                 if args.hold_delay > 0 else "按住 w/a/d/q/e 移动",
                 "按 p 结束并保存"]
        if args.show_distance:
            m = nav._metrics()
            lines.append(f"distance_to_goal: {_fmt(m.get('distance_to_goal'))}")
        return lines

    record = nav.session_records[-1] if nav.session_records else {}
    return [
        "已结束",
        f"结束原因: {nav.terminated_by}",
        f"success: {_fmt(record.get('success'))}",
        f"spl: {_fmt(record.get('spl'))}",
        f"soft_spl: {_fmt(record.get('soft_spl'))}",
        f"distance_to_goal: {_fmt(record.get('distance_to_goal'))}",
        f"步数: {record.get('num_steps', nav.step_idx)}",
        f"路径长: {_fmt(record.get('path_length_m'))} m",
        "",
        "产物已保存到:",
        os.path.basename(nav.episode_dir or ""),
    ]


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="val", help="HM3D split: train / val / val_mini")
    parser.add_argument("--episodes", type=int, default=20,
                        help="会话里预备多少集（「进入下一个场景」在其中推进）")
    parser.add_argument("--seed", type=int, default=None, help="数据集 seed")
    parser.add_argument("--shuffle", action="store_true",
                        help="打乱 episode 顺序。默认关；shuffle=False 时连着几集常是同一场景")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--out", default=DEFAULT_OUT_ROOT,
                        help="产物根目录；每集一个子目录。默认放 /DATA_HDD，"
                             "因为 / 分区只剩十几 GB，放不下累积的录像")
    parser.add_argument("--no-save", action="store_true", help="不写任何产物")
    parser.add_argument("--success-distance", type=float, default=1.0,
                        help="habitat success 判据（m）。HM3D ObjectNav 标准是 1.0，"
                             "本仓库 YAML 写的是 0.25")
    parser.add_argument("--video-fps", type=int, default=10, help="两个 mp4 的帧率")
    parser.add_argument("--topdown-video-size", type=int, default=512,
                        help="topdown 回放的编码边长（原图约 1024，缩到 512 省 3/4 体积）")
    parser.add_argument("--host", default="0.0.0.0",
                        help="网页监听地址。默认 0.0.0.0，同网络内直接用 IP 访问；"
                             "只想自己用就设 127.0.0.1 再走 SSH 隧道")
    parser.add_argument("--port", type=int, default=8080,
                        help="网页端口；传 0 让内核随机挑一个（自检脚本用的就是这个）")
    parser.add_argument("--jpeg-quality", type=int, default=70,
                        help="网页画面（第一视角与俯视图）的 JPEG 质量")
    parser.add_argument("--fps", type=int, default=5,
                        help="俯视图的推送上限（第一视角每出一帧就推，不受此限）。"
                             "俯视图是一张地图，5 Hz 足够，推快了只是白占带宽")
    parser.add_argument("--action-interval", type=float, default=0.15,
                        help="长按时两个动作之间的最小间隔（秒）")
    parser.add_argument("--hold-delay", type=float, default=1.0,
                        help="键要连续按住这么久（秒）才转为持续移动；轻触只出一步。"
                             "传 0 退回「一按下就持续动」")
    parser.add_argument("--topdown-refresh", type=float, default=0.2,
                        help="俯视图底图（地图+迷雾）的重算间隔（秒）。传 0 表示每步都重算。"
                             "只影响网页模式")
    parser.add_argument("--display-scale", type=float, default=0.5,
                        help="只缩小网页上显示的第一视角画面；录像与俯视图始终用原始分辨率")
    parser.add_argument("--show-distance", action="store_true",
                        help="在侧边栏实时显示 distance_to_goal（默认关：那是评测答案）")
    parser.add_argument("--key-debug", action="store_true", help="打印按键状态，排查长按问题")

    parser.add_argument("--headless", action="store_true",
                        help="不开窗口，用 --scripted 的动作序列跑（自检用）")
    parser.add_argument("--scripted", default=DEFAULT_SCRIPT,
                        help=f"逗号分隔的动作序列，键位或动作名都行。默认 {DEFAULT_SCRIPT}")
    parser.add_argument("--selfcheck", action="store_true",
                        help="headless 下额外校验 world->pixel 换算与 habitat 是否一致")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # SIGTERM 转成 KeyboardInterrupt，以便走到 finally 落盘。Windows 无 SIGTERM。
    if hasattr(signal, "SIGTERM"):
        def _on_sigterm(_signum, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _on_sigterm)

    os.makedirs(args.out, exist_ok=True)
    config, fixed_scene_dataset, yaml_success = play_config(
        stage=args.stage, episodes=args.episodes, seed=args.seed,
        gpu_id=args.gpu_id, success_distance=args.success_distance,
        shuffle=args.shuffle)

    for line in summary_lines(config, args.stage, args.episodes,
                              args.success_distance, fixed_scene_dataset, yaml_success):
        print(line)
    print(f"output root: {args.out}\n")

    problems = check_split(args.stage, config)
    if problems:
        print(f"[错误] --stage {args.stage} 在本机跑不了：")
        for problem in problems:
            print(f"  - {problem}")
        print("\n本机目前只有 val 可用；train / val_mini 的场景路径与 episode 的 "
              "scene_id 对不上（见 config.check_split）。")
        print("请改用：python play2nav/app.py --stage val")
        return 2

    try:
        env = habitat.Env(config)
    except Exception as exc:
        print(f"[错误] 构造 habitat.Env 失败：{exc}")
        print(f"  scenes_dir = {config.habitat.dataset.scenes_dir}")
        print(f"  scene dataset = {config.habitat.simulator.scene_dataset}")
        print("  多半是该 split 的场景文件在本机不存在，试 --stage val。")
        return 2

    nav = Navigator(env, args, args.out,
                    max_episode_steps=config.habitat.environment.max_episode_steps)
    try:
        try:
            if args.headless:
                return run_headless(nav, args)
            run_web(nav, args)
            return 0
        except KeyboardInterrupt:
            print("\n[中断] 收到 Ctrl-C / SIGTERM，正在收尾并退出")
            return 130
    finally:
        nav.close()
        gc.collect()
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
