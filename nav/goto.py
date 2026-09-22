"""像素反投影、occupancy 落脚与离散跟随。

主路径：语义像素 / 前沿 / 回溯都投到自建 2D occupancy，用 A* + TURN/FORWARD。
navmesh greedy follower 仍保留给旧测试。
"""

import math

import habitat_sim
import numpy as np
import quaternion

from nav.geometry import get_pointcloud_from_depth_mask, translate_to_world
from nav.occupancy import (CELL_FREE, CELL_OCC, HAB_FORWARD, HAB_LEFT, HAB_RIGHT,
                           HAB_STOP, body_position, _bresenham)
from nav.transform import as_quaternion

STUCK_EPS_M = 1e-4


def _v3(p):
    """把点转成 ``(3,)`` 的 float32 数组。"""
    return np.asarray(p, dtype=np.float32).reshape(3)


def unproject_pixel(x, y, depth, intrinsic):
    """单个像素 -> 相机系点 ``[x, z, -y]``；该像素深度无效时返回 ``None``。

    Args:
        x (int): 像素列。
        y (int): 像素行。
        depth (np.ndarray): ``(H, W)`` 深度图，0 表示无效。
        intrinsic (np.ndarray): ``3x3`` 相机内参。

    Returns:
        np.ndarray: 相机系点 ``(3,)``；深度无效时为 ``None``。
    """
    d = float(depth[y, x])
    if d <= 0:
        return None
    px = (x - intrinsic[0][2]) * d / intrinsic[0][0]
    pz = (depth.shape[0] - 1 - y - intrinsic[1][2]) * d / intrinsic[1][1]
    return np.array([px, pz, -d], dtype=np.float64)


def pixel_to_world(u, v, depth, intrinsic, sensor_position, sensor_rotation):
    """像素反投影到 habitat 世界坐标。

    Args:
        u (int): 列。
        v (int): 行。
        depth (np.ndarray): ``(H, W)`` 深度，0 无效。
        intrinsic (np.ndarray): ``3x3`` 内参。
        sensor_position: 相机世界坐标。
        sensor_rotation: 相机旋转。

    Returns:
        np.ndarray: 世界坐标 ``(3,)``；深度无效时为 ``None``。
    """
    cam_pt = unproject_pixel(int(u), int(v), depth, intrinsic)
    if cam_pt is None:
        return None
    rot = quaternion.as_rotation_matrix(as_quaternion(sensor_rotation))
    pos = np.asarray(sensor_position, dtype=np.float64).reshape(3)
    return translate_to_world(cam_pt[None, :], pos, rot)[0]


def goal_from_mask(depth, mask, intrinsic, pos_local, rot_local, center_px):
    """掩码中心像素 -> mapper 局部系 3D 点 ``(x, z, y)``。

    Args:
        depth (np.ndarray): ``(H, W)`` 深度图，0 表示无效。
        mask (np.ndarray): 布尔掩码，``(H, W)``。
        intrinsic (np.ndarray): ``3x3`` 相机内参。
        pos_local: 相机位置，mapper 局部系。
        rot_local (np.ndarray): 相机旋转矩阵。
        center_px (tuple): 掩码中心像素 ``(x, y)``。

    Returns:
        np.ndarray: mapper 局部系 3D 点；无有效深度时返回 ``None``。
    """
    cam_pt = unproject_pixel(center_px[0], center_px[1], depth, intrinsic)
    if cam_pt is not None:
        return translate_to_world(cam_pt[None, :], pos_local, rot_local)[0]
    cam_pts = get_pointcloud_from_depth_mask(depth, mask, intrinsic)
    if len(cam_pts) == 0:
        return None
    return np.median(translate_to_world(cam_pts, pos_local, rot_local), axis=0)


def goal_world_from_mask(env, depth, mask, intrinsic, center_px):
    """掩码中心 -> habitat 世界坐标（走相机位姿，不经 mapper 原点）。

    Args:
        env: ``habitat.Env``。
        depth (np.ndarray): ``(H, W)`` 深度。
        mask (np.ndarray): 布尔掩码。
        intrinsic (np.ndarray): 内参。
        center_px (tuple): ``(u, v)``。

    Returns:
        np.ndarray: 世界坐标；失败为 ``None``。
    """
    from nav.occupancy import sensor_pose

    pos, rot = sensor_pose(env)
    world = pixel_to_world(center_px[0], center_px[1], depth, intrinsic, pos, rot)
    if world is not None:
        return world
    cam_rot = quaternion.as_rotation_matrix(as_quaternion(rot))
    cam_pts = get_pointcloud_from_depth_mask(depth, mask, intrinsic)
    if len(cam_pts) == 0:
        return None
    return np.median(translate_to_world(cam_pts, pos, cam_rot), axis=0)


def geodesic_path(pf, start_world, end_world):
    """navmesh 上从起点到终点的测地路径。

    Returns:
        tuple: ``(ShortestPath, ok)``。
    """
    sp = habitat_sim.ShortestPath()
    sp.requested_start = _v3(start_world)
    sp.requested_end = _v3(end_world)
    return sp, bool(pf.find_path(sp))


def detour_reject(euc, geo, max_detour_m=3.0, max_detour_ratio=4.0, detour_floor=0.5,
                  dy=0.0):
    """判断绕行是否离谱。是则返回理由，正常返回 ``None``。

    Args:
        euc (float): 直线距离，米。
        geo (float): 测地距离，米。
        max_detour_m (float): 近距区间允许的最大绝对绕路量。
        max_detour_ratio (float): 近距之外允许的最大测地/直线比。
        detour_floor (float): 划分两个区间的直线距离。
        dy (float): 高差，米。

    Returns:
        str: 拒绝理由；通过时 ``None``。
    """
    floor = f"，高差 {dy:.2f} m —— 目标多半在楼上/楼下" if abs(dy) > 1.5 else ""
    if euc < detour_floor:
        if geo - euc > max_detour_m:
            return (f"就在 {euc:.2f} m 外却要绕 {geo - euc:.2f} m（测地 {geo:.2f} m）"
                    f"—— 多半吸附到了墙的另一侧{floor}")
    elif geo / max(euc, 1e-6) > max_detour_ratio:
        return (f"绕行比 {geo / max(euc, 1e-6):.2f} > {max_detour_ratio:.1f}"
                f"（直线 {euc:.2f} m，测地 {geo:.2f} m{floor}）")
    return None


def farthest_reachable_along(pf, start_world, goal_world, n=32, detour_ok=None):
    """沿 ``agent -> goal`` 连线找最远可达点。

    Args:
        pf: pathfinder。
        start_world: agent 位置。
        goal_world: 目标位置。
        n (int): 采样点数。
        detour_ok: ``(euc, geo, dy) -> bool``。

    Returns:
        tuple: ``(point, t, path)``；找不到则为 ``(None, 0.0, None)``。
    """
    start = _v3(start_world).astype(np.float64)
    goal = _v3(goal_world).astype(np.float64)
    for t in np.linspace(1.0, 0.0, n):
        p = start + t * (goal - start)
        s = np.asarray(pf.snap_point(p), dtype=np.float64)
        if not bool(pf.is_navigable(s)):
            continue
        sp, ok = geodesic_path(pf, start, s)
        if not ok:
            continue
        geo = float(sp.geodesic_distance)
        euc = float(np.linalg.norm(s - start))
        if detour_ok is not None and not detour_ok(euc, geo, abs(float(s[1] - start[1]))):
            continue
        return s, float(t), sp
    return None, 0.0, None


def snap_goal_to_navmesh(pf, goal_world, start_world, max_offset=2.5,
                         max_detour_m=3.0, max_detour_ratio=4.0, detour_floor=0.5,
                         guard=True):
    """把目标点吸附到 navmesh，并拒绝跨墙吸附。

    Args:
        pf: pathfinder。
        goal_world: 目标世界坐标。
        start_world: agent 世界坐标。
        max_offset (float): 最大吸附位移。
        max_detour_m (float): 近距最大绕路量。
        max_detour_ratio (float): 最大绕行比。
        detour_floor (float): 近距阈值。
        guard (bool): 是否否决。

    Returns:
        tuple: ``(snapped_world, info)``。
    """
    goal_world = _v3(goal_world)
    start_world = _v3(start_world)
    snapped = pf.snap_point(goal_world)
    offset = float(np.linalg.norm(np.asarray(snapped, dtype=np.float64) - goal_world))

    info = {"offset": offset, "start_offset": None, "geodesic": None, "euclid": None,
            "excess": None, "ratio": None, "reject": None, "points": None,
            "fallback": False, "advance": 1.0}

    if not bool(pf.is_navigable(snapped)):
        info["reject"] = "吸附点不在 navmesh 上"
        if guard:
            return None, info

    if guard and offset > max_offset:
        info["reject"] = f"吸附位移 {offset:.2f} m > {max_offset:.1f} m（目标离可行走面太远）"
        return None, info

    start_snapped = pf.snap_point(start_world)
    info["start_offset"] = float(
        np.linalg.norm(np.asarray(start_snapped, dtype=np.float64) - start_world))

    sp, ok = geodesic_path(pf, start_snapped, snapped)
    if not ok:
        info["reject"] = "与 agent 不在同一连通片（被墙或家具隔开；HM3D 的 navmesh 分片成多个小岛）"
        detour_ok = None
        if guard:
            detour_ok = lambda e, g, dy: detour_reject(  # noqa: E731
                e, g, max_detour_m, max_detour_ratio, detour_floor, dy) is None
        fb, t, sp_fb = farthest_reachable_along(pf, start_snapped, goal_world,
                                                detour_ok=detour_ok)
        if fb is None:
            if guard:
                info["reject"] += "；连线上也没有既可达又不绕远的点"
            return (None, info) if guard else (np.asarray(snapped, dtype=np.float64), info)
        info["fallback"] = True
        info["advance"] = t
        snapped, sp = fb, sp_fb

    geo = float(sp.geodesic_distance)
    euc = float(np.linalg.norm(np.asarray(snapped, dtype=np.float64) - start_snapped))
    info.update(geodesic=geo, euclid=euc, excess=geo - euc,
                ratio=(geo / euc if euc > 1e-6 else float("inf")),
                points=np.asarray(sp.points, dtype=np.float64) if len(sp.points) else None)

    if not guard or info["fallback"]:
        return np.asarray(snapped, dtype=np.float64), info

    reason = detour_reject(euc, geo, max_detour_m, max_detour_ratio, detour_floor,
                           dy=abs(float(snapped[1] - start_snapped[1])))
    if reason is not None:
        info["reject"] = reason
        return None, info
    return np.asarray(snapped, dtype=np.float64), info


def oracle_goal_world(env):
    """episode 标注里测地最近的 view point。

    Args:
        env: ``habitat.Env``。

    Returns:
        np.ndarray: 世界坐标；没有标注时 ``None``。
    """
    vps = [np.asarray(vp.agent_state.position, dtype=np.float64)
           for goal in env.current_episode.goals
           for vp in (getattr(goal, "view_points", None) or [])]
    if not vps:
        vps = [np.asarray(g.position, dtype=np.float64) for g in env.current_episode.goals]
    if not vps:
        return None

    agent = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
    dists = [float(env.sim.geodesic_distance(agent, p)) for p in vps]
    return vps[int(np.argmin(dists))]


def make_follower(env, goal_radius):
    """构造贪心测地跟随器，``stop_key`` 显式为 stop。

    Args:
        env: habitat 环境。
        goal_radius (float): 到达半径，米。

    Returns:
        GreedyFollower: 跟随器。
    """
    return env.sim.make_greedy_follower(goal_radius=goal_radius, stop_key=HAB_STOP)


def pursue(env, follower, goal_world, max_steps, stop_at_goal=True):
    """朝 ``goal_world`` 走最多 ``max_steps`` 步。

    Args:
        env: habitat 环境。
        follower: ``make_follower`` 的返回值。
        goal_world: 目标世界坐标。
        max_steps (int): 最多步数。
        stop_at_goal (bool): 到达时是否发 stop。

    Returns:
        tuple: ``(steps, arrived)``。
    """
    result = pursue_leg(env, follower, goal_world, max_steps, stop_at_goal=stop_at_goal,
                        stop_on_stuck=False)
    return result["steps"], result["arrived"]


def pursue_leg(env, follower, goal_world, max_steps, stop_at_goal=False, on_step=None,
               stop_on_stuck=True):
    """``pursue`` 的详细版，带卡住与位移。

    Args:
        env: habitat 环境。
        follower: 跟随器。
        goal_world: 目标世界坐标。
        max_steps (int): 最多步数。
        stop_at_goal (bool): 到达时是否发 stop。
        on_step: 每步之后的回调 ``(env,)``。

    Returns:
        dict: ``steps`` / ``arrived`` / ``blocked`` / ``dist_moved_m``。
    """
    goal = _v3(goal_world)
    dist_moved = 0.0
    for i in range(max_steps):
        try:
            action = follower.next_action_along(goal)
        except habitat_sim.errors.GreedyFollowerError:
            print("  跟随器无法从当前位置规划到目标（GreedyFollowerError）")
            return {"steps": i, "arrived": False, "blocked": True, "dist_moved_m": dist_moved}
        if action is None or int(action) == HAB_STOP:
            if not stop_at_goal:
                return {"steps": i, "arrived": True, "blocked": False,
                        "dist_moved_m": dist_moved}
            env.step(HAB_STOP)
            if on_step is not None:
                on_step(env)
            return {"steps": i + 1, "arrived": True, "blocked": False,
                    "dist_moved_m": dist_moved}
        before = body_position(env)
        env.step(int(action))
        after = body_position(env)
        step_d = float(np.linalg.norm(after - before))
        dist_moved += step_d
        if on_step is not None:
            on_step(env)
        if env.episode_over:
            return {"steps": i + 1, "arrived": False, "blocked": False,
                    "dist_moved_m": dist_moved}
        if stop_on_stuck and int(action) == HAB_FORWARD and step_d < STUCK_EPS_M:
            return {"steps": i + 1, "arrived": False, "blocked": True,
                    "dist_moved_m": dist_moved}
    return {"steps": max_steps, "arrived": False, "blocked": False,
            "dist_moved_m": dist_moved}


def body_yaw(rotation):
    """机身朝向 yaw（弧度），XZ 平面，``atan2(forward_x, forward_z)``。

    Args:
        rotation: 机身四元数或 ``[x,y,z,w]``。

    Returns:
        float: yaw 弧度。
    """
    rot = quaternion.as_rotation_matrix(as_quaternion(rotation))
    fwd = -rot[:, 2]
    return float(math.atan2(fwd[0], fwd[2]))


def body_yaw_env(env):
    """当前机身 yaw。

    Args:
        env: ``habitat.Env``。

    Returns:
        float: 弧度。
    """
    return body_yaw(env.sim.get_agent_state().rotation)


def wrap_pi(angle):
    """把角度折到 ``(-pi, pi]``。"""
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    return a if a > -math.pi else a + 2 * math.pi


def turn_to_yaw(env, target_yaw, max_turns=12):
    """原地转到目标 yaw（每步 30°）。

    Args:
        env: ``habitat.Env``。
        target_yaw (float): 目标朝向，弧度。
        max_turns (int): 最多转几次。

    Returns:
        int: 实际转动次数。
    """
    n = 0
    while n < max_turns and not env.episode_over:
        cur = body_yaw_env(env)
        delta = wrap_pi(target_yaw - cur)
        if abs(delta) < math.radians(15.0):
            break
        env.step(HAB_RIGHT if delta < 0 else HAB_LEFT)
        n += 1
    return n


def face_pano(env, pano_id, node_yaw=None):
    """对准节点坐标系下的偶序号朝向。``pano_id`` 每档 30°。

    Args:
        env: ``habitat.Env``。
        pano_id (int): ``0,2,...,10``。
        node_yaw (float, optional): 扫描时的 yaw；缺省用当前 yaw 当 0 号。

    Returns:
        int: 转动次数。
    """
    base = body_yaw_env(env) if node_yaw is None else float(node_yaw)
    target = base - math.radians(30.0 * int(pano_id))
    return turn_to_yaw(env, target)


def occupancy_path_length(occ, a, b, success_dist=0.35):
    """两点 occupancy A* 路径长；不通为 ``inf``。"""
    from nav.astar2d import astar_or_relax

    path, length, _ = astar_or_relax(occ, a, b, success_dist=success_dist)
    if path is None or not math.isfinite(length):
        return float("inf")
    return float(length)


def _foothold_cell_blocked(occ, row, col):
    """OCC 或 inflate 视为挡住。"""
    h, w = occ.grid.shape
    if row < 0 or col < 0 or row >= h or col >= w:
        return True
    if int(occ.grid[row, col]) == CELL_OCC:
        return True
    return occ.inflate is not None and occ.inflate[row, col] != 0


def _foothold_cell_free(occ, row, col):
    """可站 FREE：本身 FREE 且不在 inflate 上。"""
    h, w = occ.grid.shape
    if row < 0 or col < 0 or row >= h or col >= w:
        return False
    if occ.inflate is not None and occ.inflate[row, col] != 0:
        return False
    return int(occ.grid[row, col]) == CELL_FREE


def foothold_from_hit(occ, cam_xyz, hit_xyz, agent_xyz, min_stand=0.35):
    """语义命中点 -> occupancy 上可走的 A* 终点。

    沿命中点→相机的视线，离开 OCC/inflate 后取第一格 FREE；失败再以命中点
    为圆心邻域搜索，最后沿连线回退。
    """
    from nav.astar2d import astar_or_relax, nearest_walkable, point_walkable, SAFETY_OPTIMISTIC

    cam = np.asarray(cam_xyz, dtype=np.float64).reshape(3)
    hit = np.asarray(hit_xyz, dtype=np.float64).reshape(3)
    agent = np.asarray(agent_xyz, dtype=np.float64).reshape(3)
    y = float(agent[1])
    if occ.grid is None:
        return None
    occ._expand_grid(np.array([cam[0], hit[0], agent[0]]),
                     np.array([cam[2], hit[2], agent[2]]), pad_m=1.0)

    def accept(stand):
        """距离足够且 A* 可达则作为落脚。"""
        dist_a = float(np.hypot(stand[0] - agent[0], stand[2] - agent[2]))
        if dist_a >= min_stand:
            path, _, _ = astar_or_relax(occ, agent, stand, success_dist=0.30)
            return path is not None
        hit_d = float(np.hypot(hit[0] - agent[0], hit[2] - agent[2]))
        return hit_d < min_stand + 0.2

    ac = occ.world_to_cell(cam[0], cam[2])
    hc = occ.world_to_cell(hit[0], hit[2])
    if ac is not None and hc is not None:
        # 命中点 → 相机：先走出 OCC/inflate，再取第一格 FREE。
        for r, c in _bresenham(hc[0], hc[1], ac[0], ac[1]):
            if _foothold_cell_blocked(occ, r, c):
                continue
            if not _foothold_cell_free(occ, r, c):
                continue
            stand = occ.cell_to_world(r, c, y=y)
            if accept(stand):
                return stand
            break
    for radius in (0.50, 0.70, 0.85):
        cand = nearest_walkable(occ, hit[0], hit[2], y=y, max_r=radius,
                                safety_mode=SAFETY_OPTIMISTIC)
        if cand is None:
            continue
        path, _, _ = astar_or_relax(occ, agent, cand, success_dist=0.30)
        if path is not None:
            return cand
    start = agent.astype(np.float64)
    goal = hit.astype(np.float64)
    for t in np.linspace(1.0, 0.15, 24):
        p = start + t * (goal - start)
        if not point_walkable(occ, p[0], p[2], SAFETY_OPTIMISTIC):
            continue
        path, _, _ = astar_or_relax(occ, agent, p, success_dist=0.30)
        if path is not None:
            return p
    return None


def project_pixel_to_occupancy(env, occ, u, v, depth, intrinsic, mask=None):
    """像素 -> occupancy 落脚点。深度无效时可用 mask 点云中位数。"""
    from nav.occupancy import sensor_pose

    pos, rot = sensor_pose(env)
    world = pixel_to_world(int(u), int(v), depth, intrinsic, pos, rot)
    if world is None and mask is not None:
        world = goal_world_from_mask(env, depth, mask, intrinsic, (int(u), int(v)))
    if world is None:
        return None, None
    stand = foothold_from_hit(occ, pos, world, body_position(env))
    return stand, world


LOOKAHEAD_M = 0.80
ACTION_ANGLE = math.pi / 6.0
YAW_TURN_RAD = ACTION_ANGLE / 1.9
STUCKING_DISTANCE = 0.05
FORWARD_MARK_M = 0.15
TARGET_WEIGHT = 150.0
TARGET_CLOSE_W1 = 2000.0
TARGET_CLOSE_W2 = 200.0
# ApexNav 脱困：右转/前进交替，避免 A* 立刻把机头拧回墙。
ESCAPE_ACTIONS = (
    HAB_RIGHT, HAB_FORWARD, HAB_RIGHT, HAB_FORWARD,
    HAB_LEFT, HAB_LEFT, HAB_LEFT, HAB_FORWARD,
    HAB_LEFT, HAB_FORWARD,
)


def _select_local_xz(cur_xz, path_xz, lookahead=LOOKAHEAD_M):
    """路径上前瞻约 0.80 m。"""
    if not path_xz:
        return cur_xz
    cx, cz = cur_xz
    best_i, best_d = 0, float("inf")
    for i, (x, z) in enumerate(path_xz):
        d = math.hypot(x - cx, z - cz)
        if d < best_d:
            best_d, best_i = d, i
    start = min(best_i + 1, len(path_xz) - 1)
    acc = math.hypot(path_xz[start][0] - cx, path_xz[start][1] - cz)
    target = path_xz[start]
    for i in range(start + 1, len(path_xz)):
        acc += math.hypot(path_xz[i][0] - path_xz[i - 1][0],
                          path_xz[i][1] - path_xz[i - 1][1])
        if acc > lookahead and math.hypot(cx - path_xz[i - 1][0], cz - path_xz[i - 1][1]) > 0.30:
            target = path_xz[i - 1]
            break
        target = path_xz[i]
    return target


def _best_step_xz(occ, pos, target_xz):
    """12 向 0.25 m 里选缩短到局部目标最多的可走步（ApexNav ``computeBestStep``）。"""
    from nav.astar2d import STEPS, point_walkable, SAFETY_OPTIMISTIC

    gx, gz = float(target_xz[0]), float(target_xz[1])
    px, pz = float(pos[0]), float(pos[2])
    cur_d = math.hypot(gx - px, gz - pz)
    best, best_cost = None, float("inf")
    for dx, dz in STEPS:
        nx, nz = px + dx, pz + dz
        if not point_walkable(occ, nx, nz, SAFETY_OPTIMISTIC):
            continue
        d = math.hypot(gx - nx, gz - nz)
        close = d - cur_d
        cost = TARGET_WEIGHT * d + (TARGET_CLOSE_W1 if close > 0 else TARGET_CLOSE_W2) * close
        if cost < best_cost:
            best_cost = cost
            best = (nx, nz)
    return best


def _action_to_xz(pos, yaw, target_xz):
    """对准 ``target_xz``：偏航过大则转，否则前进。"""
    target_yaw = math.atan2(target_xz[0] - pos[0], target_xz[1] - pos[2])
    delta = wrap_pi(target_yaw - yaw)
    if abs(delta) > YAW_TURN_RAD:
        return HAB_RIGHT if delta < 0 else HAB_LEFT
    return HAB_FORWARD


def _greedy_progress_action(occ, env, goal_xyz):
    """A* 失败时：朝能缩短距离的 0.25 m 步走或转。"""
    pos = body_position(env)
    yaw = body_yaw_env(env)
    best = _best_step_xz(occ, pos, (float(goal_xyz[0]), float(goal_xyz[2])))
    if best is None:
        target_yaw = math.atan2(float(goal_xyz[0]) - pos[0], float(goal_xyz[2]) - pos[2])
        delta = wrap_pi(target_yaw - yaw)
        return HAB_RIGHT if delta < 0 else HAB_LEFT
    return _action_to_xz(pos, yaw, best)


def _mark_collision_ahead(occ, pos, yaw):
    """脱困失败后，沿机头把 0.15 / 0.30 m 强制标 OCC。"""
    sx, sz = float(pos[0]), float(pos[2])
    occ.force_occ_xz(sx, sz)
    for dist in (FORWARD_MARK_M, FORWARD_MARK_M * 2.0):
        occ.force_occ_xz(sx + dist * math.sin(yaw), sz + dist * math.cos(yaw))


def pursue_occupancy(env, occ, goal_world, max_steps, on_step=None, stop_on_stuck=True,
                     success_dist=0.25):
    """在自建 occupancy 上离散跟随：每步 A* 前瞻，再 TURN 或 FORWARD 0.25 m。

    第一步搜不到路径则立即返回，不空转仿真步。
    """
    from nav.astar2d import astar_or_relax, point_walkable, SAFETY_OPTIMISTIC

    goal = np.asarray(goal_world, dtype=np.float64).reshape(3)
    dist_moved = 0.0
    last_path = None
    n_forward, n_turn = 0, 0
    escape_i = -1
    escape_yaw = 0.0
    escape_rounds = 0
    last_pos = body_position(env)
    last_action = HAB_STOP
    for i in range(max_steps):
        pos = body_position(env)
        yaw = body_yaw_env(env)
        if math.hypot(pos[0] - goal[0], pos[2] - goal[2]) <= success_dist:
            return {"steps": i, "arrived": True, "blocked": False, "dist_moved_m": dist_moved,
                    "path_xz": last_path, "n_forward": n_forward, "n_turn": n_turn}
        if (escape_i < 0 and last_action == HAB_FORWARD
                and float(np.hypot(pos[0] - last_pos[0], pos[2] - last_pos[2])) < STUCKING_DISTANCE):
            escape_i = 0
            escape_yaw = yaw
        if escape_i >= 0:
            if escape_i >= len(ESCAPE_ACTIONS):
                _mark_collision_ahead(occ, pos, escape_yaw)
                escape_i = -1
                escape_rounds += 1
                if stop_on_stuck and escape_rounds >= 2:
                    return {"steps": i, "arrived": False, "blocked": True,
                            "dist_moved_m": dist_moved, "path_xz": last_path,
                            "n_forward": n_forward, "n_turn": n_turn}
                action = _greedy_progress_action(occ, env, goal)
            else:
                action = ESCAPE_ACTIONS[escape_i]
                escape_i += 1
        else:
            path, _, _ = astar_or_relax(occ, pos, goal, success_dist=success_dist)
            last_path = path
            if path is None or len(path) < 2:
                if i == 0:
                    return {"steps": 0, "arrived": False, "blocked": False,
                            "dist_moved_m": dist_moved, "path_xz": last_path,
                            "n_forward": n_forward, "n_turn": n_turn}
                action = _greedy_progress_action(occ, env, goal)
            else:
                lx, lz = _select_local_xz((pos[0], pos[2]), path)
                best = _best_step_xz(occ, pos, (lx, lz))
                if best is None:
                    action = _action_to_xz(pos, yaw, (lx, lz))
                    if action == HAB_FORWARD:
                        nx = pos[0] + 0.25 * math.sin(yaw)
                        nz = pos[2] + 0.25 * math.cos(yaw)
                        if not point_walkable(occ, nx, nz, SAFETY_OPTIMISTIC):
                            action = _greedy_progress_action(occ, env, goal)
                else:
                    action = _action_to_xz(pos, yaw, best)
                    if action == HAB_FORWARD and not point_walkable(
                            occ, best[0], best[1], SAFETY_OPTIMISTIC):
                        action = _greedy_progress_action(occ, env, goal)
        last_pos = body_position(env)
        last_action = int(action)
        env.step(last_action)
        after = body_position(env)
        if last_action == HAB_FORWARD:
            n_forward += 1
        else:
            n_turn += 1
        step_d = float(np.hypot(after[0] - last_pos[0], after[2] - last_pos[2]))
        dist_moved += step_d
        if escape_i >= 0 and step_d >= STUCKING_DISTANCE:
            escape_i = -1
        occ.record_body(after)
        if on_step is not None:
            on_step(env)
        if env.episode_over:
            return {"steps": i + 1, "arrived": False, "blocked": False,
                    "dist_moved_m": dist_moved, "path_xz": last_path,
                    "n_forward": n_forward, "n_turn": n_turn}
    return {"steps": max_steps, "arrived": False, "blocked": False,
            "dist_moved_m": dist_moved, "path_xz": last_path,
            "n_forward": n_forward, "n_turn": n_turn}


def geodesic_m(env, a, b):
    """两点测地距离；不通则为 ``inf``。

    Args:
        env: ``habitat.Env``。
        a: 起点。
        b: 终点。

    Returns:
        float: 米。
    """
    d = float(env.sim.geodesic_distance(_v3(a), _v3(b)))
    if not math.isfinite(d) or d < 0:
        return float("inf")
    return d


def project_world_to_uv(xyz, intrinsic, sensor_position, sensor_rotation, image_hw):
    """世界点投到当前相机像素。在相机后方或不在画幅内时返回 ``None``。

    Args:
        xyz: 世界坐标。
        intrinsic (np.ndarray): 内参。
        sensor_position: 相机位置。
        sensor_rotation: 相机旋转。
        image_hw (tuple): ``(H, W)``。

    Returns:
        tuple: ``(u, v, depth_m)`` 或 ``None``。
    """
    p = np.asarray(xyz, dtype=np.float64).reshape(3)
    t = np.asarray(sensor_position, dtype=np.float64).reshape(3)
    r = quaternion.as_rotation_matrix(as_quaternion(sensor_rotation))
    cam = r.T.dot(p - t)
    depth = -float(cam[2])
    if depth <= 0.05:
        return None
    h, w = int(image_hw[0]), int(image_hw[1])
    u = cam[0] * intrinsic[0][0] / depth + intrinsic[0][2]
    v = (h - 1) - (cam[1] * intrinsic[1][1] / depth + intrinsic[1][2])
    if not (0 <= u < w and 0 <= v < h):
        return None
    return int(round(u)), int(round(v)), depth
