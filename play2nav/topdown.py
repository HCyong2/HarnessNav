"""HM3D navmesh 俯视地图：世界坐标换算与轨迹叠加。"""

import cv2
import numpy as np
from habitat.utils.visualizations.maps import colorize_topdown_map, draw_agent, to_grid
from habitat_sim import ShortestPath

TRAJ_COLOR = (255, 0, 255)          # 品红：走过的轨迹
START_COLOR = (0, 220, 0)           # 绿色：起点
PATH_COLOR = (0, 200, 255)          # 橙黄：navmesh 测地路径

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def topdown_geometry(env):
    """从 pathfinder 边界取出地图原点与行列跨度。

    行对应世界 z，列对应世界 x。换场景后必须重新调用。

    Args:
        env: habitat 环境。

    Returns:
        tuple: ``(origin_xz, (row_span_m, col_span_m))``。
    """
    lower_bound, upper_bound = env.sim.pathfinder.get_bounds()
    origin = (float(lower_bound[0]), float(lower_bound[2]))
    spans = (abs(float(upper_bound[2]) - float(lower_bound[2])),
             abs(float(upper_bound[0]) - float(lower_bound[0])))
    return origin, spans


def world_xz_to_pixel(point_xz, origin, spans, shape):
    """把 habitat 世界 ``(x, z)`` 换成俯视地图 ``(row, col)``。

    Args:
        point_xz: 世界坐标 ``(x, z)``。
        origin: 地图原点 ``(x, z)``。
        spans: ``(行跨度, 列跨度)``，米。
        shape: 地图 ``(H, W)``。

    Returns:
        tuple: 整数 ``(row, col)``。
    """
    x, z = float(point_xz[0]), float(point_xz[1])
    return (int(round((z - origin[1]) * shape[0] / spans[0])),
            int(round((x - origin[0]) * shape[1] / spans[1])))


def scene_bounds_ok(pf, points, margin=1.0):
    """判断世界 ``(x, z)`` 点是否都落在当前 navmesh 边界内。

    Args:
        pf: pathfinder。
        points: ``(x, z)`` 列表。
        margin (float): 边界外放宽，米。

    Returns:
        bool: 全部在界内为 True。
    """
    lower, upper = pf.get_bounds()
    for (x, z) in points:
        if not (lower[0] - margin <= x <= upper[0] + margin
                and lower[2] - margin <= z <= upper[2] + margin):
            return False
    return True


def closest_view_point(env, agent_xyz, view_points, episode):
    """在标注 view point 中取离 agent 测地距离最近的一个。

    使用 ``geodesic_distance(agent, [p], episode)`` 三参数接口，与官方
    ``distance_to_goal`` 口径一致。

    Args:
        env: habitat 环境。
        agent_xyz: agent 世界坐标。
        view_points: episode 标注的视点列表。
        episode: 当前 episode。

    Returns:
        tuple: ``(世界坐标, 测地距离)``；算不出则为 ``(None, nan)``。
    """
    agent = np.asarray(agent_xyz, dtype=np.float32).reshape(3)
    best, best_distance = None, float("inf")
    for view_point in view_points:
        position = np.asarray(view_point.agent_state.position, dtype=np.float32).reshape(3)
        try:
            distance = float(env.sim.geodesic_distance(agent, [position], episode))
        except Exception:
            continue
        if np.isfinite(distance) and distance < best_distance:
            best, best_distance = position, distance
    if best is None:
        return None, float("nan")
    return best, best_distance


def navmesh_path_xz(pf, start_xyz, goal_xyz, snap_goal=True):
    """在 navmesh 上求测地路径，返回 ``[(x, z), ...]``。

    终点不在可行走面上时先 ``snap_point`` 再求路。跨岛不可达则返回 ``None``。

    Args:
        pf: pathfinder。
        start_xyz: 起点世界坐标。
        goal_xyz: 终点世界坐标。
        snap_goal (bool): 失败时是否吸附后再试。

    Returns:
        list: 路径点；走不通则为 ``None``。
    """
    start = np.asarray(start_xyz, dtype=np.float32).reshape(3)

    def attempt(end):
        sp = ShortestPath()
        sp.requested_start = start
        sp.requested_end = np.asarray(end, dtype=np.float32).reshape(3)
        if pf.find_path(sp) and sp.points:
            return [(float(p[0]), float(p[2])) for p in sp.points]
        return None

    try:
        path = attempt(goal_xyz)
        if path is None and snap_goal:
            snapped = np.asarray(pf.snap_point(np.asarray(
                goal_xyz, dtype=np.float32).reshape(3)), dtype=np.float32)
            if np.isfinite(snapped).all() and not np.allclose(
                    snapped, goal_xyz, atol=1e-4):
                path = attempt(snapped)
        return path
    except Exception:
        return None


def topdown_info(env):
    """读取 habitat 原生 topdown measurement（未上色）。

    Args:
        env: habitat 环境。

    Returns:
        dict: 含 ``map`` / ``fog_of_war_mask`` / ``agent_map_coord`` 等。

    Raises:
        RuntimeError: 配置里没有 top_down_map。
    """
    info = env.get_metrics().get("top_down_map")
    if info is None:
        raise RuntimeError("top_down_map measurement 不在 config 里，无法绘制俯视地图")
    return info


def colorize_topdown(info):
    """把原始 topdown 图上色成 RGB 画布。

    Args:
        info (dict): :func:`topdown_info` 的返回值。

    Returns:
        np.ndarray: RGB 画布。
    """
    return colorize_topdown_map(info["map"], info["fog_of_war_mask"])


def build_topdown_image(env):
    """取 info 并上色，返回 ``(canvas_rgb, info)``。

    Args:
        env: habitat 环境。

    Returns:
        tuple: 上色画布与原始 info。
    """
    info = topdown_info(env)
    return colorize_topdown(info), info


def draw_agents(canvas, info):
    """在俯视图上画 agent 图标，半径按图幅缩放。

    Args:
        canvas (np.ndarray): RGB 画布，原地修改。
        info (dict): topdown info。

    Returns:
        np.ndarray: 同一张画布。
    """
    radius = max(min(canvas.shape[:2]) // 32, 4)
    for i in range(len(info["agent_map_coord"])):
        draw_agent(canvas, tuple(info["agent_map_coord"][i]),
                   float(info["agent_angle"][i]), radius)
    return canvas


def draw_trajectory(canvas, origin, spans, trajectory_xz, start_xz=None, path_xz=None,
                    with_legend=True):
    """叠加走过的轨迹、测地路径与起点。

    Args:
        canvas (np.ndarray): RGB 画布。
        origin: 地图原点 ``(x, z)``。
        spans: 行列跨度，米。
        trajectory_xz: 历史轨迹 ``(x, z)`` 列表。
        start_xz: 起点；``None`` 则不画。
        path_xz: 测地路径；``None`` 则不画。
        with_legend (bool): 是否在底部加图例。

    Returns:
        np.ndarray: 画好的图（可能拼了图例行）。
    """
    h, w = canvas.shape[:2]
    unit = max(min(h, w), 1)
    start_radius = max(unit // 70, 5)

    def to_px(point_xz):
        if point_xz is None:
            return None
        rc = world_xz_to_pixel(point_xz, origin, spans, (h, w))
        return rc if (0 <= rc[0] < h and 0 <= rc[1] < w) else None

    def polyline(points_xz, color, thickness=2):
        px = [p for p in (to_px(p) for p in points_xz) if p is not None]
        for a, b in zip(px[:-1], px[1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 4 * max(h, w):
                continue
            cv2.line(canvas, (a[1], a[0]), (b[1], b[0]), color, thickness, cv2.LINE_AA)

    if path_xz:
        polyline(path_xz, PATH_COLOR)
    polyline(trajectory_xz, TRAJ_COLOR)

    start_px = to_px(start_xz)
    if start_px is not None:
        cv2.circle(canvas, (start_px[1], start_px[0]), start_radius, START_COLOR, -1, cv2.LINE_AA)

    if not with_legend:
        return canvas

    entries = [("agent path", TRAJ_COLOR)]
    if path_xz:
        entries.append(("navmesh shortest path", PATH_COLOR))
    if start_px is not None:
        entries.append(("start", START_COLOR))

    legend = np.full((8 + 20 * len(entries), w, 3), 24, dtype=np.uint8)
    for i, (text, color) in enumerate(entries):
        y = 18 + i * 20
        cv2.circle(legend, (16, y - 5), 6, color, -1, cv2.LINE_AA)
        cv2.putText(legend, text, (32, y), _FONT, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([canvas, legend])


def draw_caption(image_rgb, lines):
    """在图顶部压一条英文信息条（cv2 不渲染中文）。

    Args:
        image_rgb (np.ndarray): RGB 图。
        lines: 字符串或字符串列表。

    Returns:
        np.ndarray: 拼了信息条的新图。
    """
    if isinstance(lines, str):
        lines = [lines]

    out = image_rgb
    thick, pad, scale = 1, 6, 0.5
    width = out.shape[1]
    text_w = max(width - 2 * pad, 16)

    line_h = int(cv2.getTextSize("Ag", _FONT, scale, thick)[0][1]) + pad
    strip = np.full((line_h * len(lines) + pad, width, 3), 24, dtype=np.uint8)
    for i, line in enumerate(lines):
        text = str(line)
        while len(text) > 4 and cv2.getTextSize(text, _FONT, scale, thick)[0][0] > text_w:
            text = text[:-1]
        cv2.putText(strip, text, (pad, pad + (i + 1) * line_h - 3), _FONT, scale,
                    (235, 235, 235), thick, cv2.LINE_AA)

    return np.vstack([strip, out])


def resize_for_video(canvas, size):
    """按长边缩到 ``size``，短边补黑边，保持几何比例。

    Args:
        canvas (np.ndarray): RGB 图。
        size (int): 输出正方形边长。

    Returns:
        np.ndarray: 缩放后的图。
    """
    h, w = canvas.shape[:2]
    if size <= 0 or (h == size and w == size):
        return canvas
    if h != w:
        scale = size / max(h, w)
        resized = cv2.resize(canvas, (max(int(round(w * scale)), 1),
                                      max(int(round(h * scale)), 1)),
                             interpolation=cv2.INTER_AREA)
        out = np.zeros((size, size, 3), dtype=canvas.dtype)
        out[: resized.shape[0], : resized.shape[1]] = resized
        return out
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def save_image_rgb(path, image_rgb):
    """把 RGB 图写成文件（仅在此处转为 BGR）。

    Args:
        path (str): 输出路径。
        image_rgb (np.ndarray): RGB 图。

    Returns:
        str: 写出路径。

    Raises:
        IOError: 写入失败。
    """
    if not cv2.imwrite(path, cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write image: {path}")
    return path


def selfcheck(env, samples=64):
    """把 :func:`world_xz_to_pixel` 与 habitat ``maps.to_grid`` 逐点比对。

    Args:
        env: habitat 环境。
        samples (int): 采样点数的大约值。

    Returns:
        tuple: ``(是否通过, 最大像素差, 实际比对点数)``。
    """
    canvas, _info = build_topdown_image(env)
    origin, spans = topdown_geometry(env)
    pf = env.sim.pathfinder
    grid_resolution = canvas.shape[:2]

    lower, upper = pf.get_bounds()
    n = max(int(np.sqrt(max(samples, 1))), 2)
    xs = np.linspace(lower[0], upper[0], n)
    zs = np.linspace(lower[2], upper[2], n)

    worst, checked = 0, 0
    for x in xs:
        for z in zs:
            expected = to_grid(float(z), float(x), grid_resolution, pathfinder=pf)
            got = world_xz_to_pixel((x, z), origin, spans, grid_resolution)
            worst = max(worst, abs(int(expected[0]) - got[0]), abs(int(expected[1]) - got[1]))
            checked += 1
    return worst <= 1, worst, checked
