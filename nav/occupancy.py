"""从 RGB-D 增量构建点云占用图，并渲染全景与 BEV。

占用栅格对齐 ApexNav：障碍带高度点、两遍射线、log-odds 多数票、膨胀。
规划与跟随走自建 2D occupancy A*，点云仍用于彩色 BEV。
"""

import math
import os

import cv2
import numpy as np
import open3d as o3d
import quaternion

from nav.geometry import get_pointcloud_from_depth, preprocess_depth, translate_to_world
from nav.transform import as_quaternion, habitat_camera_intrinsic

# habitat ``_DefaultHabitatSimActions`` 编号。
HAB_STOP, HAB_FORWARD, HAB_LEFT, HAB_RIGHT = 0, 1, 2, 3
HAB_LOOK_UP, HAB_LOOK_DOWN = 4, 5

# 12 张 30° 环视的默认落盘序号：奇数朝向；0 是未转动的当前观测。
DEFAULT_PANO_INDICES = (1, 3, 5, 7, 9, 11)
EVEN_PANO_INDICES = (0, 2, 4, 6, 8, 10)
# 送给规划器的拼图与俯视图上限。
VLM_PANO_WH = (960, 480)
VLM_BEV_MAX_WH = (960, 480)


# 扇区半宽（弧度）：偶序号朝向中心 ±40°，相邻扇区约 20° 重叠。
SECTOR_HALF_RAD = math.radians(40.0)
# Planner / Mover 近距探索点：测地小于该值才进 unexplored 与 Mover 候选。
NEAR_FRONTIER_GEO_M = 5.0
# 按 A* 路径分扇区时，取距机身约该距离的采样点方位。
PATH_SECTOR_SAMPLE_M = 0.75


def _rel_bearing(xyz, origin_xyz, node_yaw):
    """相对扫描朝向的方位角，落在 ``[0, 2π)``。"""
    p = np.asarray(xyz, dtype=np.float64).reshape(3)
    origin = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
    bearing = math.atan2(p[0] - origin[0], p[2] - origin[2])
    return (float(node_yaw) - bearing) % (2 * math.pi)


def even_pano_dir(xyz, origin_xyz, node_yaw):
    """把世界点划到扫描朝向的偶序号方向（0、2、…、10）。

    Args:
        xyz: 目标世界坐标。
        origin_xyz: 节点位置。
        node_yaw (float): 扫描时机身朝向，弧度。

    Returns:
        int: 偶序号。
    """
    rel = _rel_bearing(xyz, origin_xyz, node_yaw)
    idx = int(round(rel / (math.pi / 6.0))) % 12
    return idx if idx % 2 == 0 else (idx + 1) % 12


def path_sample_xyz(occ, agent_xyz, goal_xyz, sample_m=PATH_SECTOR_SAMPLE_M):
    """沿 A* 路径取距机身约 ``sample_m`` 的点；无路径则 None。

    Args:
        occ: ``OccupancyMap``。
        agent_xyz: 起点。
        goal_xyz: 终点。
        sample_m (float): 沿路径采样距离，米。

    Returns:
        np.ndarray | None: ``(3,)`` 世界坐标。
    """
    from nav.astar2d import astar_or_relax

    agent = np.asarray(agent_xyz, dtype=np.float64).reshape(3)
    goal = np.asarray(goal_xyz, dtype=np.float64).reshape(3)
    path, _geo, _ = astar_or_relax(occ, agent, goal, success_dist=0.35)
    if path is None or len(path) < 2:
        return None
    sx, sz = float(agent[0]), float(agent[2])
    px, pz = sx, sz
    acc = 0.0
    chosen = path[1]
    for i in range(1, len(path)):
        x, z = float(path[i][0]), float(path[i][1])
        acc += math.hypot(x - px, z - pz)
        px, pz = x, z
        chosen = path[i]
        if acc >= float(sample_m) - 1e-9:
            break
    return np.array([float(chosen[0]), float(agent[1]), float(chosen[1])],
                    dtype=np.float64)


def path_even_pano_dir(occ, agent_xyz, goal_xyz, node_yaw,
                       sample_m=PATH_SECTOR_SAMPLE_M):
    """按 A* 路径起步方位划分偶序号扇区；无路径则回退欧氏。

    Args:
        occ: ``OccupancyMap``。
        agent_xyz: 机身位置。
        goal_xyz: 目标世界坐标。
        node_yaw (float): 扫描朝向。
        sample_m (float): 路径采样距离。

    Returns:
        int: 偶序号朝向。
    """
    sample = path_sample_xyz(occ, agent_xyz, goal_xyz, sample_m=sample_m)
    ref = sample if sample is not None else goal_xyz
    return even_pano_dir(ref, agent_xyz, node_yaw)


def planner_sector_dirs(frontiers, agent_xyz, node_yaw, occ,
                        max_geo=NEAR_FRONTIER_GEO_M):
    """近距且 A* 可达的探索点按路径起步朝向得到的扇区集合。

    Args:
        frontiers (list): ``extract_frontiers`` 结果。
        agent_xyz: 节点位置。
        node_yaw (float): 扫描朝向。
        occ: ``OccupancyMap``。
        max_geo (float): 测地上限，米。

    Returns:
        set: 偶序号 ``pano_id``。
    """
    dirs = set()
    for fr in frontiers or []:
        geo = fr.get("geodesic_m")
        if geo is None:
            continue
        try:
            g = float(geo)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(g) or g >= float(max_geo) - 1e-9:
            continue
        dirs.add(int(path_even_pano_dir(occ, agent_xyz, fr["xyz"], node_yaw)))
    return dirs


def frontiers_in_dir(frontiers, pano_id, origin_xyz, node_yaw,
                     half_rad=SECTOR_HALF_RAD, occ=None):
    """取出落入某一偶序号朝向（含边缘重叠带）的探索点。

    若提供 ``occ``，用 A* 路径起步点方位分扇区，否则用目标欧氏方位。

    Args:
        frontiers (list): extract_frontiers 的结果。
        pano_id (int): 朝向编号。
        origin_xyz: 节点位置。
        node_yaw (float): 扫描时机身朝向。
        half_rad (float): 相对扇区中心的半宽，默认 40°。
        occ: ``OccupancyMap``，可选。

    Returns:
        list: 属于该朝向的探索点。
    """
    pid = int(pano_id)
    center = (pid % 12) * (math.pi / 6.0)
    out = []
    for fr in frontiers:
        ref = fr["xyz"]
        if occ is not None:
            sample = path_sample_xyz(occ, origin_xyz, fr["xyz"])
            if sample is not None:
                ref = sample
        rel = _rel_bearing(ref, origin_xyz, node_yaw)
        delta = (rel - center + math.pi) % (2 * math.pi) - math.pi
        if abs(delta) <= float(half_rad) + 1e-9:
            out.append(fr)
    return out


def _draw_dashed_polyline(image, pts, color, thickness=1, dash=5, gap=4):
    """在像素折线上画虚线。

    Args:
        image (np.ndarray): ``(H, W, 3)``。
        pts (list): ``(col, row)`` 像素点。
        color: RGB 三元组。
        thickness (int): 线宽。
        dash (int): 实线段像素长。
        gap (int): 空隙像素长。
    """
    if len(pts) < 2:
        return
    dash = max(1, int(dash))
    gap = max(0, int(gap))
    draw_on = True
    remain = dash
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = float(x1 - x0), float(y1 - y0)
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            continue
        ux, uy = dx / seg_len, dy / seg_len
        traveled = 0.0
        cx, cy = float(x0), float(y0)
        while traveled < seg_len - 1e-9:
            step = min(float(remain), seg_len - traveled)
            nx = cx + ux * step
            ny = cy + uy * step
            if draw_on:
                cv2.line(image, (int(round(cx)), int(round(cy))),
                         (int(round(nx)), int(round(ny))), color, thickness,
                         cv2.LINE_AA)
            traveled += step
            cx, cy = nx, ny
            remain -= step
            if remain <= 1e-9:
                draw_on = not draw_on
                remain = float(dash if draw_on else gap)
                if remain <= 0:
                    draw_on = not draw_on
                    remain = float(dash if draw_on else gap)

# BEV 上扫描节点与连线（RGB）。
SCAN_NODE_COLOR = (0, 80, 255)
SCAN_PATH_COLOR = (0, 80, 255)
AGENT_COLOR = (255, 0, 0)
SECTOR_RAY_COLOR = (0, 255, 255)
SECTOR_TEXT_COLOR = (255, 0, 0)
FRONTIER_COLOR = (0, 220, 0)

# 2D 占用栅格：行↔z、列↔x。离散态由 log-odds 阈值得到。
CELL_UNKNOWN, CELL_FREE, CELL_OCC = 0, 1, 2
GRID_UNKNOWN_RGB = (28, 28, 28)
GRID_FREE_RGB = (120, 120, 120)
GRID_OCC_RGB = (230, 230, 230)
GRID_INFLATE_RGB = (90, 70, 70)
ASTAR_PATH_COLOR = (0, 200, 255)
# Occ 调试图：A* 路径用红虚线（RGB）。
OCC_ASTAR_DASH_RGB = (220, 40, 40)
OCC_AGENT_RGB = (255, 60, 60)

# ApexNav algorithm.xml（Habitat 仿真）。
P_HIT, P_MISS, P_MIN, P_MAX, P_OCC = 0.90, 0.48, 0.10, 0.98, 0.80
INFLATE_RADIUS_M = 0.18
OBSTACLE_H_LO, OBSTACLE_H_HI = 0.28, 1.18
CLUSTER_MIN, CLUSTER_SIZE_XY = 8, 0.65
CLUSTER_MIN_NECK = 3
OPENING_MIN_M = 0.60
NECK_WIDTH_LO_M, NECK_WIDTH_HI_M = 0.45, 1.30
FRONTIER_NEAR_DROP_M = 0.15
FRONTIER_NMS_M = 0.80
FRONTIER_PER_DIR = 2
FRONTIER_MAX_N = 12


def _logit(p):
    """概率 -> log-odds。"""
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


PROB_HIT_LOG = _logit(P_HIT)
PROB_MISS_LOG = _logit(P_MISS)
CLAMP_MIN_LOG = _logit(P_MIN)
CLAMP_MAX_LOG = _logit(P_MAX)
MIN_OCCUPANCY_LOG = _logit(P_OCC)
UNKNOWN_LOG = CLAMP_MIN_LOG - 0.01


def _bresenham(r0, c0, r1, c1):
    """二维栅格直线上的格子（含两端）。

    Args:
        r0, c0, r1, c1 (int): 起终点行列。

    Returns:
        list: ``(row, col)``。
    """
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells


def _connected_components(mask, connectivity=8):
    """连通域标记，标签从 1 起。

    Args:
        mask (np.ndarray): 布尔图。
        connectivity (int): 4 或 8。

    Returns:
        tuple: ``(labels int32, n_labels)``。
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    n = 0
    if connectivity == 8:
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    else:
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            n += 1
            stack = [(y, x)]
            labels[y, x] = n
            while stack:
                cy, cx = stack.pop()
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = n
                        stack.append((ny, nx))
    return labels, n


N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
N8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def to_rgb_uint8(rgb):
    """把观测压成 ``(H, W, 3)`` uint8 RGB。

    Args:
        rgb (np.ndarray): habitat RGB / RGBA / 灰度观测。

    Returns:
        np.ndarray: ``(H, W, 3)`` uint8。
    """
    img = np.asarray(rgb)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[-1] == 4:
        return np.ascontiguousarray(img[:, :, :3])
    return np.ascontiguousarray(img[:, :, :3])


def clean_depth(raw_depth, min_depth, max_depth):
    """深度转 float32 米制，并按传感器量程剔除哨兵值。

    Args:
        raw_depth: habitat 原始深度，``(H, W)`` 或 ``(H, W, 1)``。
        min_depth (float): 最近量程，也是未命中哨兵。
        max_depth (float): 最远量程。

    Returns:
        np.ndarray: ``(H, W)`` float32，无效处为 0。
    """
    depth = np.asarray(raw_depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return preprocess_depth(depth.copy(), lower_bound=min_depth + 1e-3,
                            upper_bound=max_depth - 0.1)


def sensor_pose(env):
    """返回相机位姿 ``(位置[3], quaternion)``，只用于反投影。

    Args:
        env: ``habitat.Env``。

    Returns:
        tuple: ``(np.ndarray (3,), quaternion.quaternion)``。
    """
    state = env.sim.get_agent_state(0)
    if len(state.sensor_states) == 0:
        return np.asarray(state.position, dtype=np.float64), as_quaternion(state.rotation)
    key = "rgb" if "rgb" in state.sensor_states else next(iter(state.sensor_states))
    sensor = state.sensor_states[key]
    return np.asarray(sensor.position, dtype=np.float64), as_quaternion(sensor.rotation)


def body_position(env):
    """机身世界坐标，用于轨迹节点。不要拿相机高度去画地面轨迹。

    Args:
        env: ``habitat.Env``。

    Returns:
        np.ndarray: ``(3,)`` float64。
    """
    return np.asarray(env.sim.get_agent_state().position, dtype=np.float64)


def save_rgb(path, image_rgb):
    """把 RGB 图写成文件。

    Args:
        path (str): 输出路径。
        image_rgb (np.ndarray): ``(H, W, 3)`` RGB uint8。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def resize_exact(image_rgb, width, height):
    """缩放到指定宽高。

    Args:
        image_rgb (np.ndarray): ``(H, W, 3)`` RGB。
        width (int): 目标宽。
        height (int): 目标高。

    Returns:
        np.ndarray: 缩放后的 RGB。
    """
    w, h = int(width), int(height)
    if w < 1 or h < 1:
        raise ValueError("目标宽高必须为正")
    img = to_rgb_uint8(image_rgb)
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def fit_within(image_rgb, max_width, max_height):
    """保持纵横比缩进矩形；已更小则原样返回。

    Args:
        image_rgb (np.ndarray): ``(H, W, 3)`` RGB。
        max_width (int): 最大宽。
        max_height (int): 最大高。

    Returns:
        np.ndarray: 缩放或原图。
    """
    img = to_rgb_uint8(image_rgb)
    h, w = img.shape[:2]
    mw, mh = int(max_width), int(max_height)
    if w <= mw and h <= mh:
        return img
    scale = min(mw / float(w), mh / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def annotate_index(image, index, label_prefix="Direction"):
    """在画面上写朝向序号。

    Args:
        image (np.ndarray): RGB 图，不会被原地修改。
        index (int): 环视序号。
        label_prefix (str): 文字前缀。

    Returns:
        np.ndarray: 带序号的拷贝。
    """
    canvas = np.ascontiguousarray(image.copy())
    h, w = canvas.shape[:2]
    text = f"{label_prefix} {index}"
    scale = max(0.8, min(w, h) / 320.0)
    thickness = max(2, int(round(scale * 2.5)))
    org = (int(0.06 * w), int(0.18 * h))
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 0, 0), thickness, cv2.LINE_AA)
    return canvas


def concat_panorama(images, indices, gap=10):
    """按给定序号选出若干张图，标号后拼成一张。

    Args:
        images (list): 长度至少覆盖 ``max(indices)`` 的 RGB 列表。
        indices (sequence): 要放入拼图的序号。
        gap (int): 图块间距像素。

    Returns:
        np.ndarray: 拼好的 RGB 图；``indices`` 为空时返回 8×8 黑图。
    """
    picked = []
    for idx in indices:
        if idx < 0 or idx >= len(images):
            continue
        picked.append(annotate_index(images[idx], idx))
    if not picked:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    h, w = picked[0].shape[:2]
    n = len(picked)
    cols = min(3, n)
    rows = int(math.ceil(n / float(cols)))
    canvas = np.zeros((rows * h + (rows + 1) * gap,
                       cols * w + (cols + 1) * gap, 3), dtype=np.uint8)
    for i, img in enumerate(picked):
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0 = gap * (r + 1) + r * h
        x0 = gap * (c + 1) + c * w
        canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


def save_panorama(images, out_dir, stem, indices=DEFAULT_PANO_INDICES, merge=True):
    """保存全景：合并成一张，或按序号各存一张；两种方式都会写上序号。

    Args:
        images (list): ``rotate_times`` 张 RGB。
        out_dir (str): 输出目录。
        stem (str): 文件名前缀，如 ``scan0_pano``。
        indices (sequence): 要保存的朝向序号。
        merge (bool): True 拼成一张，False 分别落盘。

    Returns:
        list: 写出去的文件路径。
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    if merge:
        canvas = concat_panorama(images, indices)
        path = os.path.join(out_dir, f"{stem}.png")
        save_rgb(path, canvas)
        paths.append(path)
        return paths
    for idx in indices:
        if idx < 0 or idx >= len(images):
            continue
        path = os.path.join(out_dir, f"{stem}_{idx}.png")
        save_rgb(path, annotate_index(images[idx], idx))
        paths.append(path)
    return paths


class OccupancyMap:
    """全局点云占用图：增量融合、环视扫描、BEV 渲染。"""

    def __init__(self, intrinsic, min_depth=0.5, max_depth=5.0, voxel_size=0.10,
                 height_low=0.4, height_high=1.6, skip_near_wall=True, grid_res=0.10):
        """初始化空点云与空栅格。

        高度切片相对**机身**（贴地），不是相对相机。相机约在机身上方 0.88 m；若误用
        相机高度再减 0.7 m，上沿只剩约 0.2 m，桌子和墙会被切掉，地板只剩薄薄一层。

        Args:
            intrinsic (np.ndarray): ``3x3`` 相机内参。
            min_depth (float): 深度传感器近端，用于剔除未命中哨兵。
            max_depth (float): 深度传感器远端。
            voxel_size (float): Open3D 体素下采样边长，米。
            height_low (float): 相对机身往下保留的米数（含地板）。
            height_high (float): 相对机身往上保留的米数（家具与墙，切掉天花板）。
            skip_near_wall (bool): 近距离大面积贴墙时跳过该帧，避免噪点灌进地图。
            grid_res (float): 2D 三值栅格边长，米。
        """
        self.intrinsic = np.asarray(intrinsic, dtype=np.float32)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.voxel_size = float(voxel_size)
        self.height_low = float(height_low)
        self.height_high = float(height_high)
        self.skip_near_wall = bool(skip_near_wall)
        self.grid_res = float(grid_res)
        self.inflate_radius = float(INFLATE_RADIUS_M)
        self.global_pcd = o3d.geometry.PointCloud()
        self.initial_height = None
        self.path_positions = []
        self.scan_nodes = []
        self.grid = None
        self.occupancy_log = None
        self.inflate = None
        self._ox = 0.0
        self._oz = 0.0

    @classmethod
    def from_config(cls, config, **kwargs):
        """从 habitat 配置读内参与深度量程。

        Args:
            config: ``hm3d_config`` / ``play_config`` 返回的配置。
            **kwargs: 覆盖 ``OccupancyMap`` 构造参数。

        Returns:
            OccupancyMap: 新的空地图。
        """
        sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        spec = getattr(sensors, "depth_sensor", None) or sensors.rgb_sensor
        kwargs.setdefault("min_depth", float(spec.min_depth))
        kwargs.setdefault("max_depth", float(spec.max_depth))
        return cls(habitat_camera_intrinsic(config), **kwargs)

    def reset(self):
        """清空点云、轨迹、扫描节点与栅格。"""
        self.global_pcd = o3d.geometry.PointCloud()
        self.initial_height = None
        self.path_positions = []
        self.scan_nodes = []
        self.grid = None
        self.occupancy_log = None
        self.inflate = None
        self._ox = 0.0
        self._oz = 0.0

    def record_body(self, position, min_step=1e-3):
        """把机身位置记进历史轨迹；位移过小则忽略。

        Args:
            position: 机身世界坐标。
            min_step (float): 与上一点的最小欧氏距离，米。
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        if self.path_positions and np.linalg.norm(pos - self.path_positions[-1]) < min_step:
            return
        self.path_positions.append(pos)

    def mark_scan_node(self, position):
        """在一次环视结束的位置记下蓝色空心节点。

        Args:
            position: 机身世界坐标。
        """
        self.scan_nodes.append(np.asarray(position, dtype=np.float64).reshape(3))

    def _expand_grid(self, xs, zs, pad_m=1.5):
        """保证栅格覆盖这些 XZ 点。

        Args:
            xs (np.ndarray): x 坐标。
            zs (np.ndarray): z 坐标。
            pad_m (float): 外扩，米。
        """
        min_x = float(np.min(xs) - pad_m)
        max_x = float(np.max(xs) + pad_m)
        min_z = float(np.min(zs) - pad_m)
        max_z = float(np.max(zs) + pad_m)
        res = self.grid_res
        if self.grid is None:
            w = max(8, int(math.ceil((max_x - min_x) / res)))
            h = max(8, int(math.ceil((max_z - min_z) / res)))
            self.grid = np.zeros((h, w), dtype=np.uint8)
            self.occupancy_log = np.full((h, w), UNKNOWN_LOG, dtype=np.float32)
            self.inflate = np.zeros((h, w), dtype=np.uint8)
            self._ox, self._oz = min_x, min_z
            return
        h, w = self.grid.shape
        old_min_x, old_min_z = self._ox, self._oz
        old_max_x = old_min_x + w * res
        old_max_z = old_min_z + h * res
        new_min_x = min(old_min_x, min_x)
        new_min_z = min(old_min_z, min_z)
        new_max_x = max(old_max_x, max_x)
        new_max_z = max(old_max_z, max_z)
        if (new_min_x >= old_min_x - 1e-9 and new_min_z >= old_min_z - 1e-9
                and new_max_x <= old_max_x + 1e-9 and new_max_z <= old_max_z + 1e-9):
            return
        nw = max(8, int(math.ceil((new_max_x - new_min_x) / res)))
        nh = max(8, int(math.ceil((new_max_z - new_min_z) / res)))
        grown = np.zeros((nh, nw), dtype=np.uint8)
        grown_log = np.full((nh, nw), UNKNOWN_LOG, dtype=np.float32)
        grown_inf = np.zeros((nh, nw), dtype=np.uint8)
        off_c = int(round((old_min_x - new_min_x) / res))
        off_r = int(round((old_min_z - new_min_z) / res))
        grown[off_r:off_r + h, off_c:off_c + w] = self.grid
        grown_log[off_r:off_r + h, off_c:off_c + w] = self.occupancy_log
        grown_inf[off_r:off_r + h, off_c:off_c + w] = self.inflate
        self.grid = grown
        self.occupancy_log = grown_log
        self.inflate = grown_inf
        self._ox, self._oz = new_min_x, new_min_z

    def world_to_cell(self, x, z):
        """世界 ``(x, z)`` -> 栅格 ``(row, col)``。栅格未初始化时返回 ``None``。

        Args:
            x (float): 世界 x。
            z (float): 世界 z。

        Returns:
            tuple: ``(row, col)`` 或 ``None``。
        """
        if self.grid is None:
            return None
        col = int(math.floor((float(x) - self._ox) / self.grid_res))
        row = int(math.floor((float(z) - self._oz) / self.grid_res))
        h, w = self.grid.shape
        if row < 0 or col < 0 or row >= h or col >= w:
            return None
        return row, col

    def cell_to_world(self, row, col, y=0.88):
        """栅格中心 -> 世界 ``(x, y, z)``。

        Args:
            row (int): 行。
            col (int): 列。
            y (float): 高度。

        Returns:
            np.ndarray: ``(3,)``。
        """
        x = self._ox + (col + 0.5) * self.grid_res
        z = self._oz + (row + 0.5) * self.grid_res
        return np.array([x, y, z], dtype=np.float64)

    def _discretize_cell(self, row, col):
        """log-odds -> 三值。"""
        log = float(self.occupancy_log[row, col])
        if log < CLAMP_MIN_LOG - 1e-3:
            self.grid[row, col] = CELL_UNKNOWN
        elif log > MIN_OCCUPANCY_LOG:
            self.grid[row, col] = CELL_OCC
        else:
            self.grid[row, col] = CELL_FREE

    def _refresh_inflate(self, r0, c0, r1, c1):
        """在包围盒（再外扩膨胀半径）内清圈并按 OCC 重涂。"""
        if self.grid is None:
            return
        h, w = self.grid.shape
        step = max(1, int(math.ceil(self.inflate_radius / self.grid_res)))
        r0 = max(0, r0 - step)
        c0 = max(0, c0 - step)
        r1 = min(h - 1, r1 + step)
        c1 = min(w - 1, c1 + step)
        self.inflate[r0:r1 + 1, c0:c1 + 1] = 0
        occ_rs, occ_cs = np.where(self.grid[r0:r1 + 1, c0:c1 + 1] == CELL_OCC)
        for rr, cc in zip(occ_rs, occ_cs):
            r, c = r0 + int(rr), c0 + int(cc)
            r_lo, r_hi = max(0, r - step), min(h, r + step + 1)
            c_lo, c_hi = max(0, c - step), min(w, c + step + 1)
            for ir in range(r_lo, r_hi):
                for ic in range(c_lo, c_hi):
                    dx = (ic - c) * self.grid_res
                    dz = (ir - r) * self.grid_res
                    if dx * dx + dz * dz <= self.inflate_radius * self.inflate_radius + 1e-9:
                        self.inflate[ir, ic] = 1

    def stamp_agent_free(self, xyz, radius=0.22):
        """把机身脚下 UNKNOWN/膨胀标成 FREE，避免出生点卡死。"""
        if xyz is None:
            return
        p = np.asarray(xyz, dtype=np.float64).reshape(3)
        self._expand_grid(np.array([p[0]]), np.array([p[2]]), pad_m=radius + 0.3)
        cell = self.world_to_cell(p[0], p[2])
        if cell is None:
            return
        cr, cc = cell
        n = max(1, int(math.ceil(radius / self.grid_res)))
        h, w = self.grid.shape
        r0, r1, c0, c1 = cr, cr, cc, cc
        for dr in range(-n, n + 1):
            for dc in range(-n, n + 1):
                r, c = cr + dr, cc + dc
                if r < 0 or c < 0 or r >= h or c >= w:
                    continue
                dx, dz = dc * self.grid_res, dr * self.grid_res
                if dx * dx + dz * dz > radius * radius:
                    continue
                if self.grid[r, c] == CELL_OCC:
                    continue
                self.occupancy_log[r, c] = float(CLAMP_MIN_LOG + 0.15)
                self.grid[r, c] = CELL_FREE
                r0, r1 = min(r0, r), max(r1, r)
                c0, c1 = min(c0, c), max(c1, c)
        self._refresh_inflate(r0, c0, r1, c1)
        for dr in range(-n, n + 1):
            for dc in range(-n, n + 1):
                r, c = cr + dr, cc + dc
                if r < 0 or c < 0 or r >= h or c >= w:
                    continue
                dx, dz = dc * self.grid_res, dr * self.grid_res
                if dx * dx + dz * dz > radius * radius:
                    continue
                if self.grid[r, c] != CELL_OCC:
                    self.inflate[r, c] = 0

    def force_occ_xz(self, x, z):
        """把世界 XZ 强制标成 OCC 并刷新膨胀（对齐 ApexNav ``setForceOccGrid``）。"""
        if self.grid is None:
            return
        self._expand_grid(np.array([float(x)]), np.array([float(z)]), pad_m=0.4)
        cell = self.world_to_cell(float(x), float(z))
        if cell is None:
            return
        r, c = cell
        self.occupancy_log[r, c] = float(CLAMP_MAX_LOG)
        self.grid[r, c] = CELL_OCC
        self._refresh_inflate(r, c, r, c)

    def sculpt_hits(self, cam_xyz, hit_xyz):
        """两遍射线 + 每帧多数票 log-odds，再离散并膨胀。

        Pass1 标记障碍命中格；Pass2 沿射线 miss，跳过 occ 端点，避免把墙挖空。
        """
        if hit_xyz is None or len(hit_xyz) == 0:
            return
        hits = np.asarray(hit_xyz, dtype=np.float64)
        if hits.ndim == 1:
            hits = hits.reshape(1, 3)
        cam = np.asarray(cam_xyz, dtype=np.float64).reshape(3)
        xs = np.concatenate([hits[:, 0], [cam[0]]])
        zs = np.concatenate([hits[:, 2], [cam[2]]])
        self._expand_grid(xs, zs)
        h, w = self.grid.shape
        cam_cell = self.world_to_cell(cam[0], cam[2])
        if cam_cell is None:
            return
        cr, cc = cam_cell
        max_ray = max(self.max_depth - 1e-3, 0.5)
        stride = max(1, int(len(hits) / 2500))
        sampled = hits[::stride]
        flag_occ = set()
        hit_cnt = {}
        miss_cnt = {}

        def bump(store, key):
            store[key] = store.get(key, 0) + 1

        for px, pz in sampled[:, [0, 2]]:
            length = math.hypot(float(px) - cam[0], float(pz) - cam[2])
            is_occ = length <= max_ray
            if length > max_ray and length > 1e-6:
                t = max_ray / length
                px = cam[0] + t * (px - cam[0])
                pz = cam[2] + t * (pz - cam[2])
                is_occ = False
            cell = self.world_to_cell(px, pz)
            if cell is None:
                continue
            if is_occ:
                flag_occ.add(cell)
                bump(hit_cnt, cell)

        for px, pz in sampled[:, [0, 2]]:
            length = math.hypot(float(px) - cam[0], float(pz) - cam[2])
            if length > max_ray and length > 1e-6:
                t = max_ray / length
                px = cam[0] + t * (px - cam[0])
                pz = cam[2] + t * (pz - cam[2])
            cell = self.world_to_cell(px, pz)
            if cell is None:
                continue
            hr, hc = cell
            ray = _bresenham(cr, cc, hr, hc)
            for r, c in ray:
                if r == hr and c == hc:
                    if (r, c) not in flag_occ:
                        bump(miss_cnt, (r, c))
                    break
                if 0 <= r < h and 0 <= c < w:
                    if (r, c) in flag_occ:
                        continue
                    bump(miss_cnt, (r, c))

        touched = set(hit_cnt) | set(miss_cnt)
        if not touched:
            return
        r0 = min(t[0] for t in touched)
        r1 = max(t[0] for t in touched)
        c0 = min(t[1] for t in touched)
        c1 = max(t[1] for t in touched)
        for r, c in touched:
            if r < 0 or c < 0 or r >= h or c >= w:
                continue
            n_hit = hit_cnt.get((r, c), 0)
            n_miss = miss_cnt.get((r, c), 0)
            update = PROB_HIT_LOG if n_hit >= n_miss else PROB_MISS_LOG
            if self.occupancy_log[r, c] < CLAMP_MIN_LOG - 1e-3:
                self.occupancy_log[r, c] = MIN_OCCUPANCY_LOG
            val = float(self.occupancy_log[r, c]) + update
            self.occupancy_log[r, c] = min(max(val, CLAMP_MIN_LOG), CLAMP_MAX_LOG)
            self._discretize_cell(r, c)
        self._refresh_inflate(r0, c0, r1, c1)

    def _is_frontier_cell(self, row, col, allow_inflate=False):
        """UNKNOWN ∧ 四邻 FREE；默认排除外扩，门缝模式可放行外扩格。"""
        if self.grid is None:
            return False
        h, w = self.grid.shape
        if row < 0 or col < 0 or row >= h or col >= w:
            return False
        if (not allow_inflate) and self.inflate is not None and self.inflate[row, col] != 0:
            return False
        if self.grid[row, col] != CELL_UNKNOWN:
            return False
        for dr, dc in N4:
            nr, nc = row + dr, col + dc
            if 0 <= nr < h and 0 <= nc < w and self.grid[nr, nc] == CELL_FREE:
                return True
        return False

    def _visible_frontier_xyz(self, agent_xyz, xyz, y, allow_inflate=False):
        """agent 到质心的占用栅格视线：穿 OCC 则退到最后一个可见边界格。"""
        agent = np.asarray(agent_xyz, dtype=np.float64).reshape(3)
        tgt = np.asarray(xyz, dtype=np.float64).reshape(3)
        ac = self.world_to_cell(agent[0], agent[2])
        tc = self.world_to_cell(tgt[0], tgt[2])
        if ac is None or tc is None:
            return tgt
        cells = _bresenham(ac[0], ac[1], tc[0], tc[1])
        h, w = self.grid.shape
        hit_occ = False
        last_fr = None
        for r, c in cells[1:]:
            if r < 0 or c < 0 or r >= h or c >= w:
                hit_occ = True
                break
            if self.grid[r, c] == CELL_OCC:
                hit_occ = True
                break
            if self._is_frontier_cell(r, c, allow_inflate=allow_inflate):
                last_fr = (r, c)
        if not hit_occ:
            return tgt
        if last_fr is None or last_fr == (ac[0], ac[1]):
            return None
        vis = self.cell_to_world(last_fr[0], last_fr[1], y=y)
        if float(np.hypot(vis[0] - agent[0], vis[2] - agent[2])) < FRONTIER_NEAR_DROP_M:
            return None
        return vis

    def _clear_width_m(self, row, col, dr, dc):
        """沿 ``(dr,dc)`` 垂直方向量两侧到障碍的净宽，米。"""
        h, w = self.grid.shape
        norm = math.hypot(dr, dc)
        if norm < 1e-9:
            return 0.0
        # 路径切向 (dr,dc) 的平面垂直：(dc, -dr)
        pr, pc = float(dc) / norm, -float(dr) / norm
        res = float(self.grid_res)

        def ray(sr, sc):
            dist = 0.0
            for i in range(1, max(1, int(math.ceil(NECK_WIDTH_HI_M / res)) + 2)):
                rr = int(round(row + sr * i))
                cc = int(round(col + sc * i))
                if rr < 0 or cc < 0 or rr >= h or cc >= w:
                    return dist
                if int(self.grid[rr, cc]) == CELL_OCC:
                    return dist
                dist = i * res
            return dist

        return ray(pr, pc) + ray(-pr, -pc)

    def _path_has_door_neck(self, agent_xyz, xyz):
        """机身到目标格子连线上是否存在门宽缩窄（两侧障碍、净宽 0.45–1.30 m）。"""
        agent = np.asarray(agent_xyz, dtype=np.float64).reshape(3)
        tgt = np.asarray(xyz, dtype=np.float64).reshape(3)
        ac = self.world_to_cell(agent[0], agent[2])
        tc = self.world_to_cell(tgt[0], tgt[2])
        if ac is None or tc is None:
            return False
        cells = _bresenham(ac[0], ac[1], tc[0], tc[1])
        if len(cells) < 2:
            return False
        for i in range(1, len(cells)):
            r0, c0 = cells[i - 1]
            r1, c1 = cells[i]
            dr, dc = r1 - r0, c1 - c0
            if int(self.grid[r1, c1]) == CELL_OCC:
                continue
            width = self._clear_width_m(r1, c1, dr, dc)
            if NECK_WIDTH_LO_M - 1e-6 <= width <= NECK_WIDTH_HI_M + 1e-6:
                return True
        return False

    def grid_line_clear(self, a_xyz, b_xyz):
        """占用图直线是否不穿 OCC（可穿 FREE / UNKNOWN / 外扩）。"""
        a = np.asarray(a_xyz, dtype=np.float64).reshape(3)
        b = np.asarray(b_xyz, dtype=np.float64).reshape(3)
        ac = self.world_to_cell(a[0], a[2])
        bc = self.world_to_cell(b[0], b[2])
        if ac is None or bc is None:
            return False
        h, w = self.grid.shape
        for r, c in _bresenham(ac[0], ac[1], bc[0], bc[1])[1:]:
            if r < 0 or c < 0 or r >= h or c >= w:
                return False
            if int(self.grid[r, c]) == CELL_OCC:
                return False
        return True

    def _opening_dir_xz(self, group):
        """簇内从可走格指向未知格的平均平面方向。"""
        sx = sz = n = 0.0
        h, w = self.grid.shape
        for _x, _z, r, c in group:
            for dr, dc in N4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and self.grid[nr, nc] == CELL_FREE:
                    sx += float(c - nc)
                    sz += float(r - nr)
                    n += 1.0
        if n < 1.0:
            return None
        dx, dz = sx / n, sz / n
        norm = math.hypot(dx, dz)
        if norm < 1e-6:
            return None
        return dx / norm, dz / norm

    def _opening_depth_m(self, group, xyz):
        """沿未知开口走，直到障碍、外扩或可走格；出图视为开口足够。"""
        dvec = self._opening_dir_xz(group)
        if dvec is None:
            return 0.0
        dx, dz = dvec
        start = self.world_to_cell(float(xyz[0]), float(xyz[2]))
        if start is None:
            start = (int(group[0][2]), int(group[0][3]))
        h, w = self.grid.shape
        max_steps = max(1, int(math.ceil(2.0 / self.grid_res)))
        unk_run = 0
        for i in range(1, max_steps + 1):
            rr = int(round(start[0] + dz * i))
            cc = int(round(start[1] + dx * i))
            if rr < 0 or cc < 0 or rr >= h or cc >= w:
                return max(unk_run * self.grid_res, OPENING_MIN_M)
            if self.inflate is not None and int(self.inflate[rr, cc]) != 0:
                break
            g = int(self.grid[rr, cc])
            if g == CELL_OCC or g == CELL_FREE:
                break
            if g == CELL_UNKNOWN:
                unk_run += 1
        return unk_run * self.grid_res

    def _split_cluster_pca(self, rows, cols, y, min_cluster=None):
        """PCA 切分过大簇，返回世界点列表（每块一组格子的质心用 cells 表示）。"""
        if min_cluster is None:
            min_cluster = CLUSTER_MIN
        cells_xz = []
        for r, c in zip(rows, cols):
            p = self.cell_to_world(int(r), int(c), y=y)
            cells_xz.append((float(p[0]), float(p[2]), int(r), int(c)))

        def split(group):
            if len(group) < min_cluster:
                return []
            mean_x = sum(g[0] for g in group) / len(group)
            mean_z = sum(g[1] for g in group) / len(group)
            if all(math.hypot(g[0] - mean_x, g[1] - mean_z) <= CLUSTER_SIZE_XY for g in group):
                return [group]
            cxx = czz = cxz = 0.0
            n = float(len(group))
            for g in group:
                dx, dz = g[0] - mean_x, g[1] - mean_z
                cxx += dx * dx
                czz += dz * dz
                cxz += dx * dz
            cxx /= n
            czz /= n
            cxz /= n
            cov = np.array([[cxx, cxz], [cxz, czz]], dtype=np.float64)
            vals, vecs = np.linalg.eigh(cov)
            pc = vecs[:, int(np.argmax(vals))]
            a, b = [], []
            for g in group:
                if (g[0] - mean_x) * pc[0] + (g[1] - mean_z) * pc[1] >= 0:
                    a.append(g)
                else:
                    b.append(g)
            if not a or not b:
                return [group]
            return split(a) + split(b)

        return split(cells_xz)

    def extract_frontiers(self, agent_xyz=None, pf=None, min_cluster=None,
                          max_n=None, node_yaw=None):
        """未知且四邻可走的连通块；门缝缩窄簇放宽开口与外扩边界。

        ``pf`` 保留形参，不再读导航网格。``geodesic_m`` 为占用图最短路径长。
        从机身搜不到有限路径的簇直接丢弃，不写哨兵距离。
        """
        del pf
        if min_cluster is None:
            min_cluster = CLUSTER_MIN_NECK
        if max_n is None:
            max_n = FRONTIER_MAX_N
        if self.grid is None:
            return []
        unk = self.grid == CELL_UNKNOWN
        free = self.grid == CELL_FREE
        n_free = np.zeros_like(free, dtype=bool)
        n_free[1:, :] |= free[:-1, :]
        n_free[:-1, :] |= free[1:, :]
        n_free[:, 1:] |= free[:, :-1]
        n_free[:, :-1] |= free[:, 1:]
        # 门缝喉部常落在外扩上：先收下 UNKNOWN∧邻 FREE，再分流过滤。
        mask = unk & n_free
        if not np.any(mask):
            return []
        labels, n_lab = _connected_components(mask, connectivity=8)
        clusters = []
        for lab in range(1, n_lab + 1):
            ys, xs = np.where(labels == lab)
            if len(ys) < min_cluster:
                continue
            clusters.append((ys, xs))
        agent = None if agent_xyz is None else np.asarray(agent_xyz, dtype=np.float64).reshape(3)
        y = 0.88 if agent is None else float(agent[1])
        pieces = []
        for ys, xs in clusters:
            pieces.extend(self._split_cluster_pca(ys, xs, y, min_cluster=min_cluster))
        pieces.sort(key=lambda g: -len(g))
        from nav.astar2d import astar_or_relax

        raw = []
        for i, group in enumerate(pieces):
            if i >= 48 and raw:
                break
            mx = sum(g[0] for g in group) / len(group)
            mz = sum(g[1] for g in group) / len(group)
            xyz = np.array([mx, y, mz], dtype=np.float64)
            is_neck = False
            if agent is not None:
                is_neck = self._path_has_door_neck(agent, xyz)
            if not is_neck:
                if len(group) < CLUSTER_MIN:
                    continue
                if self._opening_depth_m(group, xyz) < OPENING_MIN_M:
                    continue
                # 非门缝：质心所在格不得在外扩上（与旧掩膜一致）
                cc = self.world_to_cell(float(xyz[0]), float(xyz[2]))
                if (cc is not None and self.inflate is not None
                        and int(self.inflate[cc[0], cc[1]]) != 0):
                    # 簇内若全在外扩上则丢；否则用非外扩格重算质心
                    free_cells = [g for g in group
                                  if self.inflate is None
                                  or int(self.inflate[g[2], g[3]]) == 0]
                    if len(free_cells) < CLUSTER_MIN:
                        continue
                    group = free_cells
                    mx = sum(g[0] for g in group) / len(group)
                    mz = sum(g[1] for g in group) / len(group)
                    xyz = np.array([mx, y, mz], dtype=np.float64)
            if agent is not None:
                vis = self._visible_frontier_xyz(
                    agent, xyz, y, allow_inflate=bool(is_neck))
                if vis is None:
                    from nav.astar2d import nearest_walkable, SAFETY_OPTIMISTIC
                    vis = nearest_walkable(self, xyz[0], xyz[2], y=y, max_r=0.70,
                                           safety_mode=SAFETY_OPTIMISTIC)
                if vis is None:
                    continue
                xyz = np.asarray(vis, dtype=np.float64).reshape(3)
                euc = float(np.hypot(xyz[0] - agent[0], xyz[2] - agent[2]))
                if euc < FRONTIER_NEAR_DROP_M:
                    continue
                path, geo, _ = astar_or_relax(self, agent, xyz, success_dist=0.35)
                if path is None or not math.isfinite(geo):
                    continue
                yaw = math.atan2(xyz[0] - agent[0], xyz[2] - agent[2])
            else:
                geo = None
                yaw = 0.0
            raw.append({
                "n": len(group),
                "xyz": np.asarray(xyz, dtype=np.float64).tolist(),
                "world_yaw": float(yaw),
                "geodesic_m": None if geo is None else float(geo),
                "is_neck": bool(is_neck),
            })
        raw.sort(key=lambda t: -int(t["n"]))
        kept = []
        for it in raw:
            p = np.asarray(it["xyz"], dtype=np.float64)
            too_near = False
            for k in kept:
                q = np.asarray(k["xyz"], dtype=np.float64)
                if math.hypot(p[0] - q[0], p[2] - q[2]) < FRONTIER_NMS_M:
                    too_near = True
                    break
            if not too_near:
                kept.append(it)
        if agent is not None and node_yaw is not None:
            buckets = {int(d): [] for d in EVEN_PANO_INDICES}
            for it in kept:
                d = even_pano_dir(it["xyz"], agent, node_yaw)
                if d in buckets:
                    buckets[d].append(it)
            capped = []
            for d in EVEN_PANO_INDICES:
                bucket = buckets[d]
                if not bucket:
                    continue
                largest = max(bucket, key=lambda t: int(t["n"]))
                necks = [t for t in bucket if t.get("is_neck")]
                nearest_neck = None
                if necks:
                    nearest_neck = min(
                        necks,
                        key=lambda t: float(np.hypot(
                            t["xyz"][0] - agent[0], t["xyz"][2] - agent[2])))
                chosen = []
                if nearest_neck is not None:
                    chosen.append(nearest_neck)
                if largest is not nearest_neck:
                    chosen.append(largest)
                # 去重后最多 FRONTIER_PER_DIR
                capped.extend(chosen[:FRONTIER_PER_DIR])
            kept = capped
        kept.sort(key=lambda t: -int(t["n"]))
        out = []
        for it in kept[:max_n]:
            out.append({
                "fid": f"F{len(out)}",
                "xyz": it["xyz"],
                "world_yaw": it["world_yaw"],
                "geodesic_m": it["geodesic_m"],
                "is_neck": bool(it.get("is_neck")),
            })
        return out

    def leftover_by_dir(self, frontiers, node_xyz, node_yaw, even_dirs=EVEN_PANO_INDICES):
        """把探索点按扫描朝向分到偶序号方向。

        Args:
            frontiers (list): 提取结果。
            node_xyz: 节点位置。
            node_yaw (float): 扫描时机身朝向，弧度。
            even_dirs (sequence): 偶序号。

        Returns:
            list: ``{"dir": int, "n": int}``。
        """
        counts = {int(d): 0 for d in even_dirs}
        for fr in frontiers:
            even = even_pano_dir(fr["xyz"], node_xyz, node_yaw)
            if even in counts:
                counts[even] += 1
            else:
                counts[min(counts, key=lambda d: abs(d - even))] += 1
        return [{"dir": d, "n": counts[d]} for d in even_dirs if counts[d] > 0]

    def integrate_observation(self, rgb, depth, sensor_position, sensor_rotation,
                              skip_near_wall=None, body_y=None):
        """把一帧 RGB-D 融入全局点云。

        Args:
            rgb (np.ndarray): RGB 观测。
            depth (np.ndarray): 深度观测。
            sensor_position (np.ndarray): 相机世界坐标。
            sensor_rotation: 相机旋转，数组或 ``quaternion``。
            skip_near_wall (bool, optional): 覆盖构造时的贴墙跳过；低头扫地板时应关。
            body_y (float, optional): 机身高度，用于锁切片原点；缺省用相机高度减 0.88。

        Returns:
            int: 本帧通过高度切片后、下采样前的点数；跳过则为 0。
        """
        rgb = to_rgb_uint8(rgb)
        raw = np.asarray(depth, dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[:, :, 0]

        skip = self.skip_near_wall if skip_near_wall is None else bool(skip_near_wall)

        if skip and len(self.global_pcd.points) > 0:
            valid = (raw > 0.1) & (raw < 4.9)
            if np.any(valid) and float(np.mean(raw[valid] < 0.3)) > 0.30:
                return 0

        depth = clean_depth(raw, self.min_depth, self.max_depth)
        local_points, local_colors = get_pointcloud_from_depth(rgb, depth, self.intrinsic)
        if local_points.shape[0] == 0:
            return 0

        local_colors = np.asarray(local_colors)
        if local_colors.ndim == 3:
            local_colors = local_colors.reshape(-1, 3)
        n = min(len(local_points), len(local_colors))
        local_points = np.asarray(local_points)[:n]
        local_colors = local_colors[:n]
        if n == 0:
            return 0

        position = np.asarray(sensor_position, dtype=np.float64).reshape(3)
        rotation = quaternion.as_rotation_matrix(as_quaternion(sensor_rotation))
        world_points = translate_to_world(local_points, position, rotation)

        if self.initial_height is None:
            if body_y is not None:
                self.initial_height = float(body_y)
            else:
                self.initial_height = float(position[1]) - 0.88
        lo = self.initial_height - self.height_low
        hi = self.initial_height + self.height_high
        height_mask = (world_points[:, 1] >= lo) & (world_points[:, 1] <= hi)
        world_points = world_points[height_mask]
        local_colors = local_colors[height_mask]
        if len(world_points) == 0:
            return 0

        occ_lo = self.initial_height + OBSTACLE_H_LO
        occ_hi = self.initial_height + OBSTACLE_H_HI
        occ_mask = (world_points[:, 1] >= occ_lo) & (world_points[:, 1] <= occ_hi)
        occ_points = world_points[occ_mask]

        frame_pcd = o3d.geometry.PointCloud()
        frame_pcd.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(world_points.astype(np.float64)))
        frame_pcd.colors = o3d.utility.Vector3dVector(
            np.ascontiguousarray(local_colors.astype(np.float64) / 255.0))
        frame_down = frame_pcd.voxel_down_sample(voxel_size=self.voxel_size)

        if len(self.global_pcd.points) == 0:
            self.global_pcd = frame_down
        else:
            combined = o3d.geometry.PointCloud()
            combined.points = o3d.utility.Vector3dVector(np.vstack([
                np.asarray(self.global_pcd.points), np.asarray(frame_down.points)]))
            combined.colors = o3d.utility.Vector3dVector(np.vstack([
                np.asarray(self.global_pcd.colors), np.asarray(frame_down.colors)]))
            self.global_pcd = combined.voxel_down_sample(voxel_size=self.voxel_size)
        if len(occ_points) > 0:
            self.sculpt_hits(position, occ_points)
        if body_y is not None:
            self.stamp_agent_free(np.array([position[0], float(body_y), position[2]]))
        else:
            self.stamp_agent_free(position)
        return int(len(world_points))

    def integrate_from_env(self, env, skip_near_wall=None):
        """从当前 env 观测融入一帧。

        Args:
            env: ``habitat.Env``。
            skip_near_wall (bool, optional): 见 ``integrate_observation``。

        Returns:
            int: 见 ``integrate_observation``。
        """
        obs = env.sim.get_sensor_observations()
        pos, rot = sensor_pose(env)
        return self.integrate_observation(obs["rgb"], obs["depth"], pos, rot,
                                          skip_near_wall=skip_near_wall,
                                          body_y=float(body_position(env)[1]))

    def scan_around(self, env, rotate_times=12, look_down_floor=True, mark_node=True):
        """原地右转一圈采集全景，并（可选）低头再扫一圈地面点云。

        先存当前朝向再转动，因此序号 0 就是未转动的当前观测。一圈结束后机身 yaw
        回到起点；低头扫描只改 pitch，扫完会抬回平视。

        Args:
            env: ``habitat.Env``。
            rotate_times (int): 环视张数，默认 12（每步 30°）。
            look_down_floor (bool): 是否再低头扫地面，补相机近处盲区。
            mark_node (bool): 是否把本次位置记为扫描节点。

        Returns:
            list: ``rotate_times`` 张 RGB uint8。
        """
        images = []
        obs = env.sim.get_sensor_observations()
        for _ in range(rotate_times):
            if env.episode_over:
                break
            images.append(to_rgb_uint8(obs["rgb"]))
            self.integrate_from_env(env)
            obs = env.step(HAB_RIGHT)

        if look_down_floor and not env.episode_over:
            env.step(HAB_LOOK_DOWN)
            env.step(HAB_LOOK_DOWN)
            for _ in range(rotate_times):
                if env.episode_over:
                    break
                self.integrate_from_env(env, skip_near_wall=False)
                obs = env.step(HAB_RIGHT)
            if not env.episode_over:
                env.step(HAB_LOOK_UP)
                env.step(HAB_LOOK_UP)
            self.integrate_from_env(env, skip_near_wall=False)

        pos = body_position(env)
        self.record_body(pos)
        if mark_node:
            self.mark_scan_node(pos)
        return images

    def render_occ_debug(self, agent_xyz, frontiers=None):
        """黑白灰占用栅格 + 绿色前沿点 + 红虚线 A* 路径（一格一像素）。

        Args:
            agent_xyz: 机身世界坐标。
            frontiers (list, optional): ``extract_frontiers`` 结果；缺省只画栅格与机身。

        Returns:
            np.ndarray: ``(H, W, 3)`` RGB uint8；无栅格时返回小黑图。
        """
        if self.grid is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        gh, gw = self.grid.shape
        palette = np.array([
            GRID_UNKNOWN_RGB,  # 0 UNKNOWN
            GRID_FREE_RGB,     # 1 FREE
            GRID_OCC_RGB,      # 2 OCC
        ], dtype=np.uint8)
        idx = np.clip(self.grid.astype(np.int32), 0, 2)
        bev = palette[idx]
        agent = np.asarray(agent_xyz, dtype=np.float64).reshape(3)
        ac = self.world_to_cell(agent[0], agent[2])
        if ac is not None:
            ar, acols = int(ac[0]), int(ac[1])
            cv2.circle(bev, (acols, ar), 3, OCC_AGENT_RGB, -1, cv2.LINE_AA)

        from nav.astar2d import astar_or_relax

        for fr in frontiers or []:
            tgt = np.asarray(fr["xyz"], dtype=np.float64).reshape(3)
            path, _geo, _ = astar_or_relax(self, agent, tgt, success_dist=0.35)
            if path is not None and len(path) > 1:
                pts = []
                for xz in path:
                    cell = self.world_to_cell(float(xz[0]), float(xz[1]))
                    if cell is not None:
                        pts.append((int(cell[1]), int(cell[0])))  # (col, row)
                _draw_dashed_polyline(bev, pts, OCC_ASTAR_DASH_RGB, thickness=1,
                                      dash=5, gap=4)
            fc = self.world_to_cell(float(tgt[0]), float(tgt[2]))
            if fc is not None:
                px, py = int(fc[1]), int(fc[0])
                cv2.circle(bev, (px, py), 3, FRONTIER_COLOR, -1, cv2.LINE_AA)
                fid = str(fr.get("fid", "F"))
                cv2.putText(bev, fid, (px + 4, py - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, FRONTIER_COLOR, 1, cv2.LINE_AA)
        return bev

    def render_bev(self, agent_position, agent_rotation, color_mode="color",
                   resolution=0.02, point_size=5, sector_labels=None,
                   rotate_times=12, frontiers=None, node_overlays=None, edges=None,
                   paint_grid=False, path_xyz=None):
        """把全局点云投到 XZ 平面，叠扫描节点、历史轨迹和当前朝向。

        Args:
            agent_position: 机身世界坐标。
            agent_rotation: 机身旋转。
            color_mode (str): ``color`` 用点云 RGB；``bw`` 占用点画成白、底为黑。
            resolution (float): 每像素对应的米数。
            point_size (int): 占用点半径，像素。
            sector_labels (sequence, optional): 画在 agent 周围的方向序号。
            rotate_times (int): 一圈的朝向总数。
            frontiers (list, optional): ``{"xyz", "fid"}`` 绿点。
            node_overlays (list, optional): ``{"xyz", "node_id"}``；给定则用它画节点号。
            edges: 已忽略（保留形参兼容调用方）；BEV 只画实际轨迹。
            paint_grid (bool): 先铺三值栅格底色。

        Returns:
            np.ndarray: RGB BEV 图。
        """
        points = np.asarray(self.global_pcd.points) if len(self.global_pcd.points) else np.zeros((0, 3))
        colors = np.asarray(self.global_pcd.colors) if len(self.global_pcd.points) else np.zeros((0, 3))

        extras = []
        if self.scan_nodes:
            extras.append(np.stack(self.scan_nodes, axis=0))
        if self.path_positions:
            extras.append(np.stack(self.path_positions, axis=0))
        extras.append(np.asarray(agent_position, dtype=np.float64).reshape(1, 3))
        if frontiers:
            extras.append(np.array([fr["xyz"] for fr in frontiers], dtype=np.float64))
        if node_overlays:
            extras.append(np.array([n["xyz"] for n in node_overlays], dtype=np.float64))
        if self.grid is not None:
            h0, w0 = self.grid.shape
            extras.append(np.array([
                [self._ox, 0.0, self._oz],
                [self._ox + w0 * self.grid_res, 0.0, self._oz + h0 * self.grid_res],
            ], dtype=np.float64))
        bound_pts = np.vstack([points, *extras]) if len(points) else np.vstack(extras)

        min_x = float(bound_pts[:, 0].min() - 1.0)
        max_x = float(bound_pts[:, 0].max() + 1.0)
        min_z = float(bound_pts[:, 2].min() - 1.0)
        max_z = float(bound_pts[:, 2].max() + 1.0)
        width_pixels = int(np.clip(np.ceil((max_x - min_x) / resolution), 128, 2048))
        height_pixels = int(np.clip(np.ceil((max_z - min_z) / resolution), 128, 2048))
        bev = np.zeros((height_pixels, width_pixels, 3), dtype=np.uint8)

        def to_px(xyz):
            """世界 ``(x, z)`` -> 像素 ``(col, row)``。"""
            xs = np.floor((xyz[..., 0] - min_x) / resolution).astype(int)
            ys = np.floor((xyz[..., 2] - min_z) / resolution).astype(int)
            return np.clip(xs, 0, width_pixels - 1), np.clip(ys, 0, height_pixels - 1)

        if paint_grid and self.grid is not None:
            gh, gw = self.grid.shape
            palette = {
                CELL_UNKNOWN: GRID_UNKNOWN_RGB,
                CELL_FREE: GRID_FREE_RGB,
                CELL_OCC: GRID_OCC_RGB,
            }
            for r in range(gh):
                for c in range(gw):
                    color = palette.get(int(self.grid[r, c]), GRID_UNKNOWN_RGB)
                    if (self.inflate is not None and self.inflate[r, c]
                            and int(self.grid[r, c]) != CELL_OCC):
                        color = GRID_INFLATE_RGB
                    p0 = np.array([[self._ox + c * self.grid_res, 0.0,
                                    self._oz + r * self.grid_res]])
                    p1 = np.array([[self._ox + (c + 1) * self.grid_res, 0.0,
                                    self._oz + (r + 1) * self.grid_res]])
                    x0, y0 = to_px(p0)
                    x1, y1 = to_px(p1)
                    cv2.rectangle(bev, (int(x0[0]), int(y0[0])),
                                  (int(x1[0]), int(y1[0])), color, -1)

        if len(points) > 0:
            grid_x, grid_y = to_px(points)
            order = np.argsort(points[:, 1])
            grid_x, grid_y = grid_x[order], grid_y[order]
            colors_sorted = colors[order]
            combined = grid_x.astype(np.int64) + grid_y.astype(np.int64) * 1000000
            _, uniq = np.unique(combined, return_index=True)
            uniq = np.sort(uniq)
            fx, fy = grid_x[uniq], grid_y[uniq]
            if color_mode in ("bw", "mono", "gray", "white"):
                paint = np.full((len(fx), 3), 255, dtype=np.uint8)
            else:
                paint = (np.clip(colors_sorted[uniq], 0, 1) * 255).astype(np.uint8)
            for i in range(len(fx)):
                cv2.circle(bev, (int(fx[i]), int(fy[i])), int(point_size),
                           paint[i].tolist(), -1)

        if len(self.path_positions) > 1:
            traj = np.stack(self.path_positions, axis=0)
            tx, ty = to_px(traj)
            for i in range(len(tx) - 1):
                p1, p2 = (int(tx[i]), int(ty[i])), (int(tx[i + 1]), int(ty[i + 1]))
                if p1 != p2:
                    cv2.line(bev, p1, p2, SCAN_PATH_COLOR, thickness=1, lineType=cv2.LINE_AA)
        if path_xyz is not None and len(path_xyz) > 1:
            path = np.asarray(path_xyz, dtype=np.float64)
            if path.shape[1] == 2:
                path3 = np.stack([path[:, 0], np.zeros(len(path)), path[:, 1]], axis=1)
            else:
                path3 = path
            px, py = to_px(path3)
            for i in range(len(px) - 1):
                p1, p2 = (int(px[i]), int(py[i])), (int(px[i + 1]), int(py[i + 1]))
                if p1 != p2:
                    cv2.line(bev, p1, p2, ASTAR_PATH_COLOR, thickness=2, lineType=cv2.LINE_AA)

        agent_pos = np.asarray(agent_position, dtype=np.float64).reshape(3)
        ax, ay = to_px(agent_pos.reshape(1, 3))
        agent_pixel = np.array([int(ax[0]), int(ay[0])], dtype=np.float64)
        rot = quaternion.as_rotation_matrix(as_quaternion(agent_rotation))
        raw_forward = -rot[:, 2]
        forward_vec = np.array([raw_forward[0], raw_forward[2]], dtype=np.float64)
        nrm = np.linalg.norm(forward_vec)
        forward_vec = forward_vec / nrm if nrm > 1e-6 else np.array([0.0, -1.0])
        right_vec = np.array([forward_vec[1], -forward_vec[0]])

        triangle_size, triangle_width = 20.0, 10.0
        if sector_labels is None:
            sector_labels = list(DEFAULT_PANO_INDICES)
        sector_labels = list(sector_labels)
        n_dir = max(int(rotate_times), 1)
        deg_per = 360.0 / n_dir
        if sector_labels:
            overlay = bev.copy()
            ray_length = triangle_size * 4
            text_distance = triangle_size * 2.8
            # 数字在扇区中心；分割线在相邻扇区边界（6 个偶序号时即中心 ±30°）。
            half = np.deg2rad(180.0 / max(len(sector_labels), 1))
            bound_angs = []
            for label in sector_labels:
                center = np.deg2rad(float(label) * deg_per)
                bound_angs.extend((center - half, center + half))
            uniq = []
            for ang in bound_angs:
                ang = ang % (2.0 * np.pi)
                if not any(abs((ang - u + np.pi) % (2.0 * np.pi) - np.pi) < 1e-3 for u in uniq):
                    uniq.append(ang)
            for ang in uniq:
                c, s = np.cos(ang), np.sin(ang)
                ray_dir = np.array([forward_vec[0] * c - forward_vec[1] * s,
                                    forward_vec[0] * s + forward_vec[1] * c])
                end = agent_pixel + ray_dir * ray_length
                cv2.line(overlay, (int(agent_pixel[0]), int(agent_pixel[1])),
                         (int(end[0]), int(end[1])), SECTOR_RAY_COLOR, 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.5, bev, 0.5, 0, bev)
            for label in sector_labels:
                ang = np.deg2rad(float(label) * deg_per)
                c, s = np.cos(ang), np.sin(ang)
                text_dir = np.array([forward_vec[0] * c - forward_vec[1] * s,
                                     forward_vec[0] * s + forward_vec[1] * c])
                text_pos = agent_pixel + text_dir * text_distance
                text = str(label)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(bev, text,
                            (int(text_pos[0] - tw / 2), int(text_pos[1] + th / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, SECTOR_TEXT_COLOR, 2, cv2.LINE_AA)

        tip = agent_pixel + forward_vec * triangle_size
        left_pt = agent_pixel - forward_vec * (triangle_size * 0.4) + right_vec * triangle_width
        right_pt = agent_pixel - forward_vec * (triangle_size * 0.4) - right_vec * triangle_width
        triangle = np.array([tip, left_pt, right_pt], dtype=np.int32)
        cv2.fillConvexPoly(bev, triangle, AGENT_COLOR)

        # 节点号与扇区红字同一字号（scale=1, thickness=2），不画节点间连线。
        node_r = max(8, int(min(height_pixels, width_pixels) / 80))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 1.0, 2
        labeled = node_overlays if node_overlays else None
        if labeled:
            items = [(n["xyz"], n["node_id"]) for n in labeled]
        else:
            items = [(node, i) for i, node in enumerate(self.scan_nodes)]
        for xyz, nid in items:
            node = np.asarray(xyz, dtype=np.float64).reshape(3)
            if np.hypot(float(node[0] - agent_pos[0]), float(node[2] - agent_pos[2])) < 0.20:
                continue
            cx, cy = to_px(node.reshape(1, 3))
            px, py = int(cx[0]), int(cy[0])
            cv2.circle(bev, (px, py), node_r, SCAN_NODE_COLOR, 2, cv2.LINE_AA)
            text = str(nid)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.putText(bev, text, (px - tw // 2, py - node_r - 4),
                        font, font_scale, SCAN_NODE_COLOR, thickness, cv2.LINE_AA)

        if frontiers:
            for fr in frontiers:
                p = np.asarray(fr["xyz"], dtype=np.float64).reshape(1, 3)
                cx, cy = to_px(p)
                px, py = int(cx[0]), int(cy[0])
                cv2.circle(bev, (px, py), 5, FRONTIER_COLOR, -1, cv2.LINE_AA)
                fid = str(fr.get("fid", "F"))
                cv2.putText(bev, fid, (px + 6, py - 4), font, 0.45, FRONTIER_COLOR, 1, cv2.LINE_AA)
        return bev

    def render_bev_from_env(self, env, **kwargs):
        """按当前机身位姿渲染 BEV。

        Args:
            env: ``habitat.Env``。
            **kwargs: 传给 ``render_bev``。

        Returns:
            np.ndarray: RGB BEV 图。
        """
        state = env.sim.get_agent_state()
        return self.render_bev(state.position, state.rotation, **kwargs)
