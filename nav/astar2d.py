"""自建 2D occupancy 上的 Habitat 风格 A*（12 向 × 0.25 m）。

对齐 ApexNav ``astar2d.cpp``：节点按 ``grid_res`` 量化；碰撞看 inflate / OCC /
（NORMAL 下）UNKNOWN；起点 0.25 m 内放松，避免脚下膨胀把搜索掐死。
"""

import heapq
import math

import numpy as np

from nav.occupancy import CELL_OCC, CELL_UNKNOWN

SAFETY_NORMAL = 0
SAFETY_OPTIMISTIC = 1
STEP_M = 0.25
N_DIR = 12
SUCCESS_DIST = 0.25
START_RELAX_M = 0.25
RAY_DS = 0.025
LAMBDA_HEU = 1.0


def _logit_heu(dx, dz):
    """对角启发，带极小 tie-break。"""
    adx, adz = abs(dx), abs(dz)
    return (1.0 + 1e-6 * (adx + adz)) * (math.sqrt(2.0) * min(adx, adz) + abs(adx - adz))


def generate_steps():
    """12 个 0.25 m 位移 ``(dx, dz)``。"""
    out = []
    for i in range(N_DIR):
        ang = i * (math.pi / 6.0)
        out.append((STEP_M * math.cos(ang), STEP_M * math.sin(ang)))
    return out


STEPS = generate_steps()


def path_length_xz(path):
    """折线长度，米。"""
    if path is None or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
    return total


def cell_walkable(occ, row, col, safety_mode=SAFETY_NORMAL):
    """格子是否可走。越界不可走。"""
    if occ.grid is None:
        return False
    h, w = occ.grid.shape
    if row < 0 or col < 0 or row >= h or col >= w:
        return False
    if occ.inflate is not None and occ.inflate[row, col] != 0:
        return False
    state = int(occ.grid[row, col])
    if state == CELL_OCC:
        return False
    if state == CELL_UNKNOWN and safety_mode == SAFETY_NORMAL:
        return False
    return True


def point_walkable(occ, x, z, safety_mode=SAFETY_NORMAL):
    """世界 XZ 是否可走。"""
    cell = occ.world_to_cell(x, z)
    if cell is None:
        return False
    return cell_walkable(occ, cell[0], cell[1], safety_mode)


def nearest_walkable(occ, x, z, y=0.88, max_r=0.85, safety_mode=SAFETY_OPTIMISTIC):
    """在半径内找最近可走格中心。"""
    if occ.grid is None:
        return None
    cell = occ.world_to_cell(x, z)
    if cell is not None and cell_walkable(occ, cell[0], cell[1], safety_mode):
        return occ.cell_to_world(cell[0], cell[1], y=y)
    res = occ.grid_res
    n = max(1, int(math.ceil(max_r / res)))
    h, w = occ.grid.shape
    best, best_d = None, float("inf")
    cr = None if cell is None else cell[0]
    cc = None if cell is None else cell[1]
    if cr is None:
        cr = int(math.floor((float(z) - occ._oz) / res))
        cc = int(math.floor((float(x) - occ._ox) / res))
    for dr in range(-n, n + 1):
        for dc in range(-n, n + 1):
            r, c = cr + dr, cc + dc
            if not cell_walkable(occ, r, c, safety_mode):
                continue
            pt = occ.cell_to_world(r, c, y=y)
            d = math.hypot(pt[0] - x, pt[2] - z)
            if d < best_d and d <= max_r + 1e-6:
                best, best_d = pt, d
    return best


def astar_search(occ, start_xyz, goal_xyz, success_dist=SUCCESS_DIST,
                 safety_mode=SAFETY_NORMAL, max_iters=60000):
    """从起点搜到距目标 ``success_dist`` 内。

    Args:
        occ: ``OccupancyMap``。
        start_xyz: 起点世界坐标。
        goal_xyz: 终点世界坐标。
        success_dist (float): 到达半径，米。
        safety_mode (int): ``SAFETY_NORMAL`` 或 ``SAFETY_OPTIMISTIC``。
        max_iters (int): 扩展上限。

    Returns:
        tuple: ``(path_xz list[(x,z)] | None, length_m)``。
    """
    if occ.grid is None:
        return None, float("inf")
    start = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    goal = np.asarray(goal_xyz, dtype=np.float64).reshape(3)
    sx, sz = float(start[0]), float(start[2])
    gx, gz = float(goal[0]), float(goal[2])
    res = occ.grid_res
    origin_x, origin_z = occ._ox, occ._oz

    def to_idx(x, z):
        return (int(math.floor((z - origin_z) / res)),
                int(math.floor((x - origin_x) / res)))

    start_idx = to_idx(sx, sz)
    open_heap = []
    g_score = {start_idx: 0.0}
    parent = {start_idx: None}
    pos_of = {start_idx: (sx, sz)}
    heapq.heappush(open_heap, (_logit_heu(gx - sx, gz - sz), 0.0, start_idx))
    closed = set()
    counter = 0
    best_idx = start_idx

    while open_heap and counter < max_iters:
        f, g, idx = heapq.heappop(open_heap)
        if idx in closed:
            continue
        closed.add(idx)
        cx, cz = pos_of[idx]
        if math.hypot(cx - gx, cz - gz) < success_dist:
            best_idx = idx
            break
        if abs(idx[0] - to_idx(gx, gz)[0]) <= 1 and abs(idx[1] - to_idx(gx, gz)[1]) <= 1:
            if math.hypot(cx - gx, cz - gz) < max(success_dist, res * 1.5):
                best_idx = idx
                break
        counter += 1
        for dx, dz in STEPS:
            nx, nz = cx + dx, cz + dz
            nidx = to_idx(nx, nz)
            if nidx in closed:
                continue
            dist_start = math.hypot(nx - sx, nz - sz)
            occ_hit = False
            cell = occ.world_to_cell(nx, nz)
            if cell is not None and int(occ.grid[cell[0], cell[1]]) == CELL_OCC:
                occ_hit = True
            if occ_hit:
                continue
            if dist_start > START_RELAX_M:
                if not point_walkable(occ, nx, nz, safety_mode):
                    continue
                safe = True
                length = math.hypot(dx, dz)
                nrm = length if length > 1e-9 else 1.0
                ux, uz = dx / nrm, dz / nrm
                l = RAY_DS
                while l < length:
                    if not point_walkable(occ, cx + l * ux, cz + l * uz, safety_mode):
                        safe = False
                        break
                    l += RAY_DS
                if not safe:
                    continue
            ng = g + STEP_M
            if ng >= g_score.get(nidx, float("inf")):
                continue
            g_score[nidx] = ng
            parent[nidx] = idx
            pos_of[nidx] = (nx, nz)
            heapq.heappush(open_heap, (ng + LAMBDA_HEU * _logit_heu(gx - nx, gz - nz), ng, nidx))
            best_idx = nidx
    else:
        if math.hypot(pos_of[best_idx][0] - gx, pos_of[best_idx][1] - gz) >= success_dist:
            return None, float("inf")

    if math.hypot(pos_of[best_idx][0] - gx, pos_of[best_idx][1] - gz) >= success_dist:
        return None, float("inf")

    chain = []
    cur = best_idx
    while cur is not None:
        chain.append(pos_of[cur])
        cur = parent.get(cur)
    chain.reverse()
    chain.append((gx, gz))
    return chain, path_length_xz(chain)


def astar_or_relax(occ, start_xyz, goal_xyz, success_dist=SUCCESS_DIST):
    """先 NORMAL，失败再 OPTIMISTIC；仍失败则把终点吸附到可走格再搜。"""
    path, length = astar_search(occ, start_xyz, goal_xyz, success_dist=success_dist,
                                safety_mode=SAFETY_NORMAL)
    if path is not None:
        return path, length, SAFETY_NORMAL
    path, length = astar_search(occ, start_xyz, goal_xyz, success_dist=success_dist,
                                safety_mode=SAFETY_OPTIMISTIC)
    if path is not None:
        return path, length, SAFETY_OPTIMISTIC
    y = float(np.asarray(goal_xyz, dtype=np.float64).reshape(3)[1])
    gx, gz = float(goal_xyz[0]), float(goal_xyz[2])
    snapped = nearest_walkable(occ, gx, gz, y=y, max_r=0.85, safety_mode=SAFETY_OPTIMISTIC)
    if snapped is None:
        return None, float("inf"), SAFETY_OPTIMISTIC
    path, length = astar_search(occ, start_xyz, snapped, success_dist=success_dist,
                                safety_mode=SAFETY_OPTIMISTIC)
    return path, length, SAFETY_OPTIMISTIC
