"""不可达探索点在提取与选点阶段被丢掉；门缝三角形可抽出。"""

import math
import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness.overlay import frontier_candidates, path_geodesic_ok, pick_mover_candidate
from nav.goto import occupancy_path_length, pursue_occupancy
from nav.occupancy import (CELL_FREE, CELL_OCC, CELL_UNKNOWN, OccupancyMap,
                           UNKNOWN_LOG)


def _paint_map():
    """左侧可走开口，右侧墙后口袋不可达。"""
    occ = OccupancyMap(np.eye(3, dtype=np.float32), grid_res=0.10)
    occ._ox, occ._oz = -1.0, -1.0
    h, w = 80, 80
    occ.grid = np.zeros((h, w), dtype=np.uint8)
    occ.occupancy_log = np.full((h, w), UNKNOWN_LOG, dtype=np.float32)
    occ.inflate = np.zeros((h, w), dtype=np.uint8)
    occ.initial_height = 0.88
    occ.grid[10:41, 10:41] = CELL_FREE
    occ.grid[41:58, 20:32] = CELL_UNKNOWN
    occ.grid[10:51, 50:54] = CELL_OCC
    occ.grid[10:41, 54:72] = CELL_FREE
    occ.grid[41:58, 56:68] = CELL_UNKNOWN
    occ._refresh_inflate(0, 0, h - 1, w - 1)
    return occ


def _paint_door_glimpse():
    """房间 + 门缝 + 门后一小块可走与短未知边。"""
    occ = OccupancyMap(np.eye(3, dtype=np.float32), grid_res=0.10)
    occ._ox, occ._oz = -1.0, -1.0
    h, w = 80, 80
    # 外缘填 OCC，避免整图 UNKNOWN 岸成为探索点
    occ.grid = np.full((h, w), CELL_OCC, dtype=np.uint8)
    occ.occupancy_log = np.full((h, w), UNKNOWN_LOG, dtype=np.float32)
    occ.inflate = np.zeros((h, w), dtype=np.uint8)
    occ.initial_height = 0.88
    # 本侧房间
    occ.grid[20:50, 8:28] = CELL_FREE
    # 门墙：中间留约 0.7 m 缝
    occ.grid[20:32, 28:31] = CELL_OCC
    occ.grid[39:50, 28:31] = CELL_OCC
    # 门缝可走
    occ.grid[32:39, 28:31] = CELL_FREE
    # 门后一小块可走
    occ.grid[33:38, 31:36] = CELL_FREE
    # 门后短未知边（少于旧 CLUSTER_MIN=8）
    occ.grid[34:37, 36:38] = CELL_UNKNOWN
    # 门后其余仍为 OCC（已由全图 OCC 覆盖）
    occ._refresh_inflate(0, 0, h - 1, w - 1)
    return occ


class _Sim:
    """只提供机身位姿。"""

    def __init__(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)

    def get_agent_state(self, *_a, **_k):
        st = type("S", (), {})()
        st.position = self._pos
        st.rotation = np.array([0.0, 0.0, 0.0, 1.0])
        return st


class _Env:
    """假环境，记录是否步进。"""

    episode_over = False

    def __init__(self, pos):
        self.sim = _Sim(pos)
        self.n_step = 0

    def step(self, _action):
        self.n_step += 1
        return {}


class TestPathGeodesic(unittest.TestCase):
    """路径长闸门。"""

    def test_ok_and_sentinel(self):
        """有限路径通过；无穷与 1000 哨兵拒绝。"""
        self.assertTrue(path_geodesic_ok(3.2))
        self.assertFalse(path_geodesic_ok(None))
        self.assertFalse(path_geodesic_ok(float("inf")))
        self.assertFalse(path_geodesic_ok(1013.15))


class TestPickFrontier(unittest.TestCase):
    """探索选点忽略不可达。"""

    def test_skips_sentinel_picks_far_walkable(self):
        """哨兵点不参与最远比较。"""
        cands = [
            {"id": "F0", "geodesic_m": 1013.15, "xyz": [9.0, 0.88, 0.0]},
            {"id": "F1", "geodesic_m": 2.0, "xyz": [1.0, 0.88, 0.0]},
            {"id": "F2", "geodesic_m": 5.0, "xyz": [2.0, 0.88, 0.0]},
        ]
        got = pick_mover_candidate("frontier", cands)
        self.assertEqual(got["id"], "F2")

    def test_all_bad_returns_none(self):
        """全不可达则没有候选。"""
        cands = [
            {"id": "F0", "geodesic_m": float("inf")},
            {"id": "F1", "geodesic_m": 1000.5},
        ]
        self.assertIsNone(pick_mover_candidate("frontier", cands))


class TestExtractDrop(unittest.TestCase):
    """提取丢掉墙后口袋。"""

    def test_no_point_behind_wall(self):
        """墙后未知簇不进绿点；可走开口仍在。"""
        occ = _paint_map()
        agent = occ.cell_to_world(25, 25, y=0.88)
        fronts = occ.extract_frontiers(agent_xyz=agent)
        self.assertTrue(fronts)
        for fr in fronts:
            self.assertTrue(path_geodesic_ok(fr.get("geodesic_m")))
            self.assertLess(float(fr["xyz"][0]), 4.0)


class TestDoorGlimpse(unittest.TestCase):
    """侧视门缝后一小块地板仍能抽出探索点。"""

    def test_extracts_beyond_door_neck(self):
        """路径经门宽缩窄时，门后短未知边仍保留。"""
        occ = _paint_door_glimpse()
        agent = occ.cell_to_world(35, 18, y=0.88)
        unk = occ.cell_to_world(35, 37, y=0.88)
        self.assertTrue(occ._path_has_door_neck(agent, unk))
        fronts = occ.extract_frontiers(agent_xyz=agent)
        self.assertTrue(fronts)
        beyond = [fr for fr in fronts if float(fr["xyz"][0]) > 2.0]
        self.assertTrue(beyond, msg=f"fronts={fronts}")
        for fr in beyond:
            self.assertTrue(path_geodesic_ok(fr.get("geodesic_m")))


class TestPursueAbort(unittest.TestCase):
    """跟随第一步无路径则不步进。"""

    def test_first_astar_fail_no_step(self):
        """两块互不连通的可走岛：零步返回且环境未 step。"""
        occ = OccupancyMap(np.eye(3, dtype=np.float32), grid_res=0.10)
        occ._ox, occ._oz = -1.0, -1.0
        h, w = 80, 80
        occ.grid = np.full((h, w), CELL_OCC, dtype=np.uint8)
        occ.occupancy_log = np.full((h, w), UNKNOWN_LOG, dtype=np.float32)
        occ.inflate = np.zeros((h, w), dtype=np.uint8)
        occ.initial_height = 0.88
        occ.grid[20:36, 20:36] = CELL_FREE
        occ.grid[20:36, 55:71] = CELL_FREE
        occ._refresh_inflate(0, 0, h - 1, w - 1)
        agent = occ.cell_to_world(27, 27, y=0.88)
        goal = occ.cell_to_world(27, 62, y=0.88)
        env = _Env(agent)
        out = pursue_occupancy(env, occ, goal, max_steps=30, success_dist=0.35)
        self.assertEqual(out["steps"], 0)
        self.assertFalse(out["arrived"])
        self.assertEqual(env.n_step, 0)


class TestStopGeodesic(unittest.TestCase):
    """到达用占用图测地，不用评测距离。"""

    def test_near_foothold_finite_path(self):
        """同房间 0.8 m 落脚：路径有限且短于 1.0 m。"""
        occ = OccupancyMap(np.eye(3, dtype=np.float32), grid_res=0.10)
        occ._ox, occ._oz = -1.0, -1.0
        h, w = 40, 40
        occ.grid = np.full((h, w), CELL_FREE, dtype=np.uint8)
        occ.occupancy_log = np.full((h, w), UNKNOWN_LOG, dtype=np.float32)
        occ.inflate = np.zeros((h, w), dtype=np.uint8)
        occ.initial_height = 0.88
        agent = occ.cell_to_world(20, 20, y=0.88)
        goal = occ.cell_to_world(20, 28, y=0.88)  # 0.8 m
        geo = occupancy_path_length(occ, agent, goal, success_dist=0.35)
        self.assertTrue(path_geodesic_ok(geo))
        self.assertLessEqual(float(geo), 1.0 + 1e-3)

    def test_through_wall_rejected(self):
        """隔墙欧氏近但路径不通。"""
        occ = OccupancyMap(np.eye(3, dtype=np.float32), grid_res=0.10)
        occ._ox, occ._oz = -1.0, -1.0
        h, w = 40, 40
        occ.grid = np.full((h, w), CELL_FREE, dtype=np.uint8)
        occ.occupancy_log = np.full((h, w), UNKNOWN_LOG, dtype=np.float32)
        occ.inflate = np.zeros((h, w), dtype=np.uint8)
        occ.initial_height = 0.88
        occ.grid[5:35, 20:22] = CELL_OCC
        occ._refresh_inflate(0, 0, h - 1, w - 1)
        agent = occ.cell_to_world(20, 10, y=0.88)
        goal = occ.cell_to_world(20, 24, y=0.88)
        geo = occupancy_path_length(occ, agent, goal, success_dist=0.35)
        self.assertFalse(path_geodesic_ok(geo) and float(geo) <= 1.0)
        self.assertFalse(occ.grid_line_clear(agent, goal))


class TestNearFrontierDraw(unittest.TestCase):
    """测地近且绕墙可达的点仍进候选并投影，不再要求直线通视。"""

    def test_occluded_but_walkable_under_5m_drawn(self):
        """直线穿墙但绕行测地约 3 m：应投影出绿点。"""
        occ = OccupancyMap(np.eye(3, dtype=np.float32), grid_res=0.10)
        occ._ox, occ._oz = -1.0, -1.0
        h, w = 60, 60
        occ.grid = np.full((h, w), CELL_OCC, dtype=np.uint8)
        occ.occupancy_log = np.full((h, w), UNKNOWN_LOG, dtype=np.float32)
        occ.inflate = np.zeros((h, w), dtype=np.uint8)
        occ.initial_height = 0.88
        # 走廊：先向右再向上，终点在墙北侧
        occ.grid[20:40, 10:40] = CELL_FREE
        occ.grid[40:50, 30:40] = CELL_FREE
        # 竖墙挡直线视线（中间留出绕行通道已在右侧）
        occ.grid[35:45, 15:28] = CELL_OCC
        occ._refresh_inflate(0, 0, h - 1, w - 1)
        agent = occ.cell_to_world(30, 15, y=0.88)
        goal = occ.cell_to_world(45, 35, y=0.88)
        geo = occupancy_path_length(occ, agent, goal, success_dist=0.35)
        self.assertTrue(path_geodesic_ok(geo))
        self.assertLess(float(geo), 5.0)
        self.assertFalse(occ.grid_line_clear(agent, goal))
        # 相机朝 +z，目标在前方
        intrinsic = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        sensor_pos = np.array([agent[0], 1.76, agent[2]], dtype=np.float64)
        # 单位四元数：相机前方为 -z（habitat）；把目标相对位移旋到相机前
        # 用绕 y 的朝向：看向目标的水平方向
        dx = float(goal[0] - agent[0])
        dz = float(goal[2] - agent[2])
        yaw = math.atan2(-dx, -dz)
        half = 0.5 * yaw
        sensor_rot = np.array([0.0, math.sin(half), 0.0, math.cos(half)])
        fr = {"xyz": goal.tolist(), "geodesic_m": float(geo), "fid": 0}
        drawn = frontier_candidates(
            [fr], intrinsic, sensor_pos, sensor_rot, (480, 640),
            max_n=3, drop_occluded=False, occ=occ, agent_xyz=agent)
        self.assertEqual(len(drawn), 1)
        self.assertIsNotNone(drawn[0].get("uv"))

    def test_geodesic_over_5m_filtered_by_threshold(self):
        """测地 ≥ 5 m 的点按近距阈值应被排除（与主循环同一判据）。"""
        near_m = 5.0
        sector = [
            {"xyz": [1.0, 0.88, 0.0], "geodesic_m": 3.0},
            {"xyz": [6.0, 0.88, 0.0], "geodesic_m": 5.5},
            {"xyz": [2.0, 0.88, 0.0], "geodesic_m": float("inf")},
        ]
        near = [
            fr for fr in sector
            if path_geodesic_ok(fr.get("geodesic_m"))
            and float(fr["geodesic_m"]) < near_m
        ]
        self.assertEqual(len(near), 1)
        self.assertEqual(near[0]["geodesic_m"], 3.0)


if __name__ == "__main__":
    unittest.main()
