"""不可达探索点在提取与选点阶段被丢掉。"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness.overlay import path_geodesic_ok, pick_mover_candidate
from nav.goto import pursue_occupancy
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


if __name__ == "__main__":
    unittest.main()
