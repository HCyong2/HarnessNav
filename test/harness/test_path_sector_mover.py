"""路径扇区 unexplored、Mover 执行失败与 Blocked 门控。"""

import math
import os
import sys
import unittest
from unittest import mock

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness.loop import Harness, _mover_exec_fail
from harness.memory import NodeGraph, refresh_unexplored
from harness.state import NavState
from nav.occupancy import (CELL_FREE, CELL_OCC, CELL_UNKNOWN, NEAR_FRONTIER_GEO_M,
                           OccupancyMap, even_pano_dir, path_even_pano_dir,
                           planner_sector_dirs)


def _stub_occ(grid, ox=0.0, oz=0.0, res=0.1):
    """手工占用图。"""
    occ = OccupancyMap.__new__(OccupancyMap)
    occ.grid_res = res
    occ._ox, occ._oz = ox, oz
    occ.grid = np.asarray(grid, dtype=np.uint8)
    occ.inflate = np.zeros_like(occ.grid)
    return occ


class TestPathSector(unittest.TestCase):
    """A* 路径起步朝向 vs 欧氏方位。"""

    def test_l_shape_path_sector_differs_from_euclidean(self):
        """L 形通道：欧氏朝 A，路径起步朝 B。"""
        # 20x20 FREE，中间竖墙只留底部通道
        g = np.full((20, 20), CELL_FREE, dtype=np.uint8)
        g[:, 10] = CELL_OCC
        g[18, 10] = CELL_FREE  # 底部开口
        occ = _stub_occ(g)
        agent = np.array([0.5, 0.88, 0.5])   # 左下附近 col~5 row~5
        # 调整到 FREE 格中心
        agent = np.array([5 * 0.1 + 0.05, 0.88, 5 * 0.1 + 0.05])
        goal = np.array([15 * 0.1 + 0.05, 0.88, 5 * 0.1 + 0.05])  # 墙右侧同高
        yaw = 0.0  # 朝 +z
        euclid = even_pano_dir(goal, agent, yaw)
        path_dir = path_even_pano_dir(occ, agent, goal, yaw)
        # 欧氏大致朝 +x → 相对 yaw=0 约为 90° → pano 约 2 或 4
        # 路径须先向 +z 绕洞再拐 → 起步更接近 pano 0
        self.assertNotEqual(euclid, path_dir)

    def test_far_geodesic_excluded_from_sector_dirs(self):
        """测地 ≥ 5 m 的点不进 planner_sector_dirs。"""
        frs = [
            {"xyz": [1.0, 0.88, 2.0], "geodesic_m": NEAR_FRONTIER_GEO_M + 0.5},
            {"xyz": [0.5, 0.88, 1.0], "geodesic_m": 1.0},
        ]
        occ = _stub_occ(np.full((30, 30), CELL_FREE, dtype=np.uint8))
        agent = np.array([0.5, 0.88, 0.5])
        yaw = 0.0
        with mock.patch("nav.occupancy.path_even_pano_dir", return_value=0):
            dirs = planner_sector_dirs(frs, agent, yaw, occ)
        self.assertEqual(dirs, {0})


class TestMoverExecFail(unittest.TestCase):
    """一步未走 vs 真走不动。"""

    def test_mover_exec_fail_predicate(self):
        """miss/lost/seg_empty 且 legs=0 为执行失败。"""
        self.assertTrue(_mover_exec_fail({"status": "miss", "legs": 0}))
        self.assertTrue(_mover_exec_fail({"status": "seg_empty", "legs": 0}))
        self.assertTrue(_mover_exec_fail({"status": "lost", "legs": 0}))
        self.assertFalse(_mover_exec_fail({"status": "miss", "legs": 1}))
        self.assertFalse(_mover_exec_fail({"status": "blocked", "legs": 1}))
        self.assertFalse(_mover_exec_fail({"status": "arrived_subgoal", "legs": 1}))

    def test_miss_legs0_does_not_enter_blocked(self):
        """MakePlan miss legs=0 → mover_retry，状态仍 Unseen。"""
        h = Harness.__new__(Harness)
        h.state = NavState.UNSEEN
        h.blocked_from = None
        h.current_id = 0
        h.tool_counts = {}
        h.graph = NodeGraph()
        h.graph.nodes[0] = {
            "node_id": 0, "explored_dirs": [], "views": [], "leftover": [],
            "last_plan": None,
        }
        h._slog = lambda *a, **k: None
        report = {"status": "miss", "legs": 0, "dist_moved_m": 0.0, "chosen_ids": []}
        with mock.patch.object(h, "run_mover", return_value=report):
            out = h.apply_action(
                {"action": "MakePlan", "pano_id": 6, "mode": "frontier", "plan": "x"},
                {"node_id": 0, "frontiers": [], "yaw": 0.0})
        self.assertTrue(out.get("mover_retry"))
        self.assertEqual(h.state, NavState.UNSEEN)
        self.assertEqual(h.graph.nodes[0].get("explored_dirs"), [])

    def test_blocked_after_pursue_enters_blocked(self):
        """已跟随后 dist=0 → Blocked type1。"""
        h = Harness.__new__(Harness)
        h.state = NavState.UNSEEN
        h.blocked_from = None
        h.current_id = 0
        h.tool_counts = {}
        h.graph = NodeGraph()
        h.graph.nodes[0] = {
            "node_id": 0, "explored_dirs": [], "views": [
                {"pano_id": 0, "unexplored": True}],
            "leftover": [], "last_plan": None,
        }
        h._slog = lambda *a, **k: None
        report = {
            "status": "blocked", "legs": 1, "dist_moved_m": 0.0,
            "chosen_ids": ["F0"],
        }
        with mock.patch.object(h, "run_mover", return_value=report):
            out = h.apply_action(
                {"action": "MakePlan", "pano_id": 0, "mode": "frontier", "plan": "x"},
                {"node_id": 0, "frontiers": [], "yaw": 0.0})
        self.assertFalse(out.get("mover_retry"))
        self.assertEqual(h.state, NavState.BLOCKED)
        self.assertEqual(h.blocked_from, "Unseen")
        self.assertIn(0, h.graph.nodes[0]["explored_dirs"])


class TestSemanticLeftover(unittest.TestCase):
    """节点语义 leftover 计数。"""

    def test_leftover_count_prefers_phrases(self):
        """有语义短语时按条数；否则数 unexplored。"""
        g = NodeGraph()
        g.nodes[0] = {
            "leftover": ["A door to bedroom", "corridor"],
            "views": [{"pano_id": 0, "unexplored": True}],
        }
        self.assertEqual(g.leftover_count(0), 2)
        g.nodes[1] = {
            "leftover": [],
            "views": [
                {"pano_id": 0, "unexplored": True},
                {"pano_id": 2, "unexplored": False},
            ],
        }
        self.assertEqual(g.leftover_count(1), 1)

    def test_refresh_unexplored_uses_sector_dirs(self):
        """sector_dirs ∩ 未探索 → unexplored。"""
        node = {
            "explored_dirs": [2],
            "views": [
                {"pano_id": 0, "unexplored": False},
                {"pano_id": 2, "unexplored": True},
                {"pano_id": 4, "unexplored": False},
            ],
        }
        refresh_unexplored(node, {0, 2, 4})
        by = {int(v["pano_id"]): v["unexplored"] for v in node["views"]}
        self.assertTrue(by[0])
        self.assertFalse(by[2])  # 已选
        self.assertTrue(by[4])


if __name__ == "__main__":
    unittest.main()
