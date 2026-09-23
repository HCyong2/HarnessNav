"""上下文压缩与跟随段数。"""

import os
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness.memory import NodeGraph
from harness.protocol import MAX_MOVER_LEGS
from nav.occupancy import VLM_BEV_MAX_WH, VLM_PANO_WH, fit_within, resize_exact
from perception.base import SEG_SCORE_TRIES, mean_depth_window
from vlm.summary import compress_bundle


class TestResize(unittest.TestCase):
    """拼图与俯视图尺寸。"""

    def test_pano_exact_960_480(self):
        """全景缩放到 960×480。"""
        src = np.zeros((980, 1940, 3), dtype=np.uint8)
        out = resize_exact(src, VLM_PANO_WH[0], VLM_PANO_WH[1])
        self.assertEqual(out.shape[1], 960)
        self.assertEqual(out.shape[0], 480)

    def test_bev_fits_box(self):
        """俯视图不超过 960×480 且不放大。"""
        big = np.zeros((1200, 2000, 3), dtype=np.uint8)
        out = fit_within(big, VLM_BEV_MAX_WH[0], VLM_BEV_MAX_WH[1])
        self.assertLessEqual(out.shape[1], 960)
        self.assertLessEqual(out.shape[0], 480)
        small = np.zeros((100, 80, 3), dtype=np.uint8)
        same = fit_within(small, 960, 480)
        self.assertEqual(same.shape[0], 100)
        self.assertEqual(same.shape[1], 80)


class TestHistory(unittest.TestCase):
    """PlannerIn.history 不含 views，且默认不含当前节点。"""

    def test_history_is_summary_only(self):
        """远程节点也不附六向 views；当前节点无 summary 时不进列表。"""
        root = tempfile.mkdtemp(prefix="hist_")
        g = NodeGraph(root)
        rgb = {0: np.zeros((8, 8, 3), dtype=np.uint8)}
        g.upsert_scan([0, 0, 0], 0.0, rgb, {}, [], None, None)
        g.nodes[0]["views"] = [{"pano_id": 0, "goal_find": True, "landmark": "bed"}]
        self.assertEqual(g.history(0), [])

        g.nodes[0]["summary"] = "first loop done"
        g.upsert_scan([2, 0, 2], 0.0, rgb, {}, [], None, None)
        g.nodes[1]["views"] = [{"pano_id": 2, "landmark": "sofa"}]
        hist = g.history(1)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["node_id"], 0)
        self.assertEqual(hist[0]["summary"], "first loop done")
        for row in hist:
            self.assertNotIn("views", row)
            self.assertEqual(
                set(row.keys()), {"node_id", "visit_count", "summary", "leftover"})
            self.assertIsInstance(row["leftover"], list)

        # 回溯再访：当前节点已有 summary，列入 history
        g.nodes[1]["summary"] = "second loop"
        hist_tb = g.history(0)
        self.assertEqual([r["node_id"] for r in hist_tb], [0, 1])
        self.assertEqual(hist_tb[0]["summary"], "first loop done")

    def test_history_keeps_last_five(self):
        """最多保留最近 5 个有 summary 的节点。"""
        root = tempfile.mkdtemp(prefix="hist5_")
        g = NodeGraph(root)
        rgb = {0: np.zeros((8, 8, 3), dtype=np.uint8)}
        for i in range(7):
            g.upsert_scan([float(i), 0, 0], 0.0, rgb, {}, [], None, None)
            g.nodes[i]["summary"] = f"node {i}"
        hist = g.history(6)
        self.assertEqual(len(hist), 5)
        self.assertEqual([r["node_id"] for r in hist], [2, 3, 4, 5, 6])


class TestSummaryCompress(unittest.TestCase):
    """Summary 输入不含完整对话。"""

    def test_bundle_keys(self):
        """只有 views、tools、终态与执行。"""
        bundle = compress_bundle(
            [{"pano_id": 0, "goal_find": True, "landmark": "bed", "room_type": "bedroom"}],
            [{"action": "Depth", "args": {"pano_id": 0},
              "result": {"ok": True, "instances": [
                  {"id": "bed_1", "depth_m": 0.4, "score": 0.9, "uv": [1, 2]}]},
              "reasoning": "measure"}],
            {"action": "Stop"},
            "close enough",
            {"state": "Confirmed", "mover": {"status": "ok", "dist_moved_m": 0.0}},
        )
        self.assertEqual(set(bundle.keys()), {
            "views", "tools", "final_action", "final_reasoning", "execution"})
        self.assertNotIn("observe", bundle)
        self.assertNotIn("planner_messages", bundle)
        self.assertNotIn("planner_rounds", bundle)
        self.assertEqual(bundle["tools"][0]["result"]["instances"][0]["id"], "bed_1")
        self.assertNotIn("uv", bundle["tools"][0]["result"]["instances"][0])

    def test_depth_three_sig_figs(self):
        """测地距离压成 3 位有效数字；回写不含 depth_m。"""
        from harness.protocol import round_sig

        self.assertEqual(round_sig(0.5384885668754578), 0.538)
        self.assertEqual(round_sig(0.1418366301675574), 0.142)
        self.assertEqual(round_sig(12.345), 12.3)
        self.assertIsNone(round_sig(float("inf")))
        bundle = compress_bundle(
            [],
            [{"action": "Depth", "args": {},
              "result": {"ok": True, "instances": [
                  {"id": "bed_1", "depth_m": 0.5384885668754578,
                   "geodesic_m": 0.1418366301675574, "score": 0.4557759463787079}]},
              "reasoning": "x"}],
            {"action": "Stop"}, "", {})
        inst = bundle["tools"][0]["result"]["instances"][0]
        self.assertNotIn("depth_m", inst)
        self.assertNotIn("score", inst)
        self.assertEqual(inst["geodesic_m"], 0.142)


class TestSegAndDepthWindow(unittest.TestCase):
    """分割档位、深度窗口与跟随段数。"""

    def test_three_score_tries(self):
        """分割只降三档：0.5、0.4、0.3。"""
        self.assertEqual(SEG_SCORE_TRIES, (0.5, 0.4, 0.3))

    def test_mean_depth_9x9(self):
        """深度为中心周围 9×9 有效值均值，忽略 0。"""
        depth = np.zeros((20, 20), dtype=np.float32)
        depth[6:15, 6:15] = 2.0
        depth[10, 10] = 10.0
        got = mean_depth_window(depth, (10, 10), size=9)
        self.assertAlmostEqual(got, (2.0 * 80 + 10.0) / 81.0, places=5)

    def test_legs_constant(self):
        """跟随最多 3 段。"""
        self.assertEqual(MAX_MOVER_LEGS, 3)


if __name__ == "__main__":
    unittest.main()
