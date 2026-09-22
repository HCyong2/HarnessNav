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
    """PlannerIn.history 不含 views。"""

    def test_history_is_summary_only(self):
        """远程节点也不附六向 views。"""
        root = tempfile.mkdtemp(prefix="hist_")
        g = NodeGraph(root)
        rgb = {0: np.zeros((8, 8, 3), dtype=np.uint8)}
        g.upsert_scan([0, 0, 0], 0.0, rgb, {}, [], None, "first")
        g.nodes[0]["views"] = [{"pano_id": 0, "goal_find": True, "landmark": "bed"}]
        g.upsert_scan([2, 0, 2], 0.0, rgb, {}, [], None, "second")
        g.nodes[1]["views"] = [{"pano_id": 2, "landmark": "sofa"}]
        hist = g.history(1)
        self.assertEqual(len(hist), 2)
        for row in hist:
            self.assertIn("summary", row)
            self.assertNotIn("views", row)
            self.assertEqual(set(row.keys()), {"node_id", "visit_count", "summary"})


class TestSummaryCompress(unittest.TestCase):
    """Summary 输入不含完整对话。"""

    def test_bundle_keys(self):
        """只有 observe、tools、终态与执行。"""
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
            "observe", "tools", "final_action", "final_reasoning", "execution"})
        self.assertNotIn("planner_messages", bundle)
        self.assertNotIn("planner_rounds", bundle)
        self.assertEqual(bundle["tools"][0]["result"]["instances"][0]["id"], "bed_1")
        self.assertNotIn("uv", bundle["tools"][0]["result"]["instances"][0])


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
