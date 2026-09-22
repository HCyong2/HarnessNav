"""语义查询词黑名单与分割失败回退文案。"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness.protocol import (MAX_SEG_RETRIES, make_seg_retry_caption,
                              semantic_query_blocked, validate_planner_action)
from vlm.planner import VlmPlanner

_TOOLS = ["Depth", "Look", "Recall"]
_ACTIONS = ["MakePlan", "TraceBack"]


class TestPlanQuery(unittest.TestCase):
    """查询词与回退说明。"""

    def test_passage_words_blocked(self):
        """门、门口、走廊等应拒绝；室内物体与 indoor plant 应通过。"""
        for query in (
            "door", "Doors", "doorway", "bathroom doorway",
            "door frame", "doorframe", "corridor", "hallway",
            "floor", "wall", "ceiling",
        ):
            self.assertTrue(semantic_query_blocked(query), query)
        for query in ("sofa", "tv_monitor", "indoor plant", "bed", "chair"):
            self.assertFalse(semantic_query_blocked(query), query)

    def test_validate_semantic_query(self):
        """规划校验拒绝通道词，接受沙发。"""
        bad = validate_planner_action({
            "action": "MakePlan", "pano_id": 2, "mode": "semantic",
            "object_query": "door",
        }, _TOOLS, _ACTIONS)
        self.assertEqual(bad, "semantic object_query is a passage word")
        ok = validate_planner_action({
            "action": "MakePlan", "pano_id": 4, "mode": "semantic",
            "object_query": "sofa", "plan": "Approach the sofa.",
        }, _TOOLS, _ACTIONS)
        self.assertIsNone(ok)
        front = validate_planner_action({
            "action": "MakePlan", "pano_id": 0, "mode": "frontier",
        }, _TOOLS, _ACTIONS)
        self.assertIsNone(front)

    def test_retry_caption_and_limit(self):
        """回退说明含上次规划与中文提示；本圈最多回退两次。"""
        plan = {
            "action": "MakePlan", "pano_id": 2, "mode": "semantic",
            "object_query": "sofa", "plan": "Approach the sofa.",
        }
        text = make_seg_retry_caption(plan)
        self.assertIn("上一次的makeplan：", text)
        self.assertIn('"object_query": "sofa"', text)
        self.assertIn("无法有效识别到物体sofa，请尝试其他的方案。", text)
        self.assertEqual(MAX_SEG_RETRIES, 2)

    def test_feed_plan_retry_keeps_caption(self):
        """回退消息进入规划器对话，不含新的环视。"""
        planner = VlmPlanner.__new__(VlmPlanner)
        planner.messages = []
        planner.feed_plan_retry(make_seg_retry_caption({
            "action": "MakePlan", "pano_id": 2, "mode": "semantic",
            "object_query": "sofa",
        }))
        self.assertEqual(len(planner.messages), 1)
        self.assertEqual(planner.messages[0]["role"], "user")
        self.assertIn("无法有效识别到物体sofa", planner.messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
