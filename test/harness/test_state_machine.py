"""状态机：Blocked 分型、Verify/Locate、Confirmed 仅 Locate。"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness.protocol import (BLOCKED_DIST_M, MAX_LOCATE_ATTEMPTS, MAX_LOCATE_LEGS,
                              MAX_MOVER_LEGS, validate_planner_action)
from harness.state import NavState, allowed_for, needs_bev


class TestAllowed(unittest.TestCase):
    """白名单。"""

    def test_no_miss(self):
        """Miss 枚举已删除。"""
        self.assertFalse(hasattr(NavState, "MISS"))

    def test_find_only_verify(self):
        """Find 无工具、仅 Verify。"""
        tools, actions = allowed_for(NavState.FIND)
        self.assertEqual(tools, [])
        self.assertEqual(actions, ["Verify"])

    def test_confirmed_locate_only(self):
        """Confirmed 无工具、仅 Locate。"""
        tools, actions = allowed_for(NavState.CONFIRMED)
        self.assertEqual(tools, [])
        self.assertEqual(actions, ["Locate"])
        self.assertNotIn("Stop", actions)
        self.assertNotIn("TraceBack", actions)
        self.assertNotIn("MakePlan", actions)

    def test_blocked_types(self):
        """type1 可 TraceBack；type2 可 Locate，不可 TraceBack。"""
        t1, a1 = allowed_for(NavState.BLOCKED, blocked_from="Unseen")
        self.assertEqual(set(t1), {"Look", "Depth"})
        self.assertEqual(set(a1), {"MakePlan", "TraceBack"})
        self.assertNotIn("Locate", a1)
        t2, a2 = allowed_for(NavState.BLOCKED, blocked_from="Confirmed")
        self.assertEqual(set(t2), {"Look", "Depth"})
        self.assertEqual(set(a2), {"MakePlan", "Locate"})
        self.assertNotIn("TraceBack", a2)

    def test_needs_bev(self):
        """仅 Unseen 与 Blocked type1 需要俯视图。"""
        self.assertTrue(needs_bev(NavState.UNSEEN))
        self.assertTrue(needs_bev(NavState.BLOCKED, blocked_from="Unseen"))
        self.assertFalse(needs_bev(NavState.BLOCKED, blocked_from="Confirmed"))
        self.assertFalse(needs_bev(NavState.FIND))
        self.assertFalse(needs_bev(NavState.CONFIRMED))


class TestValidate(unittest.TestCase):
    """动作参数。"""

    def test_verify_pano_only(self):
        """Verify 只需扇区编号。"""
        err = validate_planner_action(
            {"action": "Verify", "pano_id": 4}, [], ["Verify"])
        self.assertIsNone(err)
        err = validate_planner_action(
            {"action": "Verify"}, [], ["Verify"])
        self.assertEqual(err, "Verify.pano_id invalid")

    def test_locate_pano(self):
        """Locate 只需扇区编号。"""
        err = validate_planner_action(
            {"action": "Locate", "pano_id": 2}, [], ["Locate"])
        self.assertIsNone(err)

    def test_constants(self):
        """腿数与卡住阈值。"""
        self.assertEqual(MAX_MOVER_LEGS, 3)
        self.assertEqual(MAX_LOCATE_LEGS, 6)
        self.assertEqual(MAX_LOCATE_ATTEMPTS, 3)
        self.assertAlmostEqual(BLOCKED_DIST_M, 0.1)


class TestMotionOutcome(unittest.TestCase):
    """Blocked 进出（无仿真）。"""

    def test_enter_and_escape(self):
        """Unseen 小位移进 type1；大位移脱困回 Unseen。"""
        from harness.loop import Harness

        h = Harness.__new__(Harness)
        h.state = NavState.UNSEEN
        h.blocked_from = None
        h.confirmed_node_floor = None
        h.current_id = 0
        h.log = type("L", (), {"emit": staticmethod(lambda *a, **k: None)})()
        h.scan_count = 0
        h._slog = lambda *a, **k: None
        h._apply_motion_outcome(0.05, "MakePlan", allow_enter_blocked=True)
        self.assertEqual(h.state, NavState.BLOCKED)
        self.assertEqual(h.blocked_from, "Unseen")
        h._apply_motion_outcome(0.2, "MakePlan", allow_enter_blocked=True)
        self.assertEqual(h.state, NavState.UNSEEN)
        self.assertIsNone(h.blocked_from)

    def test_confirmed_blocked_restores(self):
        """Confirmed 卡住进 type2，脱困回 Confirmed。"""
        from harness.loop import Harness

        h = Harness.__new__(Harness)
        h.state = NavState.CONFIRMED
        h.blocked_from = None
        h.confirmed_node_floor = 2
        h.current_id = 3
        h._slog = lambda *a, **k: None
        h._apply_motion_outcome(0.01, "Locate", allow_enter_blocked=True)
        self.assertEqual(h.state, NavState.BLOCKED)
        self.assertEqual(h.blocked_from, "Confirmed")
        h._apply_motion_outcome(0.15, "Locate", allow_enter_blocked=True)
        self.assertEqual(h.state, NavState.CONFIRMED)

    def test_verify_cannot_enter_blocked(self):
        """allow_enter_blocked=False 时不进 Blocked。"""
        from harness.loop import Harness

        h = Harness.__new__(Harness)
        h.state = NavState.UNSEEN
        h.blocked_from = None
        h._slog = lambda *a, **k: None
        h._apply_motion_outcome(0.0, "MakePlan", allow_enter_blocked=False)
        self.assertEqual(h.state, NavState.UNSEEN)


class TestPromptBuild(unittest.TestCase):
    """按状态组装 system。"""

    def test_policy_keys(self):
        """Blocked 分型键名。"""
        from vlm.prompt_build import build_system_prompt, policy_key

        self.assertEqual(policy_key("Unseen"), "Unseen")
        self.assertEqual(policy_key("Blocked", "Unseen"), "Blocked1")
        self.assertEqual(policy_key("Blocked", "Confirmed"), "Blocked2")
        text = build_system_prompt(
            "chair", "Find", [], ["Verify"], blocked_from=None)
        self.assertIn("Verify", text)
        self.assertNotIn("MakePlan:", text.split("Available tools/actions")[-1]
                         if "Available tools/actions" in text else text)


if __name__ == "__main__":
    unittest.main()
