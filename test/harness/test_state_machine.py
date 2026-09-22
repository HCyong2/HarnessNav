"""状态机：Miss 删除、Blocked 分型、Verify/Locate 校验。"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness.protocol import (BLOCKED_DIST_M, MAX_LOCATE_ATTEMPTS, MAX_LOCATE_LEGS,
                              MAX_MOVER_LEGS, validate_planner_action)
from harness.state import NavState, allowed_for


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

    def test_confirmed_has_locate_no_stop(self):
        """Confirmed 可 Locate，不可 Stop。"""
        tools, actions = allowed_for(NavState.CONFIRMED)
        self.assertIn("Locate", actions)
        self.assertNotIn("Stop", actions)
        self.assertIn("Depth", tools)

    def test_blocked_types(self):
        """type1 / type2 终态不同。"""
        t1, a1 = allowed_for(NavState.BLOCKED, blocked_from="Unseen")
        self.assertEqual(set(t1), {"Look", "Depth"})
        self.assertEqual(set(a1), {"MakePlan", "TraceBack"})
        self.assertNotIn("Locate", a1)
        t2, a2 = allowed_for(NavState.BLOCKED, blocked_from="Confirmed")
        self.assertIn("Locate", a2)
        self.assertEqual(set(t2), {"Look", "Depth"})


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


if __name__ == "__main__":
    unittest.main()
