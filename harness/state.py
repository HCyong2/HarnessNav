"""导航状态机。状态只由 Harness 改，模型不能自报。"""

from enum import Enum


class NavState(str, Enum):
    """对外可见的导航状态。"""

    UNSEEN = "Unseen"
    FIND = "Find"
    CONFIRMED = "Confirmed"
    ARRIVED = "Arrived"
    MISS = "Miss"
    BLOCKED = "Blocked"


READONLY_TOOLS = ("Depth", "Look", "Recall")
TERMINAL_OPS = ("MakePlan", "TraceBack", "Verify", "Stop")

ALLOWED = {
    NavState.UNSEEN: (READONLY_TOOLS, ("MakePlan", "TraceBack")),
    NavState.FIND: (READONLY_TOOLS, ("Verify", "MakePlan", "TraceBack")),
    NavState.CONFIRMED: (READONLY_TOOLS, ("MakePlan", "TraceBack", "Stop")),
    NavState.ARRIVED: ((), ("Stop",)),
    NavState.MISS: (READONLY_TOOLS, ("MakePlan", "TraceBack")),
    NavState.BLOCKED: (("Look", "Recall"), ("MakePlan", "TraceBack")),
}


def allowed_for(state):
    """返回 ``(allowed_tools, allowed_actions)``。

    Args:
        state (NavState): 当前状态。

    Returns:
        tuple: 两个字符串元组。
    """
    tools, actions = ALLOWED[NavState(state)]
    return list(tools), list(actions)
