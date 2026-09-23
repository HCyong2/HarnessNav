"""导航状态机。状态只由 Harness 改，模型不能自报。"""

from enum import Enum


class NavState(str, Enum):
    """对外可见的导航状态。"""

    UNSEEN = "Unseen"
    FIND = "Find"
    CONFIRMED = "Confirmed"
    ARRIVED = "Arrived"
    BLOCKED = "Blocked"


READONLY_TOOLS = ("Depth", "Look", "Recall")
BLOCKED_TOOLS = ("Look", "Depth")
TERMINAL_OPS = ("MakePlan", "TraceBack", "Verify", "Locate")

ALLOWED = {
    NavState.UNSEEN: (READONLY_TOOLS, ("MakePlan", "TraceBack")),
    NavState.FIND: ((), ("Verify",)),
    NavState.CONFIRMED: ((), ("Locate",)),
    NavState.ARRIVED: ((), ()),
    NavState.BLOCKED: (BLOCKED_TOOLS, ("MakePlan", "TraceBack")),
}

# type2 Blocked：已确认目标后卡住；无 TraceBack。
BLOCKED_TYPE2_ACTIONS = ("MakePlan", "Locate")


def allowed_for(state, blocked_from=None):
    """返回 ``(allowed_tools, allowed_actions)``。

    Args:
        state (NavState): 当前状态。
        blocked_from (str, optional): ``Unseen`` 或 ``Confirmed``；仅 Blocked 时有效。

    Returns:
        tuple: 两个字符串列表。
    """
    state = NavState(state)
    if state == NavState.BLOCKED and blocked_from == "Confirmed":
        return list(BLOCKED_TOOLS), list(BLOCKED_TYPE2_ACTIONS)
    tools, actions = ALLOWED[state]
    return list(tools), list(actions)


def needs_bev(state, blocked_from=None):
    """是否在 Observe 后向 Planner 追加俯视图。

    Args:
        state (NavState): 当前状态。
        blocked_from (str, optional): Blocked 来源。

    Returns:
        bool: Unseen 与 Blocked type1 为真。
    """
    state = NavState(state)
    if state == NavState.UNSEEN:
        return True
    if state == NavState.BLOCKED and blocked_from != "Confirmed":
        return True
    return False
