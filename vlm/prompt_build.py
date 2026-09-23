"""按状态组装 Planner system prompt。"""

import json
import os

_PROMPTS = os.path.join(os.path.dirname(__file__), "prompts")
_BASE_PATH = os.path.join(_PROMPTS, "planner_base.txt")
_POLICY_PATH = os.path.join(_PROMPTS, "state_policy.json")
_TOOL_PATH = os.path.join(_PROMPTS, "tool_and_action.json")

_ORDER = ("Depth", "Look", "Recall", "Verify", "MakePlan", "TraceBack", "Locate")

_policy_cache = None
_tool_cache = None
_base_cache = None


def _load_base():
    """读 planner_base.txt。"""
    global _base_cache
    if _base_cache is None:
        with open(_BASE_PATH, "r", encoding="utf-8") as f:
            _base_cache = f.read()
    return _base_cache


def _load_policy():
    """读 state_policy.json。"""
    global _policy_cache
    if _policy_cache is None:
        with open(_POLICY_PATH, "r", encoding="utf-8") as f:
            _policy_cache = json.load(f)
    return _policy_cache


def _load_tools():
    """读 tool_and_action.json。"""
    global _tool_cache
    if _tool_cache is None:
        with open(_TOOL_PATH, "r", encoding="utf-8") as f:
            _tool_cache = json.load(f)
    return _tool_cache


def policy_key(state, blocked_from=None):
    """状态与 blocked_from 映射到 policy 键。

    Args:
        state: NavState 或字符串。
        blocked_from (str, optional): ``Unseen`` 或 ``Confirmed``。

    Returns:
        str: policy 键。
    """
    name = getattr(state, "value", None) or str(state)
    if name == "Blocked":
        if blocked_from == "Confirmed":
            return "Blocked2"
        return "Blocked1"
    return name


def build_system_prompt(goal, state, tools, actions, blocked_from=None):
    """拼装本拍 system：base + 当前策略 + 可用工具说明。

    Args:
        goal (str): 导航目标词。
        state: 当前 NavState 或字符串。
        tools (list): 允许的只读工具名。
        actions (list): 允许的终态名。
        blocked_from (str, optional): Blocked 来源。

    Returns:
        str: system 正文。
    """
    goal = str(goal)
    base = _load_base().replace("<goal>", goal)
    key = policy_key(state, blocked_from=blocked_from)
    policies = _load_policy()
    policy = (policies.get(key) or policies.get("Unseen") or "").replace("<goal>", goal)
    catalog = _load_tools()
    names = []
    for n in _ORDER:
        if n in (tools or []) or n in (actions or []):
            names.append(n)
    lines = []
    for n in names:
        text = catalog.get(n) or n
        lines.append("- " + text.replace("<goal>", goal))
    if not lines:
        lines.append("- (none)")
    return (
        base.rstrip()
        + "\n\n--------------------\nPolicy (this state only)\n"
        + policy.strip()
        + "\n\n--------------------\nAvailable tools/actions\n"
        + "\n".join(lines)
        + "\n"
    )
