"""P0 脚本 Planner：目标分割 → 核实/语义靠近；否则未探索开口走探索点；否则回溯。"""

from harness.protocol import PANO_IDS, validate_planner_action
from harness.state import allowed_for


class ScriptedPlanner:
    """不调用 VLM 的确定性规划器。"""

    def __init__(self):
        """初始化。"""
        pass

    def act(self, payload, detections, leftover, graph, current_id, env):
        """根据当前节点检测与 leftover 给出一个终态 action。

        Args:
            payload (dict): PlannerIn。
            detections (dict): ``pano_id -> {goal: SegResult, door: SegResult}``。
            leftover (list): ``{"dir","n"}``。
            graph: ``NodeGraph``。
            current_id (int): 当前节点。
            env: ``habitat.Env``。

        Returns:
            dict: Planner 输出。
        """
        tools, actions = allowed_for(
            payload["state"], blocked_from=(
                "Confirmed" if payload.get("blocked_type") == 2 else
                ("Unseen" if payload.get("blocked_type") == 1 else None)))
        goal = payload["goal"]
        state = payload["state"]

        def emit(action):
            err = validate_planner_action(action, tools, actions)
            if err:
                return {"action": "MakePlan", "pano_id": 0, "mode": "frontier",
                        "object_query": None, "plan": f"planner_violation:{err}"}
            return action

        if state == "Arrived":
            return {"action": "MakePlan", "pano_id": 0, "mode": "frontier",
                    "object_query": None, "plan": "already arrived"}

        best_goal = _best_pano(detections, "goal")

        if state == "Find" and "Verify" in actions:
            pid = int(best_goal[0]) if best_goal is not None else 0
            return emit({"action": "Verify", "pano_id": pid})

        if state == "Confirmed" and "Locate" in actions and best_goal is not None:
            return emit({"action": "Locate", "pano_id": int(best_goal[0])})

        if best_goal is not None and "MakePlan" in actions:
            pid, result = best_goal
            return emit({
                "action": "MakePlan",
                "pano_id": pid,
                "mode": "semantic",
                "object_query": goal,
                "plan": f"Approach {goal}.",
            })

        views = payload.get("views") or []
        unexp = [v for v in views if v.get("unexplored")]
        if unexp and "MakePlan" in actions:
            pid = int(unexp[0]["pano_id"])
            return emit({
                "action": "MakePlan",
                "pano_id": pid,
                "mode": "frontier",
                "object_query": None,
                "plan": f"Explore unexplored dir {pid}.",
            })

        if leftover and "MakePlan" in actions:
            top = max(leftover, key=lambda x: int(x.get("n", 0)))
            return emit({
                "action": "MakePlan",
                "pano_id": int(top["dir"]),
                "mode": "frontier",
                "object_query": None,
                "plan": f"Explore leftover dir {top['dir']}.",
            })

        if "TraceBack" in actions and current_id is not None:
            allowed_ids = payload.get("traceback_node_ids")
            tid = graph.best_traceback_target(current_id, env, 15.0)
            if tid is not None and (allowed_ids is None or int(tid) in allowed_ids):
                return emit({"action": "TraceBack", "node_id": int(tid)})

        return emit({
            "action": "MakePlan",
            "pano_id": 0,
            "mode": "frontier",
            "object_query": None,
            "plan": "Explore forward.",
        })


def _best_pano(detections, key):
    """分数最高的朝向。

    Args:
        detections (dict): pano -> 检测。
        key (str): ``goal`` 或 ``door``。

    Returns:
        tuple: ``(pano_id, SegResult)`` 或 ``None``。
    """
    best = None
    best_s = -1.0
    for pid in PANO_IDS:
        pack = detections.get(int(pid)) or detections.get(str(pid)) or {}
        result = pack.get(key)
        if result is None or len(result) == 0:
            continue
        s = float(result.scores[0])
        if s > best_s:
            best_s = s
            best = (int(pid), result)
    return best
