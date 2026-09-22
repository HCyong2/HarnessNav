"""Planner / Mover 的 OpenAI function schema。"""

from harness.protocol import LOOK_ACTIONS, PANO_IDS, PLAN_MODES


def planner_tools(allowed_tools, allowed_actions):
    """按 FSM 门控过滤后的工具列表。

    Args:
        allowed_tools (list): 只读 Skill。
        allowed_actions (list): 终态动作。

    Returns:
        list: OpenAI tools。
    """
    catalog = {
        "Depth": {
            "type": "function",
            "function": {
                "name": "Depth",
                "description": "Measure occupancy-map geodesic distance to a GLEE instance in a panorama. Returns geodesic_m (path length); ray depth_m is diagnostic only. Read-only.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pano_id": {"type": "integer", "enum": list(PANO_IDS)},
                        "object": {"type": "string"},
                        "instance_id": {"type": "string"},
                    },
                    "required": ["pano_id", "object"],
                },
            },
        },
        "Look": {
            "type": "function",
            "function": {
                "name": "Look",
                "description": "Pitch or yaw one discrete step. Use down to see the floor.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "look": {"type": "string", "enum": list(LOOK_ACTIONS)},
                    },
                    "required": ["look"],
                },
            },
        },
        "Recall": {
            "type": "function",
            "function": {
                "name": "Recall",
                "description": "Fetch stored images of an old scan node. Does not move the body.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "integer"},
                        "pano_id": {"type": "integer", "enum": list(PANO_IDS)},
                        "query": {"type": "string"},
                    },
                    "required": ["node_id", "pano_id"],
                },
            },
        },
        "Verify": {
            "type": "function",
            "function": {
                "name": "Verify",
                "description": "Two-view VLM check of the navigation object. instance_id is like toilet_1 using the category name. Required terminal when state is Find.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instance_id": {"type": "string"},
                        "pano_id": {"type": "integer", "enum": list(PANO_IDS)},
                    },
                    "required": ["instance_id", "pano_id"],
                },
            },
        },
        "MakePlan": {
            "type": "function",
            "function": {
                "name": "MakePlan",
                "description": "Lock a subgoal. semantic needs object_query; frontier must omit it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pano_id": {"type": "integer", "enum": list(PANO_IDS)},
                        "mode": {"type": "string", "enum": list(PLAN_MODES)},
                        "object_query": {"type": "string"},
                        "plan": {"type": "string"},
                    },
                    "required": ["pano_id", "mode"],
                },
            },
        },
        "TraceBack": {
            "type": "function",
            "function": {
                "name": "TraceBack",
                "description": "Walk the body back to an old node. Not the same as Recall.",
                "parameters": {
                    "type": "object",
                    "properties": {"node_id": {"type": "integer"}},
                    "required": ["node_id"],
                },
            },
        },
        "Stop": {
            "type": "function",
            "function": {
                "name": "Stop",
                "description": "End the episode when occupancy geodesic to the goal is ≤ 1.0 m. Call Depth first and check geodesic_m. Only when state is Confirmed.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    }
    names = set(allowed_tools) | set(allowed_actions)
    return [catalog[n] for n in ("Depth", "Look", "Recall", "Verify",
                                 "MakePlan", "TraceBack", "Stop") if n in names and n in catalog]
