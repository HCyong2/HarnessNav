"""Planner / Mover / Skill 的 JSON 字段约定（P0 脚本与 P1 VLM 共用）。"""

import json
import math
import os
import re

import numpy as np

PANO_IDS = (0, 2, 4, 6, 8, 10)
LOOK_ACTIONS = ("up", "down", "left", "right")
MOVER_STATUS = ("ok", "miss", "blocked", "arrived_subgoal", "stopped", "lost", "seg_empty")
PLAN_MODES = ("semantic", "frontier")
MAX_SEG_RETRIES = 2
MAX_MOVER_LEGS = 3
MAX_LOCATE_LEGS = 6
MAX_LOCATE_ATTEMPTS = 3
# MakePlan / Locate 累计位移低于该值则进入 Blocked。
BLOCKED_DIST_M = 0.1
# 难以稳定分割的通道词，禁止作为语义查询。
_SEMANTIC_QUERY_PHRASES = (
    "door frame", "doorframe", "doorway", "doorways",
    "corridor", "hallway",
)
_SEMANTIC_QUERY_TOKENS = frozenset((
    "door", "doors", "floor", "floors", "wall", "walls", "ceiling", "ceilings",
    "corridor", "hallway", "doorway", "doorways", "doorframe",
))


def round_sig(value, n=3):
    """保留 ``n`` 位有效数字；非有限或不可转为 ``None``。

    Args:
        value: 数值。
        n (int): 有效数字位数。

    Returns:
        float: 四舍五入后的数；无法表示时为 ``None``。
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if v == 0.0:
        return 0.0
    return float(f"{v:.{int(n)}g}")


def jsonable(obj):
    """把 numpy / 路径转成可 ``json.dump`` 的结构。

    Args:
        obj: 任意对象。

    Returns:
        object: 只含 list / dict / 标量。
    """
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def dump_json(path, obj):
    """写 UTF-8 JSON。

    Args:
        path (str): 路径。
        obj: 数据。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, ensure_ascii=False, indent=2)


def _fmt_metric(value):
    """指标写成四位小数；缺失为 ``n/a``。"""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


# 与 harness.loop.TOOL_NAMES / harness.state.NavState 对齐，固定摘要顺序。
_TOOL_ORDER = ("Depth", "Look", "Recall", "Verify", "MakePlan", "TraceBack", "Locate", "Stop")
_STATE_ORDER = ("Unseen", "Find", "Confirmed", "Arrived", "Blocked")
_EXAMPLE_CAP = 3


def episode_key(row):
    """从一集记录取出 ``scene_episode_id`` 样式的键。"""
    if not isinstance(row, dict):
        return None
    key = row.get("key")
    if key:
        return str(key)
    scene = row.get("scene")
    eid = row.get("episode_id")
    if scene is not None and eid is not None:
        return f"{scene}_{eid}"
    return None


def _ordered_names(seen, preferred):
    """按 ``preferred`` 顺序排列，未见过的名排在后面。"""
    names = [n for n in preferred if n in seen]
    names.extend(sorted(n for n in seen if n not in preferred))
    return names


def tool_call_stats(records):
    """汇总工具总次数、每集均值与样例键（每工具最多三个）。

    Args:
        records (list): 每集记录，可含 ``tool_counts``。

    Returns:
        tuple: ``(names, totals, means, examples, n)``。
    """
    n = len(records)
    totals = {}
    examples = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        counts = row.get("tool_counts") or {}
        if not isinstance(counts, dict):
            continue
        key = episode_key(row)
        for name, raw in counts.items():
            try:
                count = int(raw)
            except (TypeError, ValueError):
                continue
            totals[name] = totals.get(name, 0) + count
            if count > 0 and key:
                bucket = examples.setdefault(name, [])
                if key not in bucket and len(bucket) < _EXAMPLE_CAP:
                    bucket.append(key)
    names = _ordered_names(totals, _TOOL_ORDER)
    means = {
        name: (totals[name] / n if n else 0.0) for name in names
    }
    return names, totals, means, examples, n


def final_state_stats(records):
    """汇总终态计数与样例键（每终态最多三个）。

    Args:
        records (list): 每集记录，可含 ``state``。

    Returns:
        tuple: ``(names, counts, examples)``。
    """
    counts = {}
    examples = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        state = row.get("state")
        if not state:
            continue
        state = str(state)
        counts[state] = counts.get(state, 0) + 1
        key = episode_key(row)
        if key:
            bucket = examples.setdefault(state, [])
            if key not in bucket and len(bucket) < _EXAMPLE_CAP:
                bucket.append(key)
    names = _ordered_names(counts, _STATE_ORDER)
    return names, counts, examples


def _append_tool_and_state_lines(lines, records):
    """把工具调用与终态两段追加到摘要行列表。"""
    tool_names, totals, means, tool_examples, _n = tool_call_stats(records)
    if tool_names:
        lines.append("")
        lines.append("Tool Call:")
        for name in tool_names:
            lines.append(
                f"{name.lower()}: {totals[name]}  mean {means[name]:.2f}")
        used = [name for name in tool_names if tool_examples.get(name)]
        if used:
            lines.append("")
            lines.append("Tool Call Example:")
            for name in used:
                lines.append(
                    f"{name.lower()}: {', '.join(tool_examples[name])}")

    state_names, state_counts, state_examples = final_state_stats(records)
    if state_names:
        lines.append("")
        lines.append("Final State")
        for name in state_names:
            lines.append(f"{name} {state_counts[name]}")
        lines.append("")
        lines.append("Final State Example:")
        for name in state_names:
            keys = state_examples.get(name) or []
            if keys:
                lines.append(f"{name}: {', '.join(keys)}")


def write_brief_summary(run_dir, records, means, total_s, n_assigned=None):
    """写出 ``brief_summary.txt``：指标均值、耗时、工具调用与终态。

    Args:
        run_dir (str): 本次实验目录。
        records (list): 每集记录，可含 ``time_cost`` / ``tool_counts`` / ``state``。
        means (dict): 指标均值，键为 ``success`` / ``distance_to_goal`` / ``spl`` 等。
        total_s (float): 整次运行的墙钟秒数。
        n_assigned (int, optional): 计划集数；与完成数不同时另写一行。

    Returns:
        str: 摘要文件路径。
    """
    n = len(records)
    times = []
    for row in records:
        raw = row.get("time_cost") if isinstance(row, dict) else None
        if raw is None:
            continue
        try:
            times.append(float(raw))
        except (TypeError, ValueError):
            pass
    mean_ep = (sum(times) / len(times)) if times else None
    lines = [
        f"n={n}",
    ]
    if n_assigned is not None and int(n_assigned) != n:
        lines.append(f"n_assigned={int(n_assigned)}")
    lines.extend([
        f"sr={_fmt_metric(means.get('success'))}",
        f"dtg={_fmt_metric(means.get('distance_to_goal'))}",
        f"spl={_fmt_metric(means.get('spl'))}",
        f"soft_spl={_fmt_metric(means.get('soft_spl'))}",
        f"total_time_s={float(total_s):.1f}",
        ("mean_episode_time_s={:.1f}".format(mean_ep)
         if mean_ep is not None else "mean_episode_time_s=n/a"),
    ])
    _append_tool_and_state_lines(lines, records)
    path = os.path.join(run_dir, "brief_summary.txt")
    os.makedirs(run_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def validate_planner_action(action, allowed_tools, allowed_actions):
    """检查 Planner 输出。合法返回 ``None``，否则返回错误字符串。

    Args:
        action (dict): 含 ``action`` 字段。
        allowed_tools (list): 只读工具。
        allowed_actions (list): 终态动作。

    Returns:
        str: 错误；通过为 ``None``。
    """
    if not isinstance(action, dict) or "action" not in action:
        return "missing action"
    name = action["action"]
    allowed = set(allowed_tools) | set(allowed_actions)
    if name not in allowed:
        return f"action {name} not in allowed"
    if name == "MakePlan":
        mode = action.get("mode")
        if mode not in PLAN_MODES:
            return "MakePlan.mode invalid"
        if int(action.get("pano_id", -1)) not in PANO_IDS:
            return "MakePlan.pano_id invalid"
        if mode == "semantic" and not action.get("object_query"):
            return "semantic requires object_query"
        if mode == "semantic" and semantic_query_blocked(action.get("object_query")):
            return "semantic object_query is a passage word"
        if mode == "frontier" and action.get("object_query") not in (None, ""):
            return "frontier requires object_query=null"
    if name == "TraceBack" and action.get("node_id") is None:
        return "TraceBack requires node_id"
    if name == "Verify":
        try:
            pid = int(action.get("pano_id", -1))
        except (TypeError, ValueError):
            return "Verify.pano_id invalid"
        if pid not in PANO_IDS:
            return "Verify.pano_id invalid"
    if name == "Locate":
        try:
            pid = int(action.get("pano_id", -1))
        except (TypeError, ValueError):
            return "Locate.pano_id invalid"
        if pid not in PANO_IDS:
            return "Locate.pano_id invalid"
    if name == "Look" and action.get("look") not in LOOK_ACTIONS:
        return "Look.look invalid"
    if name == "Depth" and action.get("object") is None and action.get("instance_id") is None:
        return "Depth needs object or instance_id"
    return None


def semantic_query_blocked(query):
    """语义查询词是否属于门、走廊等通道说法。

    Args:
        query: 查询词。

    Returns:
        bool: 应拒绝为 True。
    """
    text = " ".join(str(query or "").strip().lower().split())
    if not text:
        return False
    for phrase in _SEMANTIC_QUERY_PHRASES:
        if phrase in text:
            return True
    tokens = re.findall(r"[a-z0-9]+", text)
    return any(tok in _SEMANTIC_QUERY_TOKENS for tok in tokens)


def make_seg_retry_caption(plan):
    """生成写回规划器的分割失败说明。

    Args:
        plan (dict): 上一次规划。

    Returns:
        str: 用户消息正文。
    """
    query = str((plan or {}).get("object_query") or "").strip()
    slim = {k: (plan or {}).get(k) for k in ("action", "pano_id", "mode", "object_query", "plan")}
    body = json.dumps(jsonable(slim), ensure_ascii=False)
    return (
        f"上一次的makeplan：\n{body}\n"
        f"无法有效识别到物体{query}，请尝试其他的方案。"
    )


def _as_bool(value):
    """把模型常见写法转成 bool；无法识别则 None。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def validate_observe(parsed):
    """检查六向 Observe JSON。

    Args:
        parsed: extract_json 结果。

    Returns:
        tuple: ``(error_or_None, views_list_or_None)``。
    """
    if not isinstance(parsed, dict) or not isinstance(parsed.get("views"), list):
        return "missing views", None
    by_id = {}
    for row in parsed["views"]:
        if not isinstance(row, dict):
            return "view not object", None
        try:
            pid = int(row.get("pano_id"))
        except (TypeError, ValueError):
            return "bad pano_id", None
        if pid not in PANO_IDS:
            return f"pano_id {pid} invalid", None
        gf = _as_bool(row.get("goal_find"))
        if gf is None:
            return f"pano {pid} goal_find not bool", None
        landmark = row.get("landmark")
        room_type = row.get("room_type")
        if not isinstance(landmark, str) or not landmark.strip():
            return f"pano {pid} landmark empty", None
        if not isinstance(room_type, str) or not room_type.strip():
            return f"pano {pid} room_type empty", None
        by_id[pid] = {
            "pano_id": pid,
            "goal_find": gf,
            "landmark": landmark.strip(),
            "room_type": room_type.strip(),
        }
    if set(by_id.keys()) != set(PANO_IDS):
        return "views must cover all six pano_ids", None
    views = [by_id[pid] for pid in PANO_IDS]
    return None, views
