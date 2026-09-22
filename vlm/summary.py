"""把本拍 Planner 交互压成 History 摘要。"""

import json
import os

from harness.protocol import jsonable, round_sig
from vlm.client import extract_json
from vlm.retry import retry_call

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "summary.txt")
_REASON_MAX = 400
_STR_MAX = 200


def _load_prompt(goal):
    """读 summary.txt 并填目标词。"""
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().replace("<goal>", str(goal))


def _clip(text, limit):
    """截断长字符串。"""
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


def _slim_tool_result(result):
    """工具返回只留短字段。"""
    if not isinstance(result, dict):
        return {"raw": _clip(result, _STR_MAX)}
    out = {}
    if "ok" in result:
        out["ok"] = result["ok"]
    if result.get("error") is not None:
        out["error"] = _clip(result.get("error"), _STR_MAX)
    insts = result.get("instances") or []
    if insts:
        rows = []
        for item in insts[:8]:
            if not isinstance(item, dict):
                continue
            rows.append({
                "id": item.get("id"),
                "depth_m": round_sig(item.get("depth_m")),
                "geodesic_m": round_sig(item.get("geodesic_m")),
                "score": round_sig(item.get("score")),
            })
        out["instances"] = rows
    if result.get("pano_id") is not None:
        out["pano_id"] = result.get("pano_id")
    if result.get("status") is not None:
        out["status"] = result.get("status")
    return out


def compress_bundle(views, tool_log, action, reasoning, rec):
    """Summary 用的短 JSON，不含完整对话。

    Args:
        views (list): 六向 Observe（与 PlannerIn.views 同结构）。
        tool_log (list): 本圈只读工具记录。
        action (dict): 终态动作。
        reasoning (str): 终态 reasoning。
        rec (dict): 执行结果。

    Returns:
        dict: 压缩后的拍摘要。
    """
    tools = []
    for item in tool_log or []:
        args = {k: v for k, v in (item.get("args") or {}).items()
                if k != "reasoning"}
        tools.append({
            "action": item.get("action"),
            "args": args,
            "result": _slim_tool_result(item.get("result")),
            "reasoning": _clip(item.get("reasoning"), _REASON_MAX),
        })
    rec = rec if isinstance(rec, dict) else {}
    mover = rec.get("mover") or {}
    execution = {
        "state": rec.get("state"),
        "node_id": rec.get("node_id"),
        "action": rec.get("action"),
        "mover_status": mover.get("status"),
        "dist_moved_m": round_sig(mover.get("dist_moved_m")),
    }
    if rec.get("rejected"):
        execution["rejected"] = True
        if rec.get("error"):
            execution["error"] = _clip(rec.get("error"), _STR_MAX)
    final = dict(action or {})
    final.pop("reasoning", None)
    return {
        "views": jsonable(views or []),
        "tools": tools,
        "final_action": jsonable(final),
        "final_reasoning": _clip(reasoning, _REASON_MAX),
        "execution": jsonable(execution),
    }


def summarize(client, bundle, goal):
    """无图调用，返回 1–3 句英文。

    Args:
        client: ``VlmClient``。
        bundle (dict): 已压缩的拍摘要。
        goal (str): 导航目标词。

    Returns:
        str: 摘要。
    """
    messages = [
        {"role": "system", "content": _load_prompt(goal)},
        {"role": "user", "content": json.dumps(jsonable(bundle), ensure_ascii=False)},
    ]

    def once():
        out = client.chat(messages, tools=None)
        parsed = extract_json(out.get("text") or "")
        if isinstance(parsed, dict):
            text = parsed.get("summary")
            if isinstance(text, str) and text.strip():
                return text.strip()
        raise ValueError(f"Summary 回复非法: {out.get('text')!r}")

    return retry_call(once, "Summary")
