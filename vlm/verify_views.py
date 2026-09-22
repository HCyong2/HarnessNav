"""Verify 双图：靠近前后是否都含同一导航目标。"""

import os

from vlm.client import extract_json, image_part, text_part
from vlm.retry import retry_call

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "verify.txt")


def _load_verify_prompt(goal):
    """读入核对提示词并替换目标名。"""
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return text.replace("<goal>", str(goal))


def vlm_verify_pair(client, path_a, path_b, query):
    """两张靠近前后的图，判断是否都含同一目标实例。

    Args:
        client: ``VlmClient``。
        path_a (str): 靠近前。
        path_b (str): 靠近后。
        query (str): 目标类名。

    Returns:
        tuple: ``(ok: bool, parsed: dict)``。
    """
    prompt = _load_verify_prompt(query)
    messages = [{
        "role": "user",
        "content": [text_part(prompt), image_part(path_a), image_part(path_b)],
    }]

    def once():
        out = client.chat(messages)
        raw = out.get("text") or ""
        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Verify VLM 回复非法: {raw[:300]!r}")
        needed = ("goal_in_view0", "goal_in_view1", "same_instance")
        for key in needed:
            if key not in parsed:
                raise ValueError(f"Verify VLM 缺字段 {key}: {raw[:300]!r}")
            val = parsed[key]
            if not isinstance(val, bool):
                if val in (0, 1):
                    parsed[key] = bool(val)
                elif isinstance(val, str) and val.lower() in ("true", "false"):
                    parsed[key] = val.lower() == "true"
                else:
                    raise ValueError(f"{key} 不是 bool")
        parsed["parse_ok"] = True
        ok = bool(parsed["goal_in_view0"] and parsed["goal_in_view1"]
                  and parsed["same_instance"])
        return ok, parsed

    return retry_call(once, "VerifyVLM")


def vlm_same_object(client, path_a, path_b, query):
    """兼容旧接口：内部改走双图核对。"""
    return vlm_verify_pair(client, path_a, path_b, query)
