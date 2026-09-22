"""Verify 第二视角：把两张图交给 VLM 判断是否同一实例。"""

from vlm.client import extract_json, image_part, text_part
from vlm.retry import retry_call


def vlm_same_object(client, path_a, path_b, query):
    """两视角 RGB，问是否同一物体。

    Args:
        client: ``VlmClient``。
        path_a (str): 视点 0。
        path_b (str): 侧移或转头后的视点。
        query (str): 目标类名。

    Returns:
        tuple: ``(same: bool, parsed: dict)``。
    """
    prompt = (
        f"Image 1 is view 0, image 2 is another viewpoint of a candidate {query}. "
        "Decide if they show the SAME physical instance (not just the same class). "
        'Reply JSON only: {"same": true or false, "reason": "one short sentence"}.'
    )
    messages = [{
        "role": "user",
        "content": [text_part(prompt), image_part(path_a), image_part(path_b)],
    }]

    def once():
        out = client.chat(messages)
        raw = out.get("text") or ""
        parsed = extract_json(raw)
        if not isinstance(parsed, dict) or "same" not in parsed:
            raise ValueError(f"Verify VLM 回复非法: {raw[:300]!r}")
        same = parsed.get("same")
        if not isinstance(same, bool):
            if same in (0, 1):
                same = bool(same)
            elif isinstance(same, str) and same.lower() in ("true", "false"):
                same = same.lower() == "true"
            else:
                raise ValueError("same 不是 bool")
        parsed["parse_ok"] = True
        return bool(same), parsed

    return retry_call(once, "VerifyVLM")
