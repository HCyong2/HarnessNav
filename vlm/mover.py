"""System 2：可选的 VLM 选点（只回 candidates 里的 id）。"""

import json
import os

from vlm.client import content_from_paths, extract_json
from vlm.retry import retry_call

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "mover.txt")


class VlmMover:
    """看标注图，从候选里选一个 id。"""

    def __init__(self, client):
        """初始化。

        Args:
            client (VlmClient): HTTP 客户端。
        """
        self.client = client

    def pick(self, mover_in, ego_path=None):
        """选出 ``{"id": ...}``。

        Args:
            mover_in (dict): MoverIn。
            ego_path (str, optional): 标注图路径。

        Returns:
            dict: ``{"id": ...}``。
        """
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
            sys_txt = f.read()
        caption = "MoverIn:\n" + json.dumps(mover_in, ensure_ascii=False)
        depth_map = mover_in.get("depth_map")
        if depth_map:
            caption = f"depths: {depth_map}\n" + caption
        paths = [ego_path] if ego_path and os.path.isfile(ego_path) else []
        messages = [
            {"role": "system", "content": sys_txt},
            {"role": "user", "content": content_from_paths(paths, caption) if paths
             else caption},
        ]
        ids = {c["id"] for c in (mover_in.get("candidates") or []) if c.get("id")}

        def once():
            out = self.client.chat(messages)
            parsed = extract_json(out.get("text") or "")
            if not isinstance(parsed, dict) or parsed.get("id") not in ids:
                raise ValueError(f"Mover 回复非法: {out.get('text')!r}")
            return {"id": parsed["id"]}

        return retry_call(once, "Mover")
