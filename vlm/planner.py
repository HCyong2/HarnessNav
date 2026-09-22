"""System 1：VLM Planner。"""

import json
import os
import re

from harness.protocol import jsonable, validate_observe
from vlm.client import content_from_paths, extract_json
from vlm.tools import planner_tools

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "planner.txt")
READONLY = ("Depth", "Look", "Recall")

_OBSERVE_HINT = (
    "First reply JSON only, no tools: "
    '{"views":[{"pano_id":0,"goal_find":false,"landmark":"...","room_type":"..."}, ...]} '
    "with all six pano_id in {0,2,4,6,8,10}."
)


def _load_prompt(goal):
    """读 planner.txt 并填目标词。"""
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().replace("<goal>", str(goal))


def extract_reasoning(text, api_reasoning=None):
    """从正文或接口字段抠 reasoning 段。"""
    chunks = []
    raw = (text or "").strip()
    plain = raw
    brace = raw.find("{")
    if brace >= 0:
        plain = raw[:brace]
        parsed = extract_json(raw)
        if isinstance(parsed, dict):
            field = parsed.get("reasoning")
            if isinstance(field, str) and field.strip():
                chunks.append(field.strip())
    m = re.search(r"(?is)\breasoning:\s*(.*)$", plain.strip())
    if m:
        body = m.group(1).strip()
        if body and body not in chunks:
            chunks.append(body)
    extra = (api_reasoning or "").strip() if isinstance(api_reasoning, str) else ""
    if extra and extra not in chunks:
        chunks.append(extra)
    return "\n".join(chunks)


def _action_from_reply(out):
    """tool_call 或正文 JSON -> action。"""
    out["reasoning_text"] = extract_reasoning(
        out.get("text") or "", out.get("reasoning"))
    if out.get("tool_calls"):
        tc = out["tool_calls"][0]
        action = {"action": tc["name"]}
        args = tc.get("arguments") or {}
        if isinstance(args, dict):
            args = dict(args)
            if tc["name"] == "Look" and "look" not in args and "action" in args:
                args["look"] = args.pop("action")
            args.pop("action", None)
            args.pop("reasoning", None)
            action.update(args)
        return action
    parsed = extract_json(out.get("text") or "")
    if isinstance(parsed, dict):
        parsed = dict(parsed)
        parsed.pop("reasoning", None)
        if "action" in parsed:
            return parsed
        if parsed.get("id") and "action" not in parsed:
            return None
        return parsed
    return None


class VlmPlanner:
    """多轮工具调用后给出一个终态 action。"""

    def __init__(self, client):
        """初始化。

        Args:
            client (VlmClient): HTTP 客户端。
        """
        self.client = client
        self.messages = []
        self.tools = []
        self.history = []
        self.payload = None
        self._pending = None

    def start(self, payload, pano_path, bev_path, extra_captions=""):
        """开始一拍：六向拼图 + BEV + PlannerIn。

        Args:
            payload (dict): PlannerIn。
            pano_path (str): 已缩放到 960×480 的六向拼图。
            bev_path (str): 俯视图路径。
            extra_captions (str): 附加说明。
        """
        self.payload = payload
        self.tools = planner_tools(payload.get("allowed_tools") or [],
                                   payload.get("allowed_actions") or [])
        self.history = []
        self._pending = None
        goal = payload.get("goal", "object")
        labels = []
        paths = []
        if pano_path:
            paths.append(pano_path)
            labels.append("Panorama dirs 0,2,4,6,8,10")
        if bev_path:
            paths.append(bev_path)
            labels.append("Topdown")
        caption = (
            "Images in order: " + ", ".join(labels) + ".\n"
            + _OBSERVE_HINT + "\n"
            + extra_captions + "\nPlannerIn:\n"
            + json.dumps(jsonable(payload), ensure_ascii=False)
        )
        self.messages = [
            {"role": "system", "content": _load_prompt(goal)},
            {"role": "user", "content": content_from_paths(paths, caption)},
        ]

    def set_allowed(self, tools, actions):
        """Observe 后按新状态刷新工具列表。"""
        self.tools = planner_tools(tools or [], actions or [])

    def _record(self, out):
        """记下用量，不改 messages。"""
        reason = out.get("reasoning_text")
        if not reason:
            reason = extract_reasoning(out.get("text") or "", out.get("reasoning"))
        self.history.append({
            "usage": out.get("usage"),
            "elapsed_s": out.get("elapsed_s"),
            "text": out.get("text"),
            "reasoning": reason,
            "tool_calls": out.get("tool_calls"),
            "finish": out.get("finish"),
        })

    def _commit_out(self, out):
        """把 assistant 写入对话。"""
        assistant = {"role": "assistant", "content": out.get("text") or ""}
        if out.get("tool_calls"):
            assistant["tool_calls"] = [
                {
                    "id": c.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": c["name"],
                        "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False),
                    },
                }
                for i, c in enumerate(out["tool_calls"])
            ]
        self.messages.append(assistant)
        self._last_out = out

    def commit_pending(self):
        """校验通过后提交上一轮回复。"""
        out = self._pending
        self._pending = None
        if out is not None:
            self._commit_out(out)

    def last_reasoning(self):
        """上一轮已提交回复里的 reasoning 正文。"""
        out = getattr(self, "_last_out", None) or {}
        text = out.get("reasoning_text")
        if text:
            return text
        return extract_reasoning(out.get("text") or "", out.get("reasoning"))

    def observe(self):
        """无工具一轮，解析六向 views；不合格不写入 messages。"""
        out = self.client.chat(self.messages, tools=None)
        self._record(out)
        parsed = extract_json(out.get("text") or "")
        err, views = validate_observe(parsed)
        if err:
            raise ValueError(err)
        self._commit_out(out)
        return views

    def feed_observe(self, views, payload):
        """把 Observe 结果与更新后的 PlannerIn 接回对话。"""
        self.payload = payload
        body = json.dumps(jsonable({"views": views}), ensure_ascii=False)
        caption = (
            "Observe recorded: " + body + "\nUpdated PlannerIn:\n"
            + json.dumps(jsonable(payload), ensure_ascii=False)
            + "\nAfter Observe: write a short English paragraph starting with "
            '"reasoning:" (plain text, no braces), then either one tool call '
            "(Step 2, at most 3) or one terminal JSON with key \"action\" (Step 3)."
        )
        self.messages.append({"role": "user", "content": caption})

    def propose(self):
        """调一次 VLM；校验前不写入 messages。"""
        out = self.client.chat(self.messages, tools=self.tools or None)
        action = _action_from_reply(out)
        self._record(out)
        self._pending = out
        return action

    def feed_skill(self, action, result, extra_paths=None):
        """把 Skill JSON（及可选新图）接回对话。

        Args:
            action (dict): 刚才的工具 action。
            result (dict): Skill 返回（不含大数组）。
            extra_paths (list): 追加图路径。
        """
        extra_paths = [p for p in (extra_paths or []) if p and os.path.isfile(p)]
        body = json.dumps(jsonable(result), ensure_ascii=False)
        calls = (getattr(self, "_last_out", {}) or {}).get("tool_calls") or []
        tid = None
        if calls:
            tid = calls[0].get("id")
        if tid:
            self.messages.append({
                "role": "tool",
                "tool_call_id": tid,
                "content": body,
            })
        caption = (
            f"Skill {action.get('action')} returned: {body}\n"
            "Write reasoning: then another tool, or the terminal action."
        )
        if extra_paths:
            self.messages.append({
                "role": "user",
                "content": content_from_paths(extra_paths, caption),
            })
        elif not tid:
            self.messages.append({"role": "user", "content": caption})

    def feed_user(self, caption):
        """把一段说明接回对话。

        Args:
            caption (str): 用户侧正文。
        """
        self.messages.append({"role": "user", "content": str(caption)})

    def feed_plan_retry(self, caption):
        """把语义分割失败说明接回对话，请规划器改方案。

        Args:
            caption (str): 上次规划与失败说明。
        """
        self.messages.append({
            "role": "user",
            "content": (
                caption
                + "\nWrite reasoning: then another tool if needed, or one new terminal action. "
                "Do not repeat the same MakePlan."
            ),
        })

    def dump_chat(self):
        """可 JSON 序列化的对话摘要（不含图片字节）。"""
        slim = []
        for m in self.messages:
            role = m.get("role")
            item = {"role": role}
            content = m.get("content")
            if isinstance(content, str):
                item["content"] = content
            elif isinstance(content, list):
                item["content"] = [
                    {"type": "image"} if isinstance(p, dict) and p.get("type") == "image_url"
                    else p for p in content
                ]
            if m.get("tool_calls"):
                item["tool_calls"] = m["tool_calls"]
            if m.get("tool_call_id"):
                item["tool_call_id"] = m["tool_call_id"]
            slim.append(item)
        return {"messages": slim, "rounds": self.history}

    def transcript_for_summary(self):
        """Summary 不再使用完整对话；保留空结构以免旧调用方报错。"""
        return {"messages": [], "rounds": []}
