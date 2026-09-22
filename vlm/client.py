"""OpenAI 兼容的多图 chat 客户端。"""

import base64
import json
import os
import time


def _b64_file(path):
    """文件 -> data URL。"""
    with open(path, "rb") as f:
        raw = f.read()
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def image_part(path):
    """OpenAI image_url 块。"""
    return {"type": "image_url", "image_url": {"url": _b64_file(path)}}


def text_part(text):
    """OpenAI 文本块。"""
    return {"type": "text", "text": str(text)}


def extract_json(text):
    """从回复抠 JSON 对象。"""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class VlmClient:
    """连 vLLM ``/v1/chat/completions``。"""

    def __init__(self, base_url="http://127.0.0.1:8711/v1", model="Qwen-VL",
                 api_key="EMPTY", think=False, max_tokens=800):
        """初始化。

        Args:
            base_url (str): OpenAI 兼容地址。
            model (str): 模型名。
            api_key (str): 占位密钥。
            think (bool): 是否打开 thinking。
            max_tokens (int): 生成上限。
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.think = bool(think)
        self.max_tokens = int(max_tokens)
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=api_key)
        except ImportError:
            self._client = None

    def chat(self, messages, tools=None):
        """发一轮对话。

        Args:
            messages (list): OpenAI messages。
            tools (list, optional): function schema。

        Returns:
            dict: text / tool_calls / usage / elapsed_s。
        """
        if self._client is None:
            raise RuntimeError("需要 openai 包：pip install openai")
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": self.think}},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - t0
        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append({
                "id": getattr(tc, "id", None),
                "name": tc.function.name,
                "arguments": args or {},
            })
        usage = {}
        if resp.usage is not None:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        reasoning = getattr(msg, "reasoning_content", None) or ""
        return {
            "text": msg.content or "",
            "reasoning": reasoning if isinstance(reasoning, str) else "",
            "tool_calls": calls,
            "usage": usage,
            "elapsed_s": round(elapsed, 3),
            "finish": resp.choices[0].finish_reason,
        }


def content_from_paths(paths, caption):
    """路径列表 + 说明 -> user content。"""
    parts = []
    for p in paths:
        if p and os.path.isfile(p):
            parts.append(image_part(p))
    parts.append(text_part(caption))
    return parts
