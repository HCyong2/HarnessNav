#!/usr/bin/env python
"""用 vLLM 上的 Qwen3.6-27B（VLM）测全景、BEV、标注选点与 Skill 工具调用。

默认连 ``http://127.0.0.1:8711/v1``。启动服务（相对用户给出的命令：把每请求图像上限
提到 8，并打开官方 tool parser，否则 6 张分方向图和工具调用跑不起来）：

.. code-block:: bash

    conda activate qwen3
    cd /DATA_HDD/hc/llm/qwen
    CUDA_VISIBLE_DEVICES=2,3 vllm serve /DATA_HDD/hc/llm/qwen/Qwen3.6-27B --port 8711 \\
      --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.95 \\
      --max-num-seqs 16 --dtype bfloat16 --limit-mm-per-prompt '{"image": 8}' \\
      --enable-prefix-caching --reasoning-parser qwen3 \\
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \\
      --served-model-name Qwen-VL
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from openai import OpenAI
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness.protocol import LOOK_ACTIONS, PANO_IDS, PLAN_MODES, validate_planner_action

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
DEFAULT_OUT = os.path.join(HERE, "output")


def dump_json(path, obj):
    """写 UTF-8 JSON。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


PANO_STITCH = os.path.join(REPO_ROOT, "test", "occupancy", "out", "scan0_pano.png")
BEV_COLOR = os.path.join(REPO_ROOT, "test", "occupancy", "out", "scan1_bev_color.png")
NODES_BEV = os.path.join(REPO_ROOT, "test", "memory", "nodes", "output", "nodes_bev.png")
FRONTIER_BEV = os.path.join(REPO_ROOT, "test", "frontier", "output", "bev.png")
FRONTIER_EGO = os.path.join(REPO_ROOT, "test", "frontier", "output", "ego.png")
FRONTIER_META = os.path.join(REPO_ROOT, "test", "frontier", "output", "meta.json")
GLEE_EGO = os.path.join(REPO_ROOT, "test", "skills", "depth", "output", "depth_anno.png")
GLEE_META = os.path.join(REPO_ROOT, "test", "skills", "depth", "output", "meta.json")
LOOK_DOWN = os.path.join(REPO_ROOT, "test", "skills", "look", "output", "look_down.png")

PANO_PROMPT = (
    "You are looking at indoor ObjectNav panoramas. Even directions are "
    "0,2,4,6,8,10. Direction 0 is the agent's current heading. "
    "For EACH direction describe: scene (room/corridor), main objects, "
    "openings (door/window/hallway), navigable (true/false). "
    "Reply with one JSON object only, keys \"0\",\"2\",\"4\",\"6\",\"8\",\"10\". "
    "Each value: {\"scene\":str,\"objects\":[str],\"opening\":str,\"navigable\":bool}."
)

SKILL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Depth",
            "description": "Measure GLEE instance depth in a panorama direction. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pano_id": {"type": "integer", "enum": list(PANO_IDS)},
                    "object": {"type": "string", "description": "Category word, e.g. door."},
                    "instance_id": {"type": "string", "description": "Optional id like door_1."},
                },
                "required": ["pano_id", "object"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Look",
            "description": "Pitch or yaw one discrete step. Use down to see the floor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "look": {"type": "string", "enum": ["up", "down", "left", "right"]},
                },
                "required": ["look"],
            },
        },
    },
    {
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
    {
        "type": "function",
        "function": {
            "name": "Verify",
            "description": "Multi-view check that an instance is the navigation object. Only in Find.",
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
    {
        "type": "function",
        "function": {
            "name": "MakePlan",
            "description": "Lock a subgoal: face pano_id then let Mover follow semantic or frontier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pano_id": {"type": "integer", "enum": list(PANO_IDS)},
                    "mode": {"type": "string", "enum": ["semantic", "frontier"]},
                    "object_query": {"type": "string"},
                    "plan": {"type": "string"},
                },
                "required": ["pano_id", "mode"],
            },
        },
    },
    {
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
    {
        "type": "function",
        "function": {
            "name": "Stop",
            "description": "End the episode. Only when state is Arrived.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="Qwen-VL 全景/BEV/选点/工具调用评测")
    p.add_argument("--base-url", default="http://127.0.0.1:8711/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default="Qwen-VL")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--suite", default="all",
                   choices=("all", "pano", "bev", "pick", "tools"))
    p.add_argument("--think", action="store_true", help="打开 Qwen thinking")
    p.add_argument("--wait-s", type=int, default=0,
                   help="等待服务就绪的秒数；0 表示不等")
    p.add_argument("--max-tokens", type=int, default=1200)
    return p.parse_args(argv)


def file_b64(path):
    """读文件为 data URL。"""
    with open(path, "rb") as f:
        raw = f.read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def image_part(path):
    """OpenAI 多模态 image_url 块。"""
    return {"type": "image_url", "image_url": {"url": file_b64(path)}}


def text_part(text):
    """OpenAI 文本块。"""
    return {"type": "text", "text": text}


def split_pano_collage(src, dst_dir, indices=PANO_IDS, gap=10):
    """把 3×2 拼接全景裁成 6 张分方向图。

    Args:
        src (str): 拼接图路径。
        dst_dir (str): 输出目录。
        indices (sequence): 朝向序号，从左到右、从上到下。
        gap (int): 与 ``concat_panorama`` 相同的间距。

    Returns:
        list: 写出的路径，与 ``indices`` 对齐。
    """
    os.makedirs(dst_dir, exist_ok=True)
    im = Image.open(src).convert("RGB")
    n = len(indices)
    cols = min(3, n)
    rows = int((n + cols - 1) / cols)
    tw = int(round((im.width - gap * (cols + 1)) / float(cols)))
    th = int(round((im.height - gap * (rows + 1)) / float(rows)))
    paths = []
    for i, pid in enumerate(indices):
        r, c = divmod(i, cols)
        x0 = gap * (c + 1) + c * tw
        y0 = gap * (r + 1) + r * th
        crop = im.crop((x0, y0, x0 + tw, y0 + th))
        out = os.path.join(dst_dir, f"dir_{pid}.png")
        crop.save(out)
        paths.append(out)
    return paths


def prepare_fixtures():
    """从已有可视化产物裁分方向图，并核对照片是否存在。

    Returns:
        dict: 路径表。
    """
    os.makedirs(FIX, exist_ok=True)
    needed = {
        "pano_stitch": PANO_STITCH,
        "bev": BEV_COLOR if os.path.isfile(BEV_COLOR) else NODES_BEV,
        "nodes_bev": NODES_BEV if os.path.isfile(NODES_BEV) else BEV_COLOR,
        "frontier_bev": FRONTIER_BEV,
        "frontier_ego": FRONTIER_EGO,
        "glee_ego": GLEE_EGO,
    }
    missing = [k for k, v in needed.items() if not os.path.isfile(v)]
    if missing:
        raise SystemExit("缺少测试图 " + ", ".join(missing) + "，先跑 test/test.md 里对应脚本")
    dirs = split_pano_collage(PANO_STITCH, os.path.join(FIX, "dirs"))
    needed["dirs"] = dirs
    return needed


def wait_ready(base_url, timeout_s):
    """轮询 ``/models`` 直到服务起来。"""
    url = base_url.rstrip("/") + "/models"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(5)
    return False


def extract_json(text):
    """从模型回复里抠第一段 JSON。"""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def usage_dict(resp):
    """抽出 token 用量。"""
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    prompt_d = getattr(u, "prompt_tokens_details", None)
    img_tokens = None
    if prompt_d is not None:
        img_tokens = getattr(prompt_d, "image_tokens", None)
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "total_tokens": getattr(u, "total_tokens", None),
        "image_tokens": img_tokens,
    }


def chat(client, model, content, tools=None, think=False, max_tokens=1200):
    """发一轮 chat.completions，记耗时与用量。

    Args:
        client: OpenAI 客户端。
        model (str): 模型名。
        content (list): user content 块。
        tools (list, optional): 工具 schema。
        think (bool): 是否保留 thinking。
        max_tokens (int): 生成上限。

    Returns:
        dict: text / tool_calls / usage / elapsed_s / raw_finish。
    """
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": bool(think)}},
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
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
        calls.append({"name": tc.function.name, "arguments": args})
    text = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None)
    return {
        "text": text,
        "reasoning": reasoning,
        "tool_calls": calls,
        "usage": usage_dict(resp),
        "elapsed_s": round(elapsed, 3),
        "finish": resp.choices[0].finish_reason,
    }


def load_json(path, default=None):
    """读 JSON 文件。"""
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_pano(client, args, fixtures):
    """拼接全景 vs 6 张分方向图。"""
    dirs = fixtures["dirs"]
    stitch_content = [
        image_part(fixtures["pano_stitch"]),
        text_part("This is a 2x3 collage. Row-major order is directions "
                  + ",".join(str(i) for i in PANO_IDS) + ". Each tile is labeled. "
                  + PANO_PROMPT),
    ]
    seq_content = []
    for pid, path in zip(PANO_IDS, dirs):
        seq_content.append(image_part(path))
        seq_content.append(text_part(f"Image above = Direction {pid}."))
    seq_content.append(text_part(PANO_PROMPT))

    stitch = chat(client, args.model, stitch_content, think=args.think,
                  max_tokens=args.max_tokens)
    sequential = chat(client, args.model, seq_content, think=args.think,
                      max_tokens=args.max_tokens)
    js = extract_json(stitch["text"])
    jq = extract_json(sequential["text"])
    keys_ok = lambda d: isinstance(d, dict) and all(str(k) in d for k in PANO_IDS)
    return {
        "stitch": {**stitch, "parsed": js, "has_all_dirs": keys_ok(js)},
        "sequential": {**sequential, "parsed": jq, "has_all_dirs": keys_ok(jq)},
        "compare": {
            "prompt_tokens": {
                "stitch": stitch["usage"].get("prompt_tokens"),
                "sequential": sequential["usage"].get("prompt_tokens"),
            },
            "image_tokens": {
                "stitch": stitch["usage"].get("image_tokens"),
                "sequential": sequential["usage"].get("image_tokens"),
            },
            "elapsed_s": {
                "stitch": stitch["elapsed_s"],
                "sequential": sequential["elapsed_s"],
            },
            "note": "sequential 应更吃 image token、更慢；看 parsed 里 objects 是否更细。",
        },
    }


def test_bev(client, args, fixtures):
    """BEV 节点/轨迹/可走区，以及扇区与全景对应。"""
    content = [
        image_part(fixtures["nodes_bev"]),
        text_part(
            "Topdown BEV. Red triangle = agent heading. Blue rings + numbers = "
            "past scan nodes. Blue lines = trajectory. Cyan rays + red numbers "
            "0,2,4,6,8,10 = 60-degree sectors (0 is forward). Green F* = frontiers. "
            "Colored points = observed surfaces; black = unseen."
        ),
        image_part(fixtures["pano_stitch"]),
        text_part(
            "Six-tile panorama, same even directions as BEV sectors. "
            "Answer JSON only: "
            "{\"agent_heading\":\"what sector 0 faces in the pano\","
            "\"nodes\":\"how many blue node ids you see besides current\","
            "\"free_vs_obstacle\":\"how you tell walkable floor from walls\","
            "\"sector_to_pano\":{\"0\":\"...\",\"2\":\"...\",\"4\":\"...\","
            "\"6\":\"...\",\"8\":\"...\",\"10\":\"...\"},"
            "\"frontier_hint\":\"which even sector has a green F closest to heading\"}."
        ),
    ]
    out = chat(client, args.model, content, think=args.think, max_tokens=args.max_tokens)
    out["parsed"] = extract_json(out["text"])
    return out


def _pixel_err(pred, gt):
    """像素欧氏距离；缺字段则为 None。"""
    if not isinstance(pred, dict) or gt is None:
        return None
    uv = pred.get("uv")
    if not (isinstance(uv, (list, tuple)) and len(uv) >= 2):
        return None
    return round(((float(uv[0]) - gt[0]) ** 2 + (float(uv[1]) - gt[1]) ** 2) ** 0.5, 1)


def test_pick(client, args, fixtures):
    """前沿绿点与 GLEE 框，让模型报 uv。"""
    ft = load_json(FRONTIER_META, {})
    in_view = (ft.get("in_view") or [{}])[0]
    gt_f = in_view.get("uv")
    glee = load_json(GLEE_META, {})
    inst = (glee.get("instances") or [{}])[0]
    gt_d = inst.get("uv")

    frontier = chat(client, args.model, [
        image_part(FRONTIER_EGO),
        text_part(
            "First-person view with a green frontier marker labeled F1 (and maybe F0). "
            "Pick the pixel of F1 that a mover should walk toward. "
            "JSON only: {\"id\":\"F1\",\"uv\":[u,v]} with u=column v=row, origin top-left."
        ),
    ], think=args.think, max_tokens=256)
    frontier["parsed"] = extract_json(frontier["text"])
    frontier["gt_uv"] = gt_f
    frontier["pixel_error"] = _pixel_err(frontier["parsed"], gt_f)

    door = chat(client, args.model, [
        image_part(GLEE_EGO),
        text_part(
            "GLEE overlays on a door. Choose the instance that is the actual door leaf "
            "(highest-confidence door, not a huge wall mask). "
            "JSON only: {\"id\":\"door_k\",\"uv\":[u,v]} center pixel."
        ),
    ], think=args.think, max_tokens=256)
    door["parsed"] = extract_json(door["text"])
    door["gt_uv"] = gt_d
    door["pixel_error"] = _pixel_err(door["parsed"], gt_d)
    door["gt_id"] = inst.get("id")
    return {"frontier": frontier, "glee_door": door}


def _action_from_call(call):
    """工具调用 -> Planner action 字典。"""
    action = {"action": call["name"]}
    args = call.get("arguments") or {}
    if isinstance(args, dict):
        args = dict(args)
        if call["name"] == "Look" and "look" not in args and "action" in args:
            args["look"] = args.pop("action")
        args.pop("action", None)
        action.update(args)
    return action


def test_tools(client, args, fixtures):
    """每个 Skill / 终态动作一个引导场景。"""
    scenes = [
        {
            "name": "Depth",
            "expect": "Depth",
            "state": "Unseen",
            "tools": ["Depth", "Look", "Recall"],
            "actions": ["MakePlan", "TraceBack"],
            "images": [fixtures["pano_stitch"]],
            "user": "I see a door in direction 6 but I need its metric depth before planning. "
                    "Call the matching read-only skill. Do not MakePlan yet.",
        },
        {
            "name": "Look",
            "expect": "Look",
            "state": "Unseen",
            "tools": ["Depth", "Look", "Recall"],
            "actions": ["MakePlan", "TraceBack"],
            "images": [fixtures["pano_stitch"]],
            "user": "The floor near my feet is missing in the level pano. "
                    "I need a downward view of the carpet. Call Look.",
        },
        {
            "name": "Recall",
            "expect": "Recall",
            "state": "Unseen",
            "tools": ["Depth", "Look", "Recall"],
            "actions": ["MakePlan", "TraceBack"],
            "images": [fixtures["nodes_bev"]],
            "user": "I am at node 1. I forgot what the doorway at node 0 direction 4 looked like. "
                    "I must NOT walk back yet. Fetch the stored image.",
        },
        {
            "name": "Verify",
            "expect": "Verify",
            "state": "Find",
            "tools": ["Depth", "Look", "Recall"],
            "actions": ["Verify", "MakePlan", "TraceBack"],
            "images": [GLEE_EGO],
            "user": "State is Find. GLEE tagged door_1 as a possible navigation object. "
                    "Confirm it with the multi-view skill before making a plan.",
        },
        {
            "name": "MakePlan_frontier",
            "expect": "MakePlan",
            "state": "Unseen",
            "tools": ["Depth", "Look", "Recall"],
            "actions": ["MakePlan", "TraceBack"],
            "images": [FRONTIER_BEV, FRONTIER_EGO],
            "user": "No goal object in view. Explore the green frontier in the current FOV. "
                    "Emit the terminal action MakePlan with mode=frontier and a valid pano_id. "
                    "object_query must be null/omitted.",
        },
        {
            "name": "TraceBack",
            "expect": "TraceBack",
            "state": "Unseen",
            "tools": ["Depth", "Look", "Recall"],
            "actions": ["MakePlan", "TraceBack"],
            "images": [fixtures["nodes_bev"]],
            "user": "This node has no leftover frontiers. Walk the body back to node 0. "
                    "Recall only shows pictures and is the wrong action.",
        },
        {
            "name": "Stop",
            "expect": "Stop",
            "state": "Arrived",
            "tools": [],
            "actions": ["Stop"],
            "images": [LOOK_DOWN if os.path.isfile(LOOK_DOWN) else GLEE_EGO],
            "user": "Harness set state=Arrived. End the episode.",
        },
    ]
    rows = []
    for sc in scenes:
        content = [image_part(p) for p in sc["images"] if os.path.isfile(p)]
        planner_in = {
            "goal": "toilet",
            "state": sc["state"],
            "success_distance_m": 1.0,
            "current_node_id": 1,
            "allowed_tools": sc["tools"],
            "allowed_actions": sc["actions"],
            "history": [],
            "tool_log": [],
        }
        content.append(text_part(
            "You are System 1 Planner. Only use actions in allowed_tools / allowed_actions. "
            "Prefer a function/tool call. PlannerIn JSON:\n"
            + json.dumps(planner_in, ensure_ascii=False)
            + "\n" + sc["user"]
        ))
        out = chat(client, args.model, content, tools=SKILL_TOOLS,
                   think=args.think, max_tokens=512)
        names = [c["name"] for c in out["tool_calls"]]
        parsed_text = extract_json(out["text"])
        chosen = None
        if out["tool_calls"]:
            chosen = _action_from_call(out["tool_calls"][0])
        elif isinstance(parsed_text, dict) and parsed_text.get("action"):
            chosen = parsed_text
        err = None
        if chosen:
            err = validate_planner_action(chosen, sc["tools"], sc["actions"])
        hit = bool(chosen) and chosen.get("action") == sc["expect"] and err is None
        rows.append({
            "scene": sc["name"],
            "expect": sc["expect"],
            "hit": hit,
            "action": chosen,
            "validate_error": err,
            "tool_call_names": names,
            "elapsed_s": out["elapsed_s"],
            "usage": out["usage"],
            "text": out["text"],
            "finish": out["finish"],
        })
    n_hit = sum(1 for r in rows if r["hit"])
    return {"hits": n_hit, "n": len(rows), "scenes": rows}


def main(argv=None):
    """准备图、连 vLLM、跑所选套件并写 JSON。"""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.wait_s > 0:
        print(f"waiting for {args.base_url} up to {args.wait_s}s")
        if not wait_ready(args.base_url, args.wait_s):
            raise SystemExit("vLLM 未就绪")
    fixtures = prepare_fixtures()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    report = {"base_url": args.base_url, "model": args.model, "think": args.think}
    suite = args.suite
    if suite in ("all", "pano"):
        print("## pano")
        report["pano"] = test_pano(client, args, fixtures)
        print(json.dumps(report["pano"]["compare"], ensure_ascii=False, indent=2))
    if suite in ("all", "bev"):
        print("## bev")
        report["bev"] = test_bev(client, args, fixtures)
        print(report["bev"].get("text", "")[:800])
    if suite in ("all", "pick"):
        print("## pick")
        report["pick"] = test_pick(client, args, fixtures)
        for k in ("frontier", "glee_door"):
            row = report["pick"][k]
            print(k, "err", row.get("pixel_error"), "pred", row.get("parsed"), "gt", row.get("gt_uv"))
    if suite in ("all", "tools"):
        print("## tools")
        report["tools"] = test_tools(client, args, fixtures)
        print(f"hits {report['tools']['hits']}/{report['tools']['n']}")
        for row in report["tools"]["scenes"]:
            print(f"  {row['scene']}: hit={row['hit']} action={row['action']}")
    dump_json(os.path.join(args.out, "report.json"), report)
    print(os.path.join(args.out, "report.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
