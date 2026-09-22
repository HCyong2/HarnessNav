#!/usr/bin/env python
"""HM3D ObjectNav 主入口：双系统 Harness +（默认）真实 VLM Planner。"""

import argparse
import os
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness import quiet_sim  # noqa: F401  须在 habitat 之前

import habitat

from harness.config import check_split, load_config
from harness.loop import Harness
from harness.protocol import dump_json, write_brief_summary
from vlm.client import VlmClient
from vlm.log import RunLog
from vlm.mover import VlmMover
from vlm.planner import VlmPlanner

DEFAULT_OUT = "/DATA_HDD/hc/harness_outputs/p1"
METRIC_KEYS = ("distance_to_goal", "success", "spl", "soft_spl")


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="HarnessNav HM3D 评测")
    p.add_argument("--stage", default="val")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-scans", type=int, default=20)
    p.add_argument("--success-distance", type=float, default=1.0)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--run-id", default="", help="实验目录名；空则用当前时间")
    p.add_argument("--base-url", default="http://127.0.0.1:8711/v1")
    p.add_argument("--model", default="Qwen-VL")
    p.add_argument("--no-vlm", action="store_true", help="P0 脚本 Planner")
    p.add_argument("--vlm-mover", action="store_true", help="Mover 也走 VLM 选 id")
    p.add_argument("--backend", default="glee", choices=["glee", "gdino_sam"],
                   help="语义分割后端")
    p.add_argument("--no-glee", action="store_true", help="关闭分割后端")
    p.add_argument("--think", action="store_true")
    return p.parse_args(argv)


def pull_metrics(env):
    """取出可 JSON 的指标。"""
    m = env.get_metrics() or {}
    out = {}
    for k in METRIC_KEYS:
        if k in m:
            try:
                out[k] = float(m[k])
            except (TypeError, ValueError):
                out[k] = m[k]
    coll = m.get("collisions")
    if isinstance(coll, dict) and "count" in coll:
        out["collisions"] = int(coll["count"])
    return out


def build_backend(name, device):
    """按名字加载分割后端；失败则返回 None。

    Args:
        name (str): ``glee`` 或 ``gdino_sam``。
        device (str): 推理设备。

    Returns:
        SegBackend: 后端实例，失败为 ``None``。
    """
    try:
        if name == "glee":
            from perception.glee_backend import GLEEBackend
            return GLEEBackend(device=device, score_threshold=0.15, num_inst_select=15)
        if name == "gdino_sam":
            from perception.gdino_sam_backend import GDinoSAMBackend
            return GDinoSAMBackend(device=device, box_threshold=0.15,
                                   text_threshold=0.15, max_instances=15)
    except Exception as exc:
        print(f"{name} 未加载: {exc}", flush=True)
        return None
    print(f"未知分割后端: {name}", flush=True)
    return None


def main(argv=None):
    """跑若干 val episode 并写 summary。"""
    args = parse_args(argv)
    t_run = time.perf_counter()
    run_id = (args.run_id or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out, run_id)
    os.makedirs(run_dir, exist_ok=True)

    config, _fixed, yaml_success = load_config(
        stage=args.stage, episodes=args.episodes, seed=args.seed,
        gpu_id=args.gpu_id, success_distance=args.success_distance)
    problems = check_split(args.stage, config)
    if problems:
        raise SystemExit("数据路径对不上:\n  " + "\n  ".join(problems))


    client = VlmClient(base_url=args.base_url, model=args.model, think=args.think)
    planner = VlmPlanner(client)
    mover = VlmMover(client)

    env = habitat.Env(config)
    env.reset()
    backend = None if args.no_glee else build_backend(args.backend, args.device)
    if backend is not None:
        print(f"分割后端 {args.backend}", flush=True)
    records = []
    try:
        for i in range(args.episodes):
            ep_dir = os.path.join(run_dir, f"ep{i:03d}")
            os.makedirs(ep_dir, exist_ok=True)
            log = RunLog(os.path.join(ep_dir, "debug.txt"))
            goal = str(getattr(env.current_episode, "object_category", "chair"))
            scene = os.path.basename(env.current_episode.scene_id).split(".")[0]
            log.emit("Harness", f"episode {i} scene={scene} id={env.current_episode.episode_id} "
                     f"goal={goal}")
            hns = Harness(env, config, ep_dir, goal=goal, backend=backend,
                          success_distance_m=args.success_distance,
                          planner=planner, mover=mover, log=log)
            summary = hns.run(max_scans=args.max_scans)
            metrics = pull_metrics(env)
            dump_json(os.path.join(ep_dir, "metrics.json"), {
                "scene": scene, "episode_id": env.current_episode.episode_id,
                "goal": goal, "summary": summary, "metrics": metrics,
                "time_cost": summary.get("time_cost"),
                "tool_counts": summary.get("tool_counts"),
                "topdown_mp4": os.path.join(ep_dir, "topdown.mp4"),
                "topdown_frames": summary.get("topdown_frames"),
                "planner_txt": os.path.join(ep_dir, "planner.txt"),
                "aborted": summary.get("aborted"),
                "abort_reason": summary.get("abort_reason"),
            })
            log.emit("Harness", f"metrics {_short(metrics)}")
            records.append({
                "ep": i, "scene": scene, "episode_id": env.current_episode.episode_id,
                "goal": goal, "dir": ep_dir, **metrics,
                "time_cost": summary.get("time_cost"),
                "tool_counts": summary.get("tool_counts"),
            })
            print(f"=== ep{i:03d} success={metrics.get('success')} "
                  f"spl={metrics.get('spl')} state={summary['state']}"
                  f"{' aborted' if summary.get('aborted') else ''} ===", flush=True)
            if i + 1 < args.episodes:
                env.reset()
    finally:
        if backend is not None:
            backend.close()
        env.close()

    means = {}
    for k in METRIC_KEYS:
        vals = [float(r[k]) for r in records if k in r and r[k] is not None]
        means[k] = sum(vals) / max(len(vals), 1) if vals else None
    total_s = time.perf_counter() - t_run
    times = [float(r["time_cost"]) for r in records
             if r.get("time_cost") is not None]
    mean_ep = sum(times) / len(times) if times else None
    dump_json(os.path.join(run_dir, "summary.json"), {
        "n": len(records), "means": means, "yaml_success": yaml_success,
        "success_distance": args.success_distance, "no_vlm": args.no_vlm,
        "backend": None if args.no_glee else args.backend,
        "total_time_s": round(total_s, 1),
        "mean_episode_time_s": None if mean_ep is None else round(mean_ep, 1),
        "episodes": records,
    })
    write_brief_summary(run_dir, records, means, total_s)
    print("=" * 60, flush=True)
    print(f"run {run_dir}  n={len(records)}", flush=True)
    for k, v in means.items():
        if v is not None:
            print(f"  mean {k}: {v:.4f}", flush=True)
    print(f"  success_rate target 0.60  got {means.get('success')}", flush=True)
    print(f"  total_time_s {total_s:.1f}  mean_episode_time_s {mean_ep}",
          flush=True)
    return 0


def _short(obj):
    """短 JSON。"""
    import json
    from harness.protocol import jsonable
    return json.dumps(jsonable(obj), ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
