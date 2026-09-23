#!/usr/bin/env python
"""多 GPU 并行评测：导航进程共用分割队列，VLM 共用 vLLM。"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime

import multiprocessing as mp

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness import quiet_sim  # noqa: F401  须在 habitat 之前

DEFAULT_OUT = "/DATA_HDD/hc/harness_outputs/p1"
METRIC_KEYS = ("distance_to_goal", "success", "spl", "soft_spl")
VAL_EPI_PATH = os.path.join(REPO_ROOT, "val_epi.txt")
CACHE_NS = (100, 1000)
CACHE_FORMAT = "num_episode_sample"


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(description="HarnessNav 多 Agent 并行评测")
    p.add_argument("--stage", default="val")
    p.add_argument("--episodes", type=int, default=1000, help="val 上要跑的集数")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--gpu", default="0,1", help="Habitat Agent 物理 GPU，如 0,1")
    p.add_argument("--seg-gpu", default="", help="分割 worker 物理 GPU，空则同 --gpu")
    p.add_argument("--seg-num", type=int, default=10, help="分割模型副本数")
    p.add_argument("--agent_num", "--agent-num", type=int, default=8,
                   dest="agent_num", help="并行 Agent / 进程数")
    p.add_argument("--max-scans", type=int, default=20)
    p.add_argument("--success-distance", type=float, default=1.0)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--run-id", default="", help="实验目录名；已有则续跑")
    p.add_argument("--base-url", default="http://127.0.0.1:8711/v1")
    p.add_argument("--model", default="Qwen-VL")
    p.add_argument("--backend", default="glee", choices=["glee", "gdino_sam"])
    p.add_argument("--no-glee", action="store_true", help="不加载分割模型")
    p.add_argument("--think", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="保存图片、debug.txt、topdown、planner.txt 等；默认只留 metrics.json 与 episode.json")
    p.add_argument("--stagger-s", type=float, default=2.0,
                   help="相邻进程启动间隔，减轻同时占显存")
    return p.parse_args(argv)


def parse_gpus(text):
    """``0,1`` -> ``[0, 1]``。"""
    gpus = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        gpus.append(int(part))
    if not gpus:
        raise ValueError("--gpu 为空")
    return gpus


def split_round_robin(items, n):
    """把列表均分到 n 份，互不重叠。"""
    n = max(int(n), 1)
    buckets = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


def assign_gpus(agent_num, gpus):
    """每台 GPU 尽量平分 Agent。

    Args:
        agent_num (int): 进程数。
        gpus (list): 物理 GPU 编号。

    Returns:
        list: 长度 ``agent_num``，每项是该 Agent 的物理 GPU。
    """
    n_gpu = len(gpus)
    sizes = [agent_num // n_gpu + (1 if i < agent_num % n_gpu else 0)
             for i in range(n_gpu)]
    mapping = []
    for gpu, size in zip(gpus, sizes):
        mapping.extend([gpu] * size)
    return mapping[:agent_num]


def ep_dir_name(key):
    """与 StateNav 一致：``场景名_episode_id``。"""
    return str(key)


def episode_done(run_dir, key):
    """已有完整 ``metrics.json`` 则视为跑完。"""
    path = os.path.join(run_dir, ep_dir_name(key), "metrics.json")
    return os.path.isfile(path)


def ep_key(episode):
    """scene + habitat episode_id。"""
    scene = os.path.basename(episode.scene_id).split(".")[0]
    return f"{scene}_{episode.episode_id}"


def _rows_from_names(names):
    """名称列表 -> 任务行。"""
    rows = []
    for i, key in enumerate(names):
        scene, _, eid = str(key).rpartition("_")
        rows.append({
            "index": i,
            "key": str(key),
            "scene": scene,
            "episode_id": eid,
        })
    return rows


def scan_episode_names(n, args, gpu_id):
    """按 ``num_episode_sample=n`` 建 Env，reset n 次收集名称。"""
    from harness import quiet_sim
    quiet_sim.apply_io_filter()
    import habitat
    from harness.config import check_split, load_config

    n = int(n)
    config, _fixed, yaml_success = load_config(
        stage=args.stage, episodes=n, seed=args.seed,
        gpu_id=int(gpu_id), success_distance=args.success_distance)
    problems = check_split(args.stage, config)
    if problems:
        raise SystemExit("数据路径对不上:\n  " + "\n  ".join(problems))
    env = habitat.Env(config)
    try:
        print(f"num_episode_sample={n}，reset {n} 次收集名称", flush=True)
        names = []
        for _ in range(n):
            env.reset()
            names.append(ep_key(env.current_episode))
        return names, yaml_success
    finally:
        env.close()


def _read_val_epi(path):
    """解析 ``val_epi.txt``，不校验 stage/seed。"""
    file_stage, file_seed, file_fmt = None, None, None
    sections = {}
    cur = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line[1:].strip()
                if body.startswith("stage="):
                    file_stage = body.split("=", 1)[1].strip()
                elif body.startswith("seed="):
                    file_seed = body.split("=", 1)[1].strip()
                elif body.startswith("format="):
                    file_fmt = body.split("=", 1)[1].strip()
                continue
            if line.startswith("[") and line.endswith("]"):
                tag = line[1:-1].strip()
                if tag.startswith("n="):
                    cur = int(tag.split("=", 1)[1])
                    sections[cur] = []
                else:
                    cur = None
                continue
            if cur is not None:
                sections[cur].append(line)
    return file_stage, file_seed, file_fmt, sections


def load_val_epi(path, stage):
    """读 ``val_epi.txt``。

    只校验 ``stage`` 与各档长度。集列表视为固定评测集，**不**因
    ``--seed`` 不同而失效。返回 ``(sections, 缓存写入时的 seed 或 None)``。
    旧格式把 ``[n=100]`` 当成 1000 的前缀，丢弃该档。
    """
    if not os.path.isfile(path):
        return {}, None
    file_stage, file_seed, file_fmt, sections = _read_val_epi(path)
    if file_stage != str(stage):
        return {}, file_seed
    if file_fmt != CACHE_FORMAT:
        sections.pop(100, None)
    kept = {k: v for k, v in sections.items() if len(v) >= int(k)}
    return kept, file_seed


def save_val_epi(path, stage, seed, sections):
    """按档写入；``[n=100]`` 与 ``[n=1000]`` 各自来自对应 ``num_episode_sample``。

    同 stage 已有缓存时保留原 ``seed=`` 头（集列表不动），避免换种子覆写。
    """
    merged = {}
    header_seed = seed
    if os.path.isfile(path):
        existing, file_seed = load_val_epi(path, stage)
        merged.update(existing)
        if file_seed is not None:
            header_seed = file_seed
    merged.update(sections)
    lines = [
        "# HarnessNav val episode cache",
        f"# format={CACHE_FORMAT}",
        f"# stage={stage}",
        f"# seed={header_seed}",
        "",
    ]
    for n in CACHE_NS:
        names = merged.get(n)
        if not names:
            continue
        lines.append(f"[n={n}]")
        for name in names[:n]:
            lines.append(name)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def iterator_tier(n):
    """任务数 ``n`` 对应的 Habitat ``num_episode_sample`` 档。

    ``n<=50`` 当场扫 ``n``；否则用 100 / 1000 两档采样。任务取该档前 ``n`` 个，
    worker 必须按档长 reset，不能按 ``n`` 另采一份。
    """
    n = int(n)
    if n <= 50:
        return n
    if n <= 100:
        return 100
    if n <= 1000:
        return 1000
    return n


def yaml_success_of(args, gpu_id):
    """只读 YAML，不建 Env。"""
    from harness.config import load_config
    _config, _fixed, yaml_success = load_config(
        stage=args.stage, episodes=min(int(args.episodes), 50),
        seed=args.seed, gpu_id=int(gpu_id),
        success_distance=args.success_distance)
    return yaml_success


def get_episode_names(args, gpu_id):
    """返回任务行、yaml success、worker 的 ``num_episode_sample``、采样种子。

    采样种子用于 Habitat ``num_episode_sample``：有缓存时用缓存头里的 seed，
    保证 reset 顺序与 ``val_epi.txt`` 一致；``--seed`` 不触发重扫。
    """
    n = int(args.episodes)
    n_iter = iterator_tier(n)
    if n <= 50:
        print(f"episodes={n} ≤50，当场 num_episode_sample={n_iter}", flush=True)
        names, yaml_success = scan_episode_names(n_iter, args, gpu_id)
        return _rows_from_names(names[:n]), yaml_success, n_iter, int(args.seed)

    sections, file_seed = load_val_epi(VAL_EPI_PATH, args.stage)
    cached = sections.get(n_iter)
    if cached and len(cached) >= n_iter:
        epi_seed = int(file_seed) if file_seed is not None else int(args.seed)
        msg = (f"从 {VAL_EPI_PATH} 读取 n={n_iter} 缓存，任务取前 {n} 集"
               f"（采样 seed={epi_seed}")
        if file_seed is not None and str(file_seed) != str(args.seed):
            msg += f"，忽略 --seed={args.seed}"
        print(msg + "）", flush=True)
        return (_rows_from_names(cached[:n]), yaml_success_of(args, gpu_id),
                n_iter, epi_seed)

    print(f"缓存不足，num_episode_sample={n_iter} 扫描并写入 {VAL_EPI_PATH}",
          flush=True)
    names, yaml_success = scan_episode_names(n_iter, args, gpu_id)
    if n_iter in CACHE_NS:
        save_val_epi(VAL_EPI_PATH, args.stage, args.seed, {n_iter: names})
    return _rows_from_names(names[:n]), yaml_success, n_iter, int(args.seed)


def worker_process(agent_id, gpu, tasks, args, run_dir, progress_queue, n_iter,
                   epi_seed, req_q, reply_q):
    """单个 Agent：Habitat 在本卡，分割走共享队列。"""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.environ["HABITAT_LAB_LOG"] = str(40)

    time.sleep(float(args.stagger_s) * int(agent_id))

    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"agent_{agent_id:02d}.log")
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = log_fp
    sys.stderr = log_fp
    from harness import quiet_sim
    quiet_sim.redirect_c_stdio(log_fp)
    quiet_sim.apply_io_filter()

    import habitat
    from harness.config import check_split, load_config
    from harness.loop import Harness
    from harness.protocol import dump_json
    from perception.queue_backend import QueueSegBackend
    from run_HarnessNav import _short, pull_metrics
    from vlm.client import VlmClient
    from vlm.log import RunLog
    from vlm.mover import VlmMover
    from vlm.planner import VlmPlanner

    print(f"agent={agent_id} physical_gpu={gpu} CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')} tasks={len(tasks)}",
          flush=True)

    task_set = {t["key"] for t in tasks}
    n_iter = int(n_iter)
    skipped = 0
    for t in tasks:
        if episode_done(run_dir, t["key"]):
            skipped += 1
            print(f"skip {t['key']} already done", flush=True)
            progress_queue.put(1)
    if skipped == len(tasks):
        print("no pending episodes", flush=True)
        progress_queue.put("done")
        log_fp.close()
        return

    env = None
    backend = None
    ran = 0
    try:
        config, _fixed, _yaml_success = load_config(
            stage=args.stage, episodes=n_iter, seed=int(epi_seed),
            gpu_id=0, success_distance=args.success_distance)
        problems = check_split(args.stage, config)
        if problems:
            raise RuntimeError("数据路径对不上: " + "; ".join(problems))

        env = habitat.Env(config)
        print(
            f"num_episode_sample={n_iter} tasks={len(tasks)} "
            f"(len(env.episodes)={len(env.episodes)} 仍是数据集全量，不用于遍历)",
            flush=True,
        )
        client = VlmClient(base_url=args.base_url, model=args.model, think=args.think)
        planner = VlmPlanner(client)
        mover = VlmMover(client)
        if not args.no_glee:
            backend = QueueSegBackend(
                req_q, reply_q, agent_id, name=args.backend)
            print(f"分割经队列 backend={args.backend} agent={agent_id}",
                  flush=True)

        for i in range(n_iter):
            env.reset()
            key = ep_key(env.current_episode)
            if key not in task_set:
                print(f"[{i}/{n_iter}] {key} not in the task list. Skipping.",
                      flush=True)
                continue
            if episode_done(run_dir, key):
                print(f"[{i}/{n_iter}] {key} already exists. Skipping.",
                      flush=True)
                continue
            print(f"[{i}/{n_iter}] agent {agent_id} MATCHED: {key}. Starting...",
                  flush=True)
            ep_dir = os.path.join(run_dir, ep_dir_name(key))
            os.makedirs(ep_dir, exist_ok=True)
            log = RunLog(os.path.join(ep_dir, "debug.txt") if args.debug else None)
            goal = str(getattr(env.current_episode, "object_category", "chair"))
            scene = os.path.basename(env.current_episode.scene_id).split(".")[0]
            log.emit("Harness",
                     f"agent={agent_id} gpu={gpu} {key} "
                     f"scene={scene} id={env.current_episode.episode_id} goal={goal}")
            try:
                hns = Harness(env, config, ep_dir, goal=goal, backend=backend,
                              success_distance_m=args.success_distance,
                              planner=planner, mover=mover, log=log,
                              debug=bool(args.debug))
                summary = hns.run(max_scans=args.max_scans)
                metrics = pull_metrics(env)
                metrics_payload = {
                    "agent_id": agent_id,
                    "gpu": gpu,
                    "index": i,
                    "key": key,
                    "scene": scene,
                    "episode_id": env.current_episode.episode_id,
                    "goal": goal,
                    "summary": summary,
                    "metrics": metrics,
                    "time_cost": summary.get("time_cost"),
                    "tool_counts": summary.get("tool_counts"),
                    "state": summary.get("state"),
                    "topdown_frames": summary.get("topdown_frames"),
                    "aborted": summary.get("aborted"),
                    "abort_reason": summary.get("abort_reason"),
                }
                if args.debug:
                    metrics_payload["topdown_mp4"] = os.path.join(ep_dir, "topdown.mp4")
                    metrics_payload["planner_txt"] = os.path.join(ep_dir, "planner.txt")
                dump_json(os.path.join(ep_dir, "metrics.json"), metrics_payload)
                log.emit("Harness", f"metrics {_short(metrics)}")
                print(f"=== {key} "
                      f"success={metrics.get('success')} "
                      f"spl={metrics.get('spl')} state={summary.get('state')} ===",
                      flush=True)
                ran += 1
            except Exception as exc:
                print(f"[ERROR] {key}: {exc}", flush=True)
                traceback.print_exc()
            progress_queue.put(1)
    except Exception:
        traceback.print_exc()
    finally:
        if backend is not None:
            backend.close()
        if env is not None:
            env.close()
        print(f"agent={agent_id} finished skipped={skipped} ran={ran}",
              flush=True)
        progress_queue.put("done")
        log_fp.close()


def collect_summary(run_dir, tasks, args, yaml_success, total_s=None):
    """汇总已完成 episode 的 metrics.json。"""
    import json
    from harness.protocol import dump_json, jsonable, write_brief_summary

    records = []
    for t in tasks:
        path = os.path.join(run_dir, ep_dir_name(t["key"]), "metrics.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        m = rec.get("metrics") or {}
        state = rec.get("state")
        if not state:
            summary = rec.get("summary") or {}
            if isinstance(summary, dict):
                state = summary.get("state")
        row = {
            "ep": t["index"],
            "scene": rec.get("scene"),
            "episode_id": rec.get("episode_id"),
            "goal": rec.get("goal"),
            "agent_id": rec.get("agent_id"),
            "gpu": rec.get("gpu"),
            "dir": os.path.join(run_dir, ep_dir_name(t["key"])),
            "key": t["key"],
            "time_cost": rec.get("time_cost"),
            "tool_counts": rec.get("tool_counts"),
            "state": state,
            "aborted": rec.get("aborted"),
        }
        for k in METRIC_KEYS:
            if k in m:
                row[k] = m[k]
        if "collisions" in m:
            row["collisions"] = m["collisions"]
        records.append(row)
    means = {}
    for k in METRIC_KEYS:
        vals = [float(r[k]) for r in records if k in r and r[k] is not None]
        means[k] = sum(vals) / max(len(vals), 1) if vals else None
    times = [float(r["time_cost"]) for r in records
             if r.get("time_cost") is not None]
    mean_ep = sum(times) / len(times) if times else None
    payload = {
        "n": len(records),
        "n_assigned": len(tasks),
        "means": means,
        "yaml_success": yaml_success,
        "success_distance": args.success_distance,
        "backend": None if args.no_glee else args.backend,
        "gpu": args.gpu,
        "seg_gpu": getattr(args, "seg_gpu", ""),
        "seg_num": 0 if args.no_glee else int(args.seg_num),
        "agent_num": args.agent_num,
        "episodes": records,
    }
    if total_s is not None:
        payload["total_time_s"] = round(float(total_s), 1)
        payload["mean_episode_time_s"] = (
            None if mean_ep is None else round(mean_ep, 1))
        write_brief_summary(
            run_dir, records, means, total_s, n_assigned=len(tasks))
    dump_json(os.path.join(run_dir, "summary.json"), jsonable(payload))
    return records, means


def main(argv=None):
    """列出 val episode、按 GPU/Agent 均分后并行跑。"""
    args = parse_args(argv)
    t_run = time.perf_counter()
    gpus = parse_gpus(args.gpu)
    if args.agent_num < 1:
        raise SystemExit("--agent_num 至少为 1")
    gpu_map = assign_gpus(args.agent_num, gpus)
    seg_gpus = parse_gpus(args.seg_gpu) if str(args.seg_gpu).strip() else list(gpus)
    if args.no_glee:
        seg_num = 0
        seg_gpu_map = []
    else:
        if int(args.seg_num) < 1:
            raise SystemExit("--seg-num 至少为 1")
        seg_num = int(args.seg_num)
        seg_gpu_map = assign_gpus(seg_num, seg_gpus)

    run_id = (args.run_id or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    print(f"准备 {args.stage} 共 {args.episodes} 集 (seed={args.seed}) ...", flush=True)
    all_eps, yaml_success, n_iter, epi_seed = get_episode_names(args, gpus[0])
    if not all_eps:
        raise SystemExit("没有可跑的 episode")

    buckets = split_round_robin(all_eps, args.agent_num)
    from harness.protocol import dump_json, jsonable
    dump_json(os.path.join(run_dir, "assignments.json"), jsonable({
        "run_id": run_id,
        "gpu": gpus,
        "seg_gpu": seg_gpus,
        "seg_num": seg_num,
        "agent_num": args.agent_num,
        "backend": None if args.no_glee else args.backend,
        "base_url": args.base_url,
        "debug": bool(args.debug),
        "seed": int(args.seed),
        "episode_sample_seed": int(epi_seed),
        "episodes": len(all_eps),
        "num_episode_sample": n_iter,
        "agents": [
            {"agent_id": i, "gpu": gpu_map[i],
             "n": len(buckets[i]),
             "keys": [t["key"] for t in buckets[i]]}
            for i in range(args.agent_num)
        ],
        "seg_workers": [
            {"seg_id": i, "gpu": seg_gpu_map[i]}
            for i in range(seg_num)
        ],
    }))

    print(f"run_dir={run_dir}", flush=True)
    print(
        f"GPU {gpus} x {args.agent_num} agents, "
        f"seg GPU {seg_gpus} x {seg_num} models, "
        f"{len(all_eps)} episodes num_episode_sample={n_iter} "
        f"backend={None if args.no_glee else args.backend} "
        f"vllm={args.base_url} debug={bool(args.debug)}",
        flush=True,
    )
    for i in range(args.agent_num):
        print(f"  agent {i:02d} gpu={gpu_map[i]} n={len(buckets[i])}", flush=True)
    for i in range(seg_num):
        print(f"  seg {i:02d} gpu={seg_gpu_map[i]}", flush=True)

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    req_q = None
    reply_qs = [None] * args.agent_num
    seg_procs = []
    if seg_num > 0:
        from perception.queue_backend import start_seg_workers, stop_seg_workers
        req_q = ctx.Queue()
        reply_qs = [ctx.Queue() for _ in range(args.agent_num)]
        print("启动分割 worker ...", flush=True)
        seg_procs = start_seg_workers(
            ctx, seg_gpu_map, args.backend,
            os.path.join(run_dir, "logs"), req_q, reply_qs,
            stagger_s=args.stagger_s)
        print(f"分割 worker ready n={seg_num}", flush=True)

    processes = []
    try:
        for i in range(args.agent_num):
            p = ctx.Process(
                target=worker_process,
                args=(i, gpu_map[i], buckets[i], args, run_dir, progress_queue,
                      n_iter, epi_seed, req_q, reply_qs[i]),
                daemon=False,
            )
            p.start()
            processes.append(p)

        finished_agents = 0
        done_eps = 0
        total = len(all_eps)
        while finished_agents < args.agent_num:
            try:
                msg = progress_queue.get(timeout=2.0)
            except Exception:
                if not any(p.is_alive() for p in processes):
                    break
                continue
            if msg == "done":
                finished_agents += 1
                print(f"agents finished {finished_agents}/{args.agent_num}",
                      flush=True)
            else:
                done_eps += int(msg)
                print(f"progress {done_eps}/{total}", flush=True)

        for p in processes:
            p.join()
    finally:
        if seg_procs:
            stop_seg_workers(seg_procs, req_q)

    records, means = collect_summary(
        run_dir, all_eps, args, yaml_success,
        total_s=time.perf_counter() - t_run)
    print("=" * 60, flush=True)
    print(f"run {run_dir}  n={len(records)}/{len(all_eps)}", flush=True)
    for k, v in means.items():
        if v is not None:
            print(f"  mean {k}: {v:.4f}", flush=True)
    print(f"  success_rate {means.get('success')}", flush=True)
    times = [float(r["time_cost"]) for r in records
             if r.get("time_cost") is not None]
    mean_ep = sum(times) / len(times) if times else None
    print(f"  total_time_s {time.perf_counter() - t_run:.1f}  "
          f"mean_episode_time_s {mean_ep}", flush=True)
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
