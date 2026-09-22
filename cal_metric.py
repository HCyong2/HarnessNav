#!/usr/bin/env python
"""从实验目录写出 ``brief_summary.txt``（未跑完也可）。"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness.protocol import write_brief_summary

METRIC_KEYS = ("distance_to_goal", "success", "spl", "soft_spl")
_SKIP_DIRS = frozenset(("logs",))


def _load_json(path):
    """读 UTF-8 JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _means_from_records(records):
    """按已完成集计算指标均值。"""
    means = {}
    for key in METRIC_KEYS:
        vals = [float(row[key]) for row in records
                if key in row and row[key] is not None]
        means[key] = sum(vals) / len(vals) if vals else None
    return means


def _state_from_rec(rec):
    """从 ``metrics.json`` 取出终态。"""
    state = rec.get("state")
    if state:
        return state
    summary = rec.get("summary")
    if isinstance(summary, dict) and summary.get("state"):
        return summary["state"]
    return None


def _row_from_metrics(run_dir, rec, dirname):
    """把一集 ``metrics.json`` 收成摘要行。"""
    metrics = rec.get("metrics") or {}
    row = {
        "ep": rec.get("index"),
        "scene": rec.get("scene"),
        "episode_id": rec.get("episode_id"),
        "goal": rec.get("goal"),
        "agent_id": rec.get("agent_id"),
        "gpu": rec.get("gpu"),
        "dir": os.path.join(run_dir, dirname),
        "key": rec.get("key") or dirname,
        "time_cost": rec.get("time_cost"),
        "tool_counts": rec.get("tool_counts"),
        "state": _state_from_rec(rec),
        "aborted": rec.get("aborted"),
    }
    for key in METRIC_KEYS:
        if key in metrics:
            row[key] = metrics[key]
        elif key in rec:
            row[key] = rec[key]
    if "collisions" in metrics:
        row["collisions"] = metrics["collisions"]
    return row


def records_from_episode_dirs(run_dir):
    """扫描各集目录里已写出的 ``metrics.json``。"""
    records = []
    try:
        names = os.listdir(run_dir)
    except OSError:
        return records
    for name in sorted(names):
        if name in _SKIP_DIRS:
            continue
        path = os.path.join(run_dir, name, "metrics.json")
        if not os.path.isfile(path):
            continue
        rec = _load_json(path)
        if not isinstance(rec, dict):
            continue
        records.append(_row_from_metrics(run_dir, rec, name))
    return records


def n_assigned_from_assignments(run_dir):
    """从 ``assignments.json`` 读计划集数。"""
    path = os.path.join(run_dir, "assignments.json")
    if not os.path.isfile(path):
        return None
    data = _load_json(path)
    if not isinstance(data, dict):
        return None
    n = data.get("episodes")
    if n is not None:
        return int(n)
    agents = data.get("agents") or []
    total = sum(len(a.get("keys") or []) for a in agents if isinstance(a, dict))
    return total if total else None


def estimate_total_s(records):
    """用各 Agent 已完成集耗时之和的最大值估计墙钟秒数。"""
    by_agent = {}
    fallback = []
    for row in records:
        raw = row.get("time_cost")
        if raw is None:
            continue
        try:
            cost = float(raw)
        except (TypeError, ValueError):
            continue
        fallback.append(cost)
        aid = row.get("agent_id")
        key = 0 if aid is None else aid
        by_agent[key] = by_agent.get(key, 0.0) + cost
    if by_agent:
        return max(by_agent.values())
    if fallback:
        return max(fallback)
    return 0.0


def _enrich_state(run_dir, records):
    """补全 ``summary.json`` 行里缺失的终态（读各集 ``metrics.json``）。"""
    for row in records:
        if row.get("state"):
            continue
        candidates = []
        key = row.get("key")
        if key:
            candidates.append(os.path.join(run_dir, str(key), "metrics.json"))
        ep_dir = row.get("dir")
        if ep_dir:
            candidates.append(os.path.join(ep_dir, "metrics.json"))
        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                rec = _load_json(path)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(rec, dict):
                continue
            state = _state_from_rec(rec)
            if state:
                row["state"] = state
            if not row.get("tool_counts") and rec.get("tool_counts"):
                row["tool_counts"] = rec["tool_counts"]
            if row.get("state"):
                break


def collect(run_dir):
    """从 ``summary.json`` 或各集 ``metrics.json`` 取出记录与均值。

    Args:
        run_dir (str): 实验目录。

    Returns:
        tuple: ``(records, means, total_s, n_assigned)``。
    """
    summary_path = os.path.join(run_dir, "summary.json")
    records = []
    means = None
    total_s = None
    n_assigned = None
    if os.path.isfile(summary_path):
        data = _load_json(summary_path)
        if isinstance(data, dict):
            episodes = data.get("episodes")
            if isinstance(episodes, list):
                records = [row for row in episodes if isinstance(row, dict)]
            raw_means = data.get("means")
            if isinstance(raw_means, dict) and raw_means:
                means = raw_means
            if data.get("n_assigned") is not None:
                n_assigned = int(data["n_assigned"])
            if data.get("total_time_s") is not None:
                total_s = float(data["total_time_s"])
    if not records:
        records = records_from_episode_dirs(run_dir)
    else:
        _enrich_state(run_dir, records)
    if means is None:
        means = _means_from_records(records)
    if n_assigned is None:
        n_assigned = n_assigned_from_assignments(run_dir)
    if total_s is None:
        total_s = estimate_total_s(records)
    return records, means, total_s, n_assigned


def parse_args(argv=None):
    """解析命令行。"""
    p = argparse.ArgumentParser(
        description="根据 summary.json 或各集 metrics.json 写出 brief_summary.txt")
    p.add_argument(
        "run_dir",
        help="实验目录，例如 /DATA_HDD/hc/harness_outputs/p1/20260922_112840/")
    return p.parse_args(argv)


def main(argv=None):
    """写出 ``brief_summary.txt`` 并打印内容。"""
    args = parse_args(argv)
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    if not os.path.isdir(run_dir):
        raise SystemExit(f"目录不存在: {run_dir}")
    records, means, total_s, n_assigned = collect(run_dir)
    if not records:
        raise SystemExit(f"未找到 summary.json 或已完成的 metrics.json: {run_dir}")
    path = write_brief_summary(
        run_dir, records, means, total_s, n_assigned=n_assigned)
    text = open(path, encoding="utf-8").read()
    sys.stdout.write(text)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
