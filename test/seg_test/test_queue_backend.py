#!/usr/bin/env python
"""分割队列协议：DummyBackend 多客户端并发入队。"""

import os
import sys
import tempfile
import threading

import multiprocessing as mp
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from perception.queue_backend import QueueSegBackend, start_seg_workers, stop_seg_workers


def _client_work(agent_id, req_q, reply_q, n_req, out):
    """一个客户端连发若干张图。"""
    backend = QueueSegBackend(
        req_q, reply_q, agent_id, name="dummy", timeout_s=30.0)
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    rgb[0, 0] = agent_id
    labels = []
    for i in range(n_req):
        text = f"a{agent_id}_{i}"
        result = backend.segment(rgb, text, score_threshold=0.15)
        if len(result) != 1 or result.labels[0] != text:
            raise AssertionError(f"agent={agent_id} 回包不对 {result.labels}")
        labels.append(result.labels[0])
    out[agent_id] = labels


def main():
    """2 个 dummy worker、4 个客户端各 8 次请求。"""
    ctx = mp.get_context("spawn")
    n_seg = 2
    n_agent = 4
    n_req = 8
    req_q = ctx.Queue()
    reply_qs = [ctx.Queue() for _ in range(n_agent)]
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = os.path.join(tmp, "logs")
        procs = start_seg_workers(
            ctx, [0] * n_seg, "dummy", log_dir, req_q, reply_qs,
            stagger_s=0.0, ready_timeout_s=30.0)
        try:
            out = {}
            threads = []
            for i in range(n_agent):
                t = threading.Thread(
                    target=_client_work,
                    args=(i, req_q, reply_qs[i], n_req, out),
                )
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            if len(out) != n_agent:
                raise SystemExit(f"客户端未全部完成: {sorted(out)}")
            expected = n_agent * n_req
            got = sum(len(v) for v in out.values())
            if got != expected:
                raise SystemExit(f"请求数 {got} != {expected}")
            hits = {i: 0 for i in range(n_seg)}
            for i in range(n_seg):
                path = os.path.join(log_dir, f"seg_{i:02d}.log")
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                hits[i] = text.count(" agent=")
            if any(v == 0 for v in hits.values()):
                raise SystemExit(f"有 worker 没吃到任务: {hits}")
            print(f"ok clients={n_agent} reqs={got} worker_hits={hits}",
                  flush=True)
        finally:
            stop_seg_workers(procs, req_q)
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
