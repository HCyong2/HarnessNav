"""多进程分割队列：若干 GPU worker 共享请求队列，导航进程阻塞等结果。"""

import os
import sys
import time
import traceback

import numpy as np

from perception.base import SegResult

SEG_READY_TIMEOUT_S = 600.0
SEG_REPLY_TIMEOUT_S = 300.0
SEG_JOIN_TIMEOUT_S = 60.0


class DummySegBackend:
    """协议测试用：不占 GPU，把提示词写进 labels。"""

    name = "dummy"

    def segment(self, image_rgb, text, score_threshold=None):
        """返回单实例占位结果。"""
        h, w = image_rgb.shape[:2]
        time.sleep(0.02)
        mask = np.zeros((1, h, w), dtype=bool)
        mask[0, 0, 0] = True
        score = 0.9 if score_threshold is None else float(score_threshold)
        return SegResult(
            boxes=np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
            masks=mask,
            scores=np.array([score], dtype=np.float32),
            labels=[str(text)],
            time_ms=20.0,
        )

    def close(self):
        """无资源。"""
        return None


class QueueSegBackend:
    """把 ``segment`` 发到共享队列，在本 Agent 的回包队列上等待。"""

    def __init__(self, req_q, reply_q, agent_id, name="glee",
                 timeout_s=SEG_REPLY_TIMEOUT_S):
        """初始化客户端。

        Args:
            req_q: 全体 worker 共享的请求队列。
            reply_q: 本 Agent 私有回包队列。
            agent_id (int): 导航进程编号。
            name (str): 对外显示的后端名。
            timeout_s (float): 等回包超时秒数。
        """
        self.req_q = req_q
        self.reply_q = reply_q
        self.agent_id = int(agent_id)
        self.name = str(name)
        self.timeout_s = float(timeout_s)
        self._next_id = 0

    def segment(self, image_rgb, text, score_threshold=None):
        """入队并阻塞直到对应回包。"""
        self._next_id += 1
        rid = self._next_id
        rgb = np.ascontiguousarray(image_rgb)
        self.req_q.put((self.agent_id, rid, rgb, str(text), score_threshold))
        try:
            got_id, status, payload = self.reply_q.get(timeout=self.timeout_s)
        except Exception as exc:
            raise RuntimeError(
                f"分割队列超时 agent={self.agent_id} req={rid}"
            ) from exc
        if int(got_id) != int(rid):
            raise RuntimeError(
                f"分割回包序号错乱 agent={self.agent_id} "
                f"expect={rid} got={got_id}"
            )
        if status != "ok":
            raise RuntimeError(
                f"分割 worker 失败 agent={self.agent_id} req={rid}:\n{payload}"
            )
        return payload

    def close(self):
        """客户端不持有模型。"""
        return None


def seg_worker_process(worker_id, gpu, backend_name, req_q, reply_qs,
                       ready_ev, log_path, stagger_s=0.0):
    """单个分割 worker：加载一份模型，循环取请求。"""
    if backend_name != "dummy":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = log_fp
    sys.stderr = log_fp

    if stagger_s > 0:
        time.sleep(float(stagger_s) * int(worker_id))

    print(f"seg_worker={worker_id} physical_gpu={gpu} backend={backend_name}",
          flush=True)
    backend = None
    try:
        if backend_name == "dummy":
            backend = DummySegBackend()
        else:
            from run_HarnessNav import build_backend
            backend = build_backend(backend_name, "cuda:0")
            if backend is None:
                raise RuntimeError(f"{backend_name} 未加载")
        print(f"seg_worker={worker_id} ready", flush=True)
        ready_ev.set()
        while True:
            item = req_q.get()
            if item is None:
                print(f"seg_worker={worker_id} poison, exit", flush=True)
                break
            agent_id, rid, rgb, text, thr = item
            t0 = time.time()
            try:
                result = backend.segment(rgb, text, score_threshold=thr)
                reply_qs[int(agent_id)].put((rid, "ok", result))
                dt = (time.time() - t0) * 1000.0
                print(
                    f"seg_worker={worker_id} agent={agent_id} req={rid} "
                    f"text={text!r} n={len(result)} {dt:.0f}ms",
                    flush=True,
                )
            except Exception:
                err = traceback.format_exc()
                print(err, flush=True)
                reply_qs[int(agent_id)].put((rid, "err", err))
    except Exception:
        traceback.print_exc()
        if not ready_ev.is_set():
            ready_ev.set()
    finally:
        if backend is not None:
            backend.close()
        log_fp.close()


def start_seg_workers(ctx, gpu_ids, backend_name, log_dir, req_q, reply_qs,
                      stagger_s=0.0, ready_timeout_s=SEG_READY_TIMEOUT_S):
    """拉起分割 worker，等到全部 ready。

    Args:
        ctx: ``multiprocessing`` 上下文。
        gpu_ids (list): 长度等于 worker 数，每项物理 GPU。
        backend_name (str): ``glee`` / ``gdino_sam`` / ``dummy``。
        log_dir (str): 日志目录。
        req_q: 共享请求队列。
        reply_qs (list): 每个 Agent 一个回包队列。
        stagger_s (float): 相邻 worker 加载间隔。
        ready_timeout_s (float): 等全部 ready 的超时。

    Returns:
        list: ``Process`` 列表。
    """
    os.makedirs(log_dir, exist_ok=True)
    processes = []
    events = []
    for i, gpu in enumerate(gpu_ids):
        ev = ctx.Event()
        log_path = os.path.join(log_dir, f"seg_{i:02d}.log")
        p = ctx.Process(
            target=seg_worker_process,
            args=(i, gpu, backend_name, req_q, reply_qs, ev, log_path,
                  stagger_s),
            daemon=False,
        )
        p.start()
        processes.append(p)
        events.append(ev)
    deadline = time.time() + float(ready_timeout_s)
    for i, ev in enumerate(events):
        remain = max(deadline - time.time(), 0.1)
        if not ev.wait(timeout=remain) or not processes[i].is_alive():
            stop_seg_workers(processes, req_q)
            raise RuntimeError(f"分割 worker {i} 未就绪")
    return processes


def stop_seg_workers(processes, req_q):
    """投放毒丸并等待 worker 退出。"""
    n = len(processes)
    for _ in range(n):
        try:
            req_q.put(None)
        except Exception:
            break
    for p in processes:
        p.join(timeout=SEG_JOIN_TIMEOUT_S)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)
