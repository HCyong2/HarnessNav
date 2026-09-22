#!/usr/bin/env python
"""网页层端到端自检：不起浏览器，用 HTTP 打完整交互链路。

    python play2nav/test_webserver.py
    python play2nav/test_webserver.py --keep
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request


# 换场景期间探针的取样窗口与间隔。窗口要盖住 env.reset()（3~8 秒）并留出富余。
# 这几个常量必须定义在下面那行分派**之前**（分派在模块导入早期就执行了）。
PROBE_SECONDS = 12.0
PROBE_INTERVAL = 0.25
PROBE_TIMEOUT = 3.0


def _poll_probe(argv):
    """独立进程里轮询 ``/state``，把每次取样写成 JSON 打到 stdout。"""
    parser = argparse.ArgumentParser(prog="test_webserver.py --poll-probe")
    parser.add_argument("--poll-probe", action="store_true")
    parser.add_argument("base")
    parser.add_argument("seconds", type=float)
    parser.add_argument("interval", type=float)
    options = parser.parse_args(argv)

    samples = []
    deadline = time.time() + options.seconds
    while time.time() < deadline:
        started = time.monotonic()
        phase, error = None, None
        try:
            with urllib.request.urlopen(options.base + "/state", timeout=PROBE_TIMEOUT) as response:
                phase = json.loads(response.read().decode("utf-8")).get("phase")
        except Exception as exc:
            error = type(exc).__name__
        samples.append({"t": time.monotonic(), "phase": phase,
                        "cost": time.monotonic() - started, "error": error})
        time.sleep(options.interval)
    print(json.dumps(samples), flush=True)
    return 0


if "--poll-probe" in sys.argv:
    raise SystemExit(_poll_probe(sys.argv[1:]))

import cv2
import habitat
import imageio.v2 as imageio
import numpy as np                                                          # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("HABITAT_LAB_LOG", "40")

from play2nav.app import Navigator, _TopdownGate, _web_loop, parse_args  # noqa: E402
from play2nav.config import check_split, play_config, summary_lines      # noqa: E402
from play2nav.web import KEY_DEADMAN_S, Play2NavServer, _FrameSlot       # noqa: E402

DEFAULT_OUT = "/DATA_HDD/hc/play2nav_outputs/webcheck"
FRAME0_PATH = "/tmp/play2nav_webtest_frame0.jpg"

# 长按验收的时长与容差。按下当刻先出**一步**（轻触），按住满 --hold-delay 之后每
# --action-interval（0.15 s）一步。所以总按住时长 = 门槛 + 0.9 秒的移动窗口。
HOLD_DELAY = 1.0              # 与下面 parse_args 里显式传的 --hold-delay 一致
HOLD_MOVE_WINDOW_S = 0.9      # 越过门槛之后观察这么久（0.15 s/步 → 约 6 步）
HOLD_SAMPLE_INTERVAL = 0.3
HOLD_SAMPLES = 3
EXPECTED_MIN_STEPS = 5        # 下界：轻触 1 步 + 连续至少 4 步，排除「只走出轻触那一步」
EXPECTED_MAX_STEPS = 12       # 上界：排除「不按时间闸门、每 tick 都发一个动作」

# 发完控制请求后、读 /state 前要静置多久。**比一个动作间隔（0.15 s）长**：松手的一瞬间
# 可能正好有一个动作已经在 sim 线程里发出去了，它落地会把步数再抬 1。这不是卡键
# （卡键是每秒约 6 步），但不静置就分不开这两者。
SETTLE_S = 0.5

SLOW_S = 0.5                  # 环回上打一个 /state 超过这么久就算"卡住了"

# 死人开关的观察窗口：要盖住 KEY_DEADMAN_S 再留出富余（开关生效后还要看出它真的停住）。
DEADMAN_WAIT_S = 9.0

_RESULTS = []
_FAILED = []


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok), detail))
    if not ok:
        _FAILED.append(name)
    mark = "OK  " if ok else "FAIL"
    print(f"  {mark} {name}" + (f"  —— {detail}" if detail else ""), flush=True)
    return bool(ok)


def check_stall_reporting(hints, deltas, where):
    """按最后一步位移判定是否卡住，并检查提示与位移是否一致。

    Returns:
        bool: 末次取样判定为卡住。
    """
    pairs = [(d, h) for d, h in zip(deltas, hints) if isinstance(d, (int, float))]
    numbers = [d for d, _h in pairs]
    last = numbers[-1] if numbers else None
    stalled = last is not None and last <= 1e-4

    false_pos = [(d, h) for d, h in pairs if d > 1e-4 and h is not None]
    check(f"{where}：每有一次位移就不该报「碰到障碍物」（{len(pairs)} 次取样）",
          not false_pos, f"误报 {false_pos[:2]}")

    if stalled:
        check(f"{where}：上一步位置没变时，页面给出了「碰到障碍物」提示",
              bool(hints) and hints[-1] is not None and "碰到障碍物" in (hints[-1] or ""),
              f"末次位移 {last}，提示 {hints[-1] if hints else None}")
    else:
        check(f"{where}：上一步确实走动了时，不该误报「碰到障碍物」",
              bool(hints) and hints[-1] is None,
              f"末次位移 {last}（本段最大 {max(numbers) if numbers else None}）")
    return stalled


# --------------------------------------------------------------------------- #
# HTTP 小工具（只用标准库，环境里没有 requests 也能跑）
# --------------------------------------------------------------------------- #

def http_get(base, path, timeout=5.0):
    return urllib.request.urlopen(base + path, timeout=timeout)


def http_post(base, path, payload=None, timeout=5.0):
    body = json.dumps(payload).encode() if payload is not None else b""
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(base + path, data=body, method="POST", headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def get_json(base, path, timeout=5.0):
    with http_get(base, path, timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _dechunk(raw):
    """剥掉 ``Transfer-Encoding: chunked`` 的帧界，返回消息体。"""
    body = bytearray()
    index = 0
    while index < len(raw):
        end = raw.find(b"\r\n", index)
        if end < 0:
            break
        try:
            length = int(raw[index:end].split(b";")[0], 16)
        except ValueError:
            break                        # 不是长度行：后面已经不对齐了，停在这里
        index = end + 2
        body.extend(raw[index:index + length])
        index += length + 2
    return bytes(body)


def read_stream(base, seconds, socket_timeout=0.4):
    """用裸 socket 读 MJPEG 流 ``seconds`` 秒，返回已剥 chunked 的消息体。"""
    deadline = time.time() + seconds
    parts = urllib.parse.urlsplit(base)
    host, port = parts.hostname, parts.port
    request = (f"GET {parts.path or '/'}/stream HTTP/1.1\r\n"
               f"Host: {host}:{port}\r\n"
               "Accept: multipart/x-mixed-replace\r\n"
               "Connection: close\r\n\r\n")
    raw = bytearray()
    try:
        sock = socket.create_connection((host, port), timeout=socket_timeout)
    except OSError:
        return b""
    try:
        sock.settimeout(socket_timeout)
        sock.sendall(request.encode())
        while time.time() < deadline:
            try:
                chunk = sock.recv(65536)
            except (socket.timeout, TimeoutError):
                continue
            if not chunk:
                break
            raw.extend(chunk)
    except OSError:
        pass
    finally:
        try:
            sock.close()
        except OSError:
            pass
    _, _, body = bytes(raw).partition(b"\r\n\r\n")     # 空则 partition 给空 body
    return _dechunk(body)


def count_frames(body):
    return body.count(b"--frame")


def frames_in(body):
    """把 MJPEG 消息体切成每段 part 的 JPEG 载荷。"""
    payloads = []
    for part in body.split(b"--frame")[1:]:
        _, _, data = part.partition(b"\r\n\r\n")
        payloads.append(data.rstrip(b"\r\n"))
    return payloads


class _GapWatch:
    """心跳线程：记录 ``sleep`` 被拉长的最长时间，用来判断解释器是否被冻住。"""

    def __init__(self, interval=0.05):
        self._interval = interval
        self._stop = threading.Event()
        self.max_gap = 0.0

    def _run(self):
        last = time.monotonic()
        while not self._stop.is_set():
            self._stop.wait(self._interval)
            now = time.monotonic()
            self.max_gap = max(self.max_gap, now - last)
            last = now

    def start(self):
        threading.Thread(target=self._run, name="gil-gap-watch", daemon=True).start()
        return self

    def stop(self):
        self._stop.set()


def _timeline(phases):
    """把相序列压成 ``finished x3 -> loading x2 -> 超时 x1 -> navigating x12``。"""
    parts = []
    for phase in phases:
        label = phase or "超时"
        if parts and parts[-1][0] == label:
            parts[-1][1] += 1
        else:
            parts.append([label, 1])
    return " -> ".join(f"{label} x{count}" for label, count in parts)


def wait_for_phase(base, timeout=90.0, desired=("navigating", "finished", "no_navmesh")):
    """等换集结束（``env.reset()`` 要 3~8 秒，首次还要加载 glb + navmesh）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            snapshot = get_json(base, "/state", timeout=3)
        except Exception:
            time.sleep(0.2)
            continue
        if snapshot.get("phase") in desired:
            return snapshot
        time.sleep(0.2)
    raise TimeoutError(f"等了 {timeout:.0f}s 仍是 {snapshot.get('phase')!r}")


# --------------------------------------------------------------------------- #
# 客户端：在另一个线程里打 HTTP
# --------------------------------------------------------------------------- #

def client_checks(base, episode_dirs, errors):
    """全部断言都在这里。最后无论如何都 POST /quit，好让主线程的 sim 循环退出。"""
    try:
        run_checks(base, episode_dirs)
    except Exception:
        errors.append(traceback.format_exc())
    finally:
        try:
            http_post(base, "/quit", timeout=3)
        except Exception:
            pass


def run_checks(base, episode_dirs):
    print(f"\n--- 客户端开始打 {base} ---", flush=True)

    # 1) 页面
    with http_get(base, "/") as response:
        html = response.read().decode("utf-8")
        content_type = response.headers.get("Content-Type", "")
        cache = response.headers.get("Cache-Control", "")
    check("GET / 返回 HTML 且带 utf-8 charset", "text/html" in content_type
          and "charset=utf-8" in content_type, content_type)
    check("页面是 play2nav 的界面", "进入下一个场景" in html and "/frame.jpg" in html)
    check("页面禁缓存（改了 page.html 就能立刻生效）", "no-store" in cache, cache)
    check("页面无输入控件（不能有 <input>，否则按键会被抢走）", "<input" not in html.lower())

    # 2) MJPEG 响应头
    with http_get(base, "/stream") as response:
        headers = response.headers
        check("流是 multipart/x-mixed-replace + boundary=frame",
              "multipart/x-mixed-replace" in headers.get("Content-Type", "")
              and "boundary=frame" in headers.get("Content-Type", ""),
              headers.get("Content-Type", ""))
        check("流是 chunked（不预置 Content-Length）",
              headers.get("Transfer-Encoding", "").lower() == "chunked"
              and headers.get("Content-Length") is None,
              f"TE={headers.get('Transfer-Encoding')} CL={headers.get('Content-Length')}")

    with http_get(base, "/topdown_stream") as response:
        headers = response.headers
        check("俯视图走第二条 MJPEG 流（不再逐帧拉取）",
              "multipart/x-mixed-replace" in headers.get("Content-Type", "")
              and headers.get("Transfer-Encoding", "").lower() == "chunked",
              f"{headers.get('Content-Type')} / TE={headers.get('Transfer-Encoding')}")

    wait_for_phase(base)
    idle = read_stream(base, 3.0)
    idle_frames = frames_in(idle)
    check("空闲 3 秒内只推 1 帧，且同一帧发两段（未变化的画面不推送）",
          len(idle_frames) == 2 and idle_frames[0] == idle_frames[1] and idle_frames[0],
          f"{len(idle_frames)} 段 part, {len(idle)} 字节")
    check("首帧是合法 JPEG", idle.count(b"\xff\xd8\xff") >= 1)

    snapshot = get_json(base, "/state")
    check("刚加载完时 step_idx == 0", snapshot["step_idx"] == 0, f"step_idx={snapshot['step_idx']}")
    check("快照带有场景/目标/步数上限",
          snapshot["scene_id"] not in ("", "?") and snapshot["category"] not in ("", "n/a")
          and snapshot["max_steps"] > 0,
          f"{snapshot['scene_id']} ep{snapshot['episode_id']} "
          f"target={snapshot['category']} max={snapshot['max_steps']}")
    check("快照带回按住门槛（页面要按它显示倒计时）",
          abs(float(snapshot.get("hold_delay", -1)) - HOLD_DELAY) < 1e-6,
          f"hold_delay={snapshot.get('hold_delay')}")
    check("快照带回「输入检测」的实时行与事件流",
          isinstance(snapshot.get("input_live"), list) and snapshot["input_live"]
          and isinstance(snapshot.get("input_log"), list),
          f"input_live={snapshot.get('input_live')}")
    check("快照带回两条流的统计（页面上要看帧率与带宽）",
          isinstance(snapshot.get("stream"), dict)
          and snapshot["stream"].get("rgb_frames", 0) >= 1
          and snapshot["stream"].get("topdown_frames", 0) >= 1,
          str(snapshot.get("stream")))

    with http_get(base, "/frame.jpg") as response:
        frame0 = response.read()
    with open(FRAME0_PATH, "wb") as handle:
        handle.write(frame0)
    check("GET /frame.jpg 拿到 JPEG", frame0.startswith(b"\xff\xd8\xff") and len(frame0) > 1000,
          f"{len(frame0)} 字节")

    # 5) 长按 + 6) 并发（两个流读者 + 轮询同时跑）
    stream_results = []
    readers = []
    for index in range(2):
        thread = threading.Thread(
            target=lambda i=index: stream_results.append(read_stream(base, 3.0)),
            name=f"stream-reader-{index}", daemon=True)
        readers.append(thread)
    for thread in readers:
        thread.start()

    s0 = get_json(base, "/state")["step_idx"]
    pressed_at = time.monotonic()
    http_post(base, "/key", {"k": "w", "down": True, "seq": 1})
    deltas = []

    time.sleep(HOLD_DELAY - 0.25)
    during_delay = get_json(base, "/state")["step_idx"]
    check(f"轻触一下走出恰好一步（已按住 {HOLD_DELAY - 0.25:.2f}s，尚未到门槛）",
          during_delay == s0 + 1,           f"{s0} -> {during_delay}")

    samples, hints = [], []
    while time.monotonic() - pressed_at < HOLD_DELAY + HOLD_MOVE_WINDOW_S:
        time.sleep(HOLD_SAMPLE_INTERVAL)
        snapshot = get_json(base, "/state")
        samples.append(snapshot["step_idx"])
        deltas.append(snapshot.get("last_delta_m"))
        hints.append(snapshot.get("stuck_hint"))
    check("越过门槛后步数严格递增（长按真的在持续前进）",
          len(samples) >= 2 and all(b > a for a, b in zip(samples[:-1], samples[1:])),
          f"门槛后读数 {samples}")

    http_post(base, "/keys/clear")
    time.sleep(SETTLE_S)
    s1 = get_json(base, "/state")["step_idx"]
    time.sleep(1.0)
    s2 = get_json(base, "/state")["step_idx"]
    advanced = s2 - s0
    check("松手后不再前进（没卡键）", s2 == s1, f"静置后 {s1} -> 又过 1 秒 {s2}")
    check(f"按住 {HOLD_DELAY + HOLD_MOVE_WINDOW_S:.1f}s 共前进 "
          f"{EXPECTED_MIN_STEPS}~{EXPECTED_MAX_STEPS} 步（轻触 1 步 + 连续若干步）",
          EXPECTED_MIN_STEPS <= advanced <= EXPECTED_MAX_STEPS,
          f"实际 {advanced} 步（下界排除「只走出轻触那一步」，上界排除「每 tick 都发一个动作」）")

    moved = [d for d in deltas if isinstance(d, (int, float))]
    check_stall_reporting(hints, moved, "第一集")
    check("/state 带回每步位移（走不动的判据来源）",
          bool(moved) and max(moved) > 0.0, f"最大位移 {max(moved) if moved else None}")

    # 重复按下（模拟自动重复）不能多走路：同一个键再按 5 次，步数增量仍受时间闸门约束
    before_repeat = get_json(base, "/state")["step_idx"]
    for index in range(5):
        http_post(base, "/key", {"k": "w", "down": True, "seq": 100 + index})
    time.sleep(0.05)
    http_post(base, "/keys/clear")
    time.sleep(SETTLE_S)
    after_repeat = get_json(base, "/state")["step_idx"]
    check("连按 5 次同一键不会叠加出多步（时间闸门生效）", after_repeat - before_repeat <= 1,
          f"{before_repeat} -> {after_repeat}")

    before_tap = get_json(base, "/state")["step_idx"]
    http_post(base, "/key", {"k": "w", "down": True, "seq": 150})
    time.sleep(0.25)
    http_post(base, "/key", {"k": "w", "down": False, "seq": 151})
    time.sleep(SETTLE_S)
    after_tap = get_json(base, "/state")["step_idx"]
    check("轻点 w（0.25s）走且只走一步", after_tap - before_tap == 1,
          f"{before_tap} -> {after_tap}")
    snapshot = get_json(base, "/state")
    texts = [entry["text"] for entry in snapshot.get("input_log", [])]
    check("「输入检测」里有按键按下记录", any("↓ 按下" in t for t in texts),
          f"事件流 {texts[-3:]}")
    check("「输入检测」说清了轻点只出一步",
          any("只出轻触的一步" in t for t in texts), f"事件流 {texts[-2:]}")

    before_double = get_json(base, "/state")["step_idx"]
    for seq, down, wait in ((160, True, 0.10), (161, False, 0.15),
                            (162, True, 0.10), (163, False, 0.0)):
        http_post(base, "/key", {"k": "w", "down": down, "seq": seq})
        time.sleep(wait)
    time.sleep(SETTLE_S)
    after_double = get_json(base, "/state")["step_idx"]
    check("连点两下走出两步（第二下不会被时间闸门吃掉）",
          after_double - before_double == 2, f"{before_double} -> {after_double}")

    before_grace_tap = get_json(base, "/state")["step_idx"]
    http_post(base, "/key", {"k": "w", "down": True, "seq": 170})
    time.sleep(0.5)
    http_post(base, "/key", {"k": "w", "down": False, "seq": 171})
    http_post(base, "/key", {"k": "w", "down": True, "seq": 172})      # 距松手仅一个往返
    time.sleep(0.10)
    http_post(base, "/key", {"k": "w", "down": False, "seq": 173})
    time.sleep(SETTLE_S)
    after_grace_tap = get_json(base, "/state")["step_idx"]
    check("连点两下的第二下落在松手宽限期内也走出第二步",
          after_grace_tap - before_grace_tap == 2,
          f"{before_grace_tap} -> {after_grace_tap}")

    RAPID_TAPS = 5
    before_rapid = get_json(base, "/state")["step_idx"]
    for index in range(RAPID_TAPS):
        http_post(base, "/key", {"k": "w", "down": True, "seq": 180 + index * 2})
        time.sleep(0.03)                                   # 远小于一步
        http_post(base, "/key", {"k": "w", "down": False, "seq": 181 + index * 2})
        time.sleep(0.5)
    time.sleep(1.0)                                        # 最后那一步也要来得及出手
    after_rapid = get_json(base, "/state")["step_idx"]
    check(f"{RAPID_TAPS} 次 30ms 极短轻触 = {RAPID_TAPS} 步（一步都没被吞掉）",
          after_rapid - before_rapid == RAPID_TAPS,
          f"{before_rapid} -> {after_rapid}")

    settled = get_json(base, "/state")
    check("第一视角已推帧号 == 当前帧号（画面没落后）",
          settled.get("pub_rgb_seq") == settled.get("emit_seq"),
          f"{settled.get('pub_rgb_seq')}/{settled.get('emit_seq')}")
    check("俯视图已推帧号 == 当前帧号（限速只推迟、不吞帧）",
          settled.get("pub_topdown_seq") == settled.get("frame_seq"),
          f"{settled.get('pub_topdown_seq')}/{settled.get('frame_seq')}")

    with http_post(base, "/key", {"k": "w", "down": True, "seq": 1}) as response:
        verdict = json.loads(response.read().decode("utf-8"))
    check("过期 seq 的按键请求被明确拒绝", verdict.get("ok") is False
          and verdict.get("why") == "stale-seq", str(verdict))
    stale = get_json(base, "/state")["step_idx"]
    time.sleep(0.6)
    stale_after = get_json(base, "/state")["step_idx"]
    check("被拒绝的按键没有让 agent 动起来", stale_after == stale,
          f"{stale} -> 0.6 秒后 {stale_after}")
    http_post(base, "/keys/clear")
    time.sleep(SETTLE_S)

    http_post(base, "/key", {"k": "w", "down": True, "seq": 200})
    press_at = time.time()
    last, last_change = None, press_at
    while time.time() - press_at < DEADMAN_WAIT_S:
        time.sleep(0.25)
        current = get_json(base, "/state")["step_idx"]
        if last is not None and current != last:
            last_change = time.time()
        last = current
    quiet_for = time.time() - last_change
    check(f"死人开关：不发 /key 时按住的键在第 {KEY_DEADMAN_S:.0f} 秒后被松开",
          KEY_DEADMAN_S - 1.5 <= last_change - press_at <= KEY_DEADMAN_S + 2.0
          and quiet_for >= 1.5,
          f"最后一次动作在 +{last_change - press_at:.1f}s，之后 {quiet_for:.1f}s 没再动")
    http_post(base, "/keys/clear")
    time.sleep(SETTLE_S)

    for thread in readers:
        thread.join(timeout=6)
    check("两个并发流读者都拿到了帧（threaded=True 生效）",
          len(stream_results) == 2
          and all(len(frames_in(raw)) >= 2 for raw in stream_results),
          f"各收到 {[len(frames_in(raw)) for raw in stream_results]} 段 part")
    check("按住时有新帧流入（流是活的）",
          sum(len(frames_in(raw)) for raw in stream_results) >= 4,
          f"合计 {sum(len(frames_in(raw)) for raw in stream_results)} 段 part（每帧两段）")

    # 7) /stop：结算并落盘
    http_post(base, "/stop")
    time.sleep(1.0)
    snapshot = get_json(base, "/state")
    check("按 p 后 phase 变成 finished", snapshot["phase"] == "finished", snapshot["phase"])
    check("结束原因是 user_stop", snapshot["terminated_by"] == "user_stop",
          str(snapshot["terminated_by"]))
    episode_dir = snapshot["episode_dir"]
    episode_dirs.append(episode_dir)

    metrics = json.load(open(os.path.join(episode_dir, "metrics.json"), encoding="utf-8"))
    check("metrics.json 里的 num_steps 与网页上的 step_idx 一致（不是两份会漂移的真相）",
          metrics["num_steps"] == snapshot["step_idx"],
          f"metrics={metrics['num_steps']} /state={snapshot['step_idx']}")
    check("确实发出了多次 move_forward（磁盘上的动作序列也是连续的）",
          metrics["action_counts"].get("move_forward", 0) >= EXPECTED_MIN_STEPS,
          f"move_forward x{metrics['action_counts'].get('move_forward', 0)}")
    check("num_steps == len(actions)", metrics["num_steps"] == len(metrics["actions"]))
    check("video_frames == num_steps + 1", metrics["video_frames"] == metrics["num_steps"] + 1,
          f"frames={metrics['video_frames']} steps={metrics['num_steps']}")
    check("success 已结算（不是 None）", metrics.get("success") is not None,
          f"success={metrics.get('success')} distance={metrics.get('distance_to_goal')}")
    for name in ("rgb.mp4", "topdown.mp4", "trajectory.png", "metrics.json"):
        path = os.path.join(episode_dir, name)
        check(f"产物 {name} 存在且非空",
              os.path.exists(path) and os.path.getsize(path) > 1000,
              f"{os.path.getsize(path) if os.path.exists(path) else 0} 字节")

    print("  .. 正在换场景（env.reset() 会阻塞 3~8 秒，期间 HTTP 必须仍然应答）", flush=True)
    first_scene = snapshot["scene_id"]
    watch = _GapWatch().start()
    probe = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--poll-probe", base,
         str(PROBE_SECONDS), str(PROBE_INTERVAL)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    time.sleep(0.5)                    # 等探针真的开始打请求，再触发换场景
    poster = threading.Thread(target=lambda: http_post(base, "/next", timeout=60), daemon=True)
    poster.start()
    poster.join(timeout=60)
    samples = json.loads(probe.communicate(timeout=60)[0])
    watch.stop()

    costs = [sample["cost"] for sample in samples]
    phases = [sample["phase"] for sample in samples]
    slowest = max(costs) if costs else float("inf")
    loading_at = [index for index, phase in enumerate(phases) if phase == "loading"]
    loaded = samples[loading_at[-1] + 1:] if loading_at else []
    healthy = [sample for sample in loaded
               if not sample["error"] and sample["cost"] < SLOW_S]
    print(f"  [探针] {len(samples)} 次取样；相序列 {_timeline(phases)}；"
          f"卡顿 {len(samples) - len([s for s in samples if not s['error'] and s['cost'] < SLOW_S])} 次；"
          f"进程内心跳最长间隔 {watch.max_gap:.2f}s", flush=True)

    check("探针确实覆盖了换场景窗口（读到过 loading）", bool(loading_at),
          f"读到过的相 {sorted(set(p for p in phases if p))}")
    check("换场景结束后服务端立刻恢复（其后每次取样都成功且 < 0.5 秒）",
          len(loaded) >= 3 and len(healthy) == len(loaded),
          f"换场景之后 {len(loaded)} 次取样，其中 {len(loaded) - len(healthy)} 次失败/偏慢")
    check("卡顿是解释器被冻住（心跳线程同样被饿），不是 HTTP 层自己慢",
          slowest < SLOW_S or watch.max_gap >= slowest * 0.5,
          f"心跳最长间隔 {watch.max_gap:.2f}s vs 最慢一次 /state {slowest:.2f}s")

    second = wait_for_phase(base)
    episode_dirs.append(second["episode_dir"])
    check("换到了一个不同的场景（真的推进了一集，不是重复加载）",
          second["scene_id"] != first_scene,
          f"{first_scene} -> {second['scene_id']}（--shuffle 保证跨场景）")
    check("换集后步数归零、帧号已推进", second["step_idx"] == 0 and second["frame_seq"] > 0,
          f"step={second['step_idx']} frame_seq={second['frame_seq']}")
    with http_get(base, "/topdown.jpg") as response:
        topdown = response.read()
        check("GET /topdown.jpg 是新场景的合法 JPEG（网页上的实时小地图）",
              response.status == 200 and topdown.startswith(b"\xff\xd8\xff") and len(topdown) > 2000,
              f"{len(topdown)} 字节")

    http_post(base, "/key", {"k": "w", "down": True, "seq": 300})
    pressed_at = time.monotonic()
    deltas, hints = [], []
    while time.monotonic() - pressed_at < HOLD_DELAY + HOLD_MOVE_WINDOW_S:
        time.sleep(HOLD_SAMPLE_INTERVAL)
        snapshot = get_json(base, "/state")
        deltas.append(snapshot.get("last_delta_m"))
        hints.append(snapshot.get("stuck_hint"))
    http_post(base, "/keys/clear")
    time.sleep(SETTLE_S)

    moved = [d for d in deltas if isinstance(d, (int, float))]
    if check_stall_reporting(hints, moved, "第二集"):
        after = get_json(base, "/state")
        live = " | ".join(after.get("input_live") or [])
        check("该提示同时出现在「输入检测」栏的实时行里（要求 4）",
              "⚠" in live or "碰到障碍物" in live, live)
        check("该提示也进了「输入检测」的滚动事件流",
              any("碰到障碍物" in entry["text"] for entry in after.get("input_log", [])),
              str([entry["text"] for entry in after.get("input_log", [])][-2:]))

    # 9) /quit 由 client_checks 的 finally 负责
    print("  .. 检查完毕，发送 /quit", flush=True)


# --------------------------------------------------------------------------- #
# 主线程（= sim 线程）侧
# --------------------------------------------------------------------------- #

def verify_colour_order(episode_dirs):
    """对比 /frame.jpg 与同集 rgb.mp4 第 0 帧的颜色序。"""
    if not episode_dirs:
        check("颜色序：拿到过 episode 目录", False, "没有 episode")
        return
    stream = imageio.get_reader(os.path.join(episode_dirs[0], "rgb.mp4"))
    reference = np.asarray(stream.get_data(0))            # RGB
    stream.close()
    encoded = cv2.imread(FRAME0_PATH)                     # BGR
    if encoded is None:
        exists = os.path.exists(FRAME0_PATH)
        size = os.path.getsize(FRAME0_PATH) if exists else -1
        check("颜色序：能读回 /frame.jpg", False,
              f"{FRAME0_PATH} exists={exists} size={size}")
        return
    decoded = cv2.cvtColor(encoded, cv2.COLOR_BGR2RGB)
    raw_shape = reference.shape
    resized = decoded.shape != reference.shape
    if resized:
        reference = cv2.resize(reference, (decoded.shape[1], decoded.shape[0]),
                               interpolation=cv2.INTER_AREA)
    check("网页画面确实按 --display-scale 缩小了（录像仍是原始分辨率）",
          decoded.shape[1] <= raw_shape[1],
          f"网页 {decoded.shape[1]}x{decoded.shape[0]}，录像原始 {raw_shape[1]}x{raw_shape[0]}"
          f"{'（未缩小，说明 display-scale=1）' if not resized else ''}")
    diff = float(np.abs(decoded.astype(int) - reference.astype(int)).mean())
    check("网页画面的颜色序正确（RGB/BGR 没有互换）", diff < 8.0,
          f"与 rgb.mp4 第 0 帧平均差 {diff:.2f}/255（互换会是几十）")
    check("画面不是退化图（确实有内容）",
          float(decoded.reshape(-1, 3).std(axis=0).max()) > 10.0,
          f"通道标准差最大 {float(decoded.reshape(-1, 3).std(axis=0).max()):.1f}")


def check_topdown_gate():
    """``_TopdownGate``：限速窗口内的新帧被推迟、不被丢弃。"""
    gate = _TopdownGate(1.0)
    check("限速闸门：第一帧立刻发（还没发过任何一帧）", gate.due(100.0, 1))
    gate.mark(100.0, 1)
    check("限速闸门：帧号没变就不重复发", not gate.due(100.5, 1))
    check("限速闸门：窗口内的新帧被推迟、不被吞",
          not gate.due(100.5, 2) and gate.published_seq == 1, "等窗口")
    check("限速闸门：窗口一到就补发那一帧", gate.due(101.0, 2))
    gate.mark(101.0, 2)
    check("限速闸门：隔了很久之后的下一帧照发（不补历史、也不卡住）", gate.due(109.0, 7))
    check("限速闸门：间隔为 0 等于不限速", _TopdownGate(0.0).due(100.0, 1))


def check_mjpeg_double_part():
    """``_mjpeg`` 对同一帧连续 yield 两段相同的 part。"""
    server = Play2NavServer.__new__(Play2NavServer)
    server._stop_event = threading.Event()
    slot = _FrameSlot()
    generator = server._mjpeg(slot)
    jpeg = b"\xff\xd8jpeg-bytes\xff\xd9"
    slot.publish(jpeg)
    first = next(generator)
    second = next(generator)
    check("MJPEG：一帧发两段（第一段把帧推出去）",
          first.startswith(b"--frame\r\n") and jpeg in first, f"{first[:40]!r}")
    check("MJPEG：第二段与第一段逐字节相同（只把第一段顶(上屏)）",
          first == second, f"{len(first)} vs {len(second)} 字节")
    check("MJPEG：两段里都是同一帧、且带 Content-Length",
          second.count(jpeg) == 1 and b"Content-Length: %d" % len(jpeg) in second)
    server._stop_event.set()
    slot.stop()
    check("MJPEG：停机后生成器立刻收工", list(generator) == [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--keep", action="store_true", help="保留 scratch 产物（默认跑完删掉）")
    options = parser.parse_args()

    args = parse_args(["--stage", "val", "--episodes", "2", "--shuffle",
                       "--out", options.out, "--port", "0",
                       "--hold-delay", str(HOLD_DELAY), "--action-interval", "0.15"])
    args.host = "127.0.0.1"

    if os.path.exists(options.out) and not options.keep:
        shutil.rmtree(options.out)
    os.makedirs(options.out, exist_ok=True)

    print("=" * 78)
    print("play2nav 网页层自检（不需要浏览器）")
    print(f"  产物      : {options.out}")
    print(f"  按键判据  : 轻触走出 1 步；按住满 {HOLD_DELAY:.1f}s 转连续，"
          f"此后 {HOLD_MOVE_WINDOW_S:.1f}s 内共前进 {EXPECTED_MIN_STEPS}~{EXPECTED_MAX_STEPS} 步")
    print("=" * 78, flush=True)

    config, fixed, yaml_success = play_config(stage=args.stage, episodes=args.episodes,
                                              seed=args.seed, gpu_id=args.gpu_id,
                                              success_distance=args.success_distance,
                                              shuffle=args.shuffle)
    for line in summary_lines(config, args.stage, args.episodes, args.success_distance,
                              fixed, yaml_success):
        print(line, flush=True)

    problems = check_split(args.stage, config)
    if problems:
        print("[错误] 该 split 在本机跑不了：")
        for problem in problems:
            print(f"  - {problem}")
        return 2

    print("\n--- 限速闸门（纯逻辑，不碰 env） ---", flush=True)
    check_topdown_gate()

    print("\n--- MJPEG 分帧（纯逻辑，不碰 env） ---", flush=True)
    check_mjpeg_double_part()

    env = habitat.Env(config)
    nav = Navigator(env, args, args.out,
                    max_episode_steps=config.habitat.environment.max_episode_steps)
    server = Play2NavServer(host=args.host, port=0, jpeg_quality=args.jpeg_quality,
                            display_scale=args.display_scale,
                            topdown_size=args.topdown_video_size,
                            hold_delay=args.hold_delay)

    episode_dirs, errors = [], []
    try:
        server.publish_placeholder()
        port = server.start()
        base = f"http://{args.host}:{port}"
        print(f"\n[web] 测试客户端连的是 {base}", flush=True)

        client = threading.Thread(target=client_checks,
                                  args=(base, episode_dirs, errors), daemon=True)
        client.start()

        try:
            _web_loop(nav, args, server)
        finally:
            if not nav.finished:
                nav.finish("quit_early")
        client.join(timeout=30)
    finally:
        server.stop()
        nav.close()
        try:
            env.close()
        except Exception:
            pass

    if errors:
        print("\n客户端抛异常：")
        for text in errors:
            print(text)

    print("\n--- 主线程侧校验 ---")
    verify_colour_order(episode_dirs)
    summary = os.path.join(options.out, "session_summary.json")
    check("会话汇总 session_summary.json 已写出", os.path.exists(summary), summary)
    if os.path.exists(summary):
        with open(summary, encoding="utf-8") as handle:
            data = json.load(handle)
        check("会话汇总里有两集（跨场景两集都落了盘）", data["num_episodes"] == 2,
              f"{data['num_episodes']} 集")
    check("两集落在不同目录",
          len(set(episode_dirs)) == len(episode_dirs) == 2, str(episode_dirs))
    check("/quit 之后服务端已停止应答", stopped_answer(base))

    if not options.keep and os.path.exists(FRAME0_PATH):
        os.remove(FRAME0_PATH)

    print("\n" + "=" * 78)
    passed = sum(1 for _n, ok, _d in _RESULTS if ok)
    print(f"{passed}/{len(_RESULTS)} 项通过")
    if _FAILED or errors:
        print("失败项：")
        for name in _FAILED:
            print(f"  - {name}")
        if errors:
            print(f"  - 客户端抛了 {len(errors)} 个异常")
        print("=" * 78)
        return 1
    print("全部通过")
    print("=" * 78)
    return 0


def stopped_answer(base, timeout=5.0):
    """服务端是否已经不应答（/quit 之后应当如此）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            http_get(base, "/state", timeout=1.0)
        except Exception:
            return True
        time.sleep(0.2)
    return False


if __name__ == "__main__":
    sys.exit(main())
