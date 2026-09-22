"""浏览器界面：Flask 路由、帧槽、按键状态与意图标志。

HTTP 线程只改按键与意图标志，不访问 ``habitat.Env``。sim 线程独占 ``env``，
每 tick 末尾发布一份只含基本类型的快照给 ``/state``。意图用幂等标志，不用队列。
页面按帧号拉取 ``/frame.jpg`` 与 ``/topdown.jpg``；``/stream`` 留给 curl 与自检。
"""

import json
import logging
import os
import socket
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from werkzeug.serving import WSGIRequestHandler, make_server

from .keystate import HOLD_DELAY_S, HOLD_PRIORITY, KEY_ACTION, KeyState, normalize
from . import topdown as td

# 无新帧时，隔多久把缓存 JPEG 再发一次，避免中间代理掐掉静默连接。
MJPEG_KEEPALIVE_S = 10.0

# 生成器等待新帧的上限。客户端断开后，生成器在下一次写出时才能发现。
MJPEG_WAIT_S = 1.0

# 超过该时长未收到 ``POST /key``（含心跳）则松开全部按键。不计 ``/state`` 轮询。
KEY_DEADMAN_S = 5.0

# 「输入检测」面板保留多少条事件。滚动显示最近发生的按键事件与提示。
INPUT_LOG_MAX = 60

_LOGGER = logging.getLogger(__name__)


class _QuietRequestHandler(WSGIRequestHandler):
    """静音 access log，保留报错输出。关闭 Nagle，避免流式 JPEG 被延迟 ACK 拖住。"""
    disable_nagle_algorithm = True

    def log_request(self, code="-", size="-"):
        pass


class _FrameSlot:
    """只保留最新一帧：发布方写入，多个读者等待序号变化。"""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self._stopping = False
        self._bytes = 0
        self._readers = 0

    def publish(self, jpeg):
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            if jpeg is not None:
                self._bytes += len(jpeg)
            self._cond.notify_all()

    def stats(self):
        with self._cond:
            return {"seq": self._seq, "bytes": self._bytes, "readers": self._readers}

    def enter(self):
        with self._cond:
            self._readers += 1

    def leave(self):
        with self._cond:
            self._readers = max(self._readers - 1, 0)

    def wait_next(self, last_seq, timeout):
        """等到 ``seq`` 变化或超时；返回 ``(seq, jpeg)``。"""
        with self._cond:
            if self._seq == last_seq and not self._stopping:
                self._cond.wait(timeout)
            return self._seq, self._jpeg

    def current(self):
        with self._cond:
            return self._jpeg

    def stop(self):
        """叫醒所有等在 ``wait_next`` 里的读者，让它们退出。"""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()

    @property
    def stopping(self):
        with self._cond:
            return self._stopping


def _encode_jpeg(rgb, quality, scale=1.0):
    """把 RGB 图编码为 JPEG 字节。``cv2.imencode`` 需要 BGR，所以先转换。

    Args:
        rgb: ``(H, W, 3)`` RGB 数组。
        quality (int): JPEG 质量。
        scale (float): 缩放系数。

    Returns:
        bytes or None: JPEG 数据。
    """
    image = np.asarray(rgb, dtype=np.uint8)
    if scale and scale != 1.0:
        image = cv2.resize(image, (max(int(image.shape[1] * scale), 1),
                                   max(int(image.shape[0] * scale), 1)),
                           interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                              [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buffer.tobytes() if ok else None


class Play2NavServer:
    """网页服务。``start()`` 之后由 sim 线程调用 ``publish_*`` / ``publish_snapshot``。"""

    def __init__(self, host="0.0.0.0", port=8080, jpeg_quality=80, display_scale=1.0,
                 topdown_size=512, key_deadman_s=KEY_DEADMAN_S, hold_delay=HOLD_DELAY_S):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.display_scale = float(display_scale) if display_scale else 1.0
        self.topdown_size = int(topdown_size)
        self.key_deadman_s = float(key_deadman_s)
        self.hold_delay = float(hold_delay)

        self._lock = threading.Lock()
        self._keys = KeyState()
        self._pending = {"next": False, "stop": False, "quit": False}
        self._last_key_assert = time.monotonic()
        self._last_key_seq = None

        self._input_log = deque(maxlen=INPUT_LOG_MAX)
        self._logged_down = set()
        self._last_note_text = None
        self._t0 = time.monotonic()

        self._frame = _FrameSlot()
        self._topdown = _FrameSlot()
        self._topdown_jpeg = None

        self.snapshot = {
            "phase": "loading",
            "scene_id": "?", "episode_id": -1, "category": "n/a",
            "step_idx": 0, "max_steps": 0, "navmesh_ok": False,
            "frame_seq": 0, "status_lines": ["正在加载场景…"],
            "terminated_by": None, "metrics": None, "episode_dir": None,
            "last_error": None, "session": [], "repeated": False,
        }

        self._stop_event = threading.Event()
        self._server = None
        self._thread = None
        self._page = None
        self.app = Flask(__name__, static_folder=None)
        self._register_routes()

    # -- 页面 ------------------------------------------------------------- #

    def _page_html(self):
        if self._page is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "page.html")
            with open(path, "r", encoding="utf-8") as fh:
                self._page = fh.read()
        return self._page

    def _register_routes(self):
        app = self.app

        @app.get("/")
        def index():
            return Response(self._page_html(),
                            content_type="text/html; charset=utf-8",
                            headers={"Cache-Control": "no-store"})

        @app.get("/stream")
        def stream():
            return Response(self._mjpeg(self._frame),
                            mimetype="multipart/x-mixed-replace; boundary=frame",
                            headers={"Cache-Control": "no-store",
                                     "X-Accel-Buffering": "no"})

        @app.get("/topdown_stream")
        def topdown_stream():
            return Response(self._mjpeg(self._topdown),
                            mimetype="multipart/x-mixed-replace; boundary=frame",
                            headers={"Cache-Control": "no-store",
                                     "X-Accel-Buffering": "no"})

        @app.get("/frame.jpg")
        def frame_jpg():
            return self._jpeg_response(self._frame.current(), "rgb")

        @app.get("/topdown.jpg")
        def topdown_jpg():
            return self._jpeg_response(self._topdown_jpeg, "topdown")

        @app.get("/state")
        def state():
            return Response(json.dumps(self.snapshot), mimetype="application/json")

        @app.post("/key")
        def key():
            data = request.get_json(silent=True) or {}
            key_name = data.get("k")
            seq = data.get("seq")
            down = bool(data.get("down"))
            now = time.monotonic()
            with self._lock:
                if isinstance(seq, (int, float)):
                    if self._last_key_seq is not None and seq <= self._last_key_seq:
                        self._note_locked("按键事件乱序到达，已丢弃过期的那个", "warn")
                        return jsonify({"ok": False, "why": "stale-seq"})
                    self._last_key_seq = seq
                self._last_key_assert = now
                if down:
                    key_id = self._keys.press(key_name, now, heartbeat=bool(data.get("hb")))
                    if key_id is not None and key_id not in self._logged_down:
                        self._logged_down.add(key_id)
                        self._note_locked(f"{key_id} ↓ 按下 → {KEY_ACTION[key_id]}", "down")
                else:
                    key_id = normalize(key_name)
                    elapsed = self._keys.hold_elapsed(key_id, now) if key_id else None
                    key_id = self._keys.release(key_name, now)
                    if key_id is not None:
                        self._logged_down.discard(key_id)
                        self._note_locked(self._release_text(key_id, elapsed), "up")
            return jsonify({"ok": True})

        @app.post("/keys/clear")
        def keys_clear():
            with self._lock:
                if self._logged_down:
                    self._note_locked(
                        f"页面失焦/切走/关闭，已清空按住的键 {sorted(self._logged_down)}", "warn")
                self._keys.clear()
                self._logged_down.clear()
            return jsonify({"ok": True})

        @app.post("/next")
        def next_episode():
            return self._intent("next")

        @app.post("/stop")
        def stop():
            return self._intent("stop")

        @app.post("/quit")
        def quit_():
            return self._intent("quit")

    def _jpeg_response(self, jpeg, name):
        if jpeg is None:
            return Response(status=204)
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store", "X-Frame": name})

    # -- 「输入检测」事件流 ------------------------------------------------- #

    def _note_locked(self, text, kind="info"):
        """写入事件流。调用方须已持有 ``self._lock``。连续重复的消息只记一次。"""
        if text == self._last_note_text:
            return
        self._last_note_text = text
        self._input_log.append((time.monotonic(), kind, text))

    def note(self, text, kind="info"):
        """对外的事件流入口（sim 线程用）。"""
        with self._lock:
            self._note_locked(text, kind)

    def _release_text(self, key_id, elapsed):
        """松手事件文案：区分轻触一步与按住满门槛后的持续移动。"""
        if elapsed is None:
            return f"{key_id} ↑ 松开"
        if self.hold_delay > 0 and elapsed < self.hold_delay:
            return (f"{key_id} ↑ 松开（只按了 {elapsed:.2f}s，不足 "
                    f"{self.hold_delay:.1f}s 门槛 → 只出轻触的一步）")
        return (f"{key_id} ↑ 松开（按住 {elapsed:.2f}s，超过 "
                f"{self.hold_delay:.1f}s 门槛 → 期间是持续移动）")

    def input_log(self):
        """给 ``/state`` 用的事件列表：``[{t, kind, text}]``，最早的在前。"""
        with self._lock:
            base = self._t0
            return [{"t": round(t - base, 2), "kind": k, "text": s}
                    for t, k, s in self._input_log]

    def stream_stats(self):
        """两条流的累计帧数与字节数，供页面算「服务端到底推了多少」。"""
        rgb, top = self._frame.stats(), self._topdown.stats()
        return {
            "rgb_frames": rgb["seq"], "rgb_bytes": rgb["bytes"], "rgb_readers": rgb["readers"],
            "topdown_frames": top["seq"], "topdown_bytes": top["bytes"],
            "topdown_readers": top["readers"],
            "jpeg_quality": self.jpeg_quality, "display_scale": self.display_scale,
        }

    def _intent(self, name):
        with self._lock:
            self._pending[name] = True
        if name in ("next", "quit"):
            self._set_phase("loading", ["正在加载场景…"] if name == "next" else ["正在退出…"])
        return jsonify({"ok": True})

    def _set_phase(self, phase, status_lines=None):
        """整体替换快照，只改 ``phase`` / ``status_lines``。不依赖 env。"""
        snapshot = dict(self.snapshot)
        snapshot["phase"] = phase
        if status_lines is not None:
            snapshot["status_lines"] = list(status_lines)
        self.snapshot = snapshot

    # -- 供 sim 线程调用 --------------------------------------------------- #

    def take_intents(self):
        """取走并清空意图标志。调用方拿到返回值后立刻释放锁，再去访问 ``env``。"""
        with self._lock:
            pending = dict(self._pending)
            for name in self._pending:
                self._pending[name] = False
            return pending

    def active_key(self, now=None):
        """当前应当发动作的键（按住集合里优先级最高、且已按够门槛的），没有则 ``None``。"""
        now = time.monotonic() if now is None else now
        with self._lock:
            return self._keys.active_key(now, self.hold_delay)

    def key_status(self, now=None):
        """返回 ``(按住的键, 等待门槛的键, 已按住时长, 欠着轻触步的键)``。"""
        now = time.monotonic() if now is None else now
        with self._lock:
            held = self._keys.held_keys(now)
            waiting = self._keys.waiting_key(now, self.hold_delay)
            elapsed = self._keys.hold_elapsed(waiting, now) if waiting else None
            pending = [k for k in HOLD_PRIORITY if self._keys.pending_tap(k, now)]
            return held, waiting, elapsed, pending

    def housekeeping(self, now=None):
        """太久没收到 ``POST /key`` 则松开全部按键。由 sim 线程每 tick 调用。"""
        now = time.monotonic() if now is None else now
        with self._lock:
            if (now - self._last_key_assert) > self.key_deadman_s:
                if self._logged_down:
                    self._note_locked(
                        f"⚠ {self.key_deadman_s:.0f}s 没收到按键心跳，已松开全部按键"
                        f"（页面被切走 / 断网 / 笔记本睡眠）", "warn")
                self._keys.clear()
                self._logged_down.clear()

    def key_debug(self, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            return self._keys.debug(now, self.hold_delay)

    def publish_rgb(self, rgb):
        """发布一帧第一视角观测（编码后进帧槽）。"""
        jpeg = _encode_jpeg(rgb, self.jpeg_quality, self.display_scale)
        if jpeg is not None:
            self._frame.publish(jpeg)

    def publish_topdown(self, canvas):
        """发布一张俯视图到第二条流和 ``/topdown.jpg``。直接编码传入的画布。"""
        if canvas is None:
            return
        try:
            resized = td.resize_for_video(canvas, self.topdown_size)
        except Exception as exc:
            _LOGGER.warning("topdown 缩放失败：%s", exc)
            return
        jpeg = _encode_jpeg(resized, self.jpeg_quality)
        self._topdown_jpeg = jpeg
        if jpeg is not None:
            self._topdown.publish(jpeg)

    def publish_placeholder(self, text="loading scene..."):
        """开服前放入占位图，覆盖首次 ``env.reset()`` 的等待时间。"""
        height, width = 480, 640
        image = np.full((height, width, 3), 24, dtype=np.uint8)
        cv2.putText(image, text, (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 2, cv2.LINE_AA)
        jpeg = _encode_jpeg(image, self.jpeg_quality)
        self._frame.publish(jpeg)
        self._topdown_jpeg = jpeg
        self._topdown.publish(jpeg)

    def publish_snapshot(self, **fields):
        """sim 线程发布状态快照：整体替换一个只含基本类型的 dict。"""
        snapshot = dict(self.snapshot)
        snapshot.update(fields)
        self.snapshot = snapshot

    # -- 生命周期 ---------------------------------------------------------- #

    def _mjpeg(self, slot):
        """multipart MJPEG 生成器。同一帧连续 yield 两次，浏览器才能把当前 part 上屏。"""
        last_seq, last_send = -1, 0.0
        slot.enter()
        try:
            while not self._stop_event.is_set():
                seq, jpeg = slot.wait_next(last_seq, MJPEG_WAIT_S)
                if self._stop_event.is_set():
                    return
                now = time.monotonic()
                if jpeg is None:
                    continue
                if seq == last_seq and (now - last_send) < MJPEG_KEEPALIVE_S:
                    continue
                last_seq, last_send = seq, now
                part = (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                yield part
                yield part
        finally:
            slot.leave()

    def start(self):
        """绑定端口，在 daemon 线程里启动 Werkzeug。返回实际监听端口。"""
        self._server = make_server(self.host, self.port, self.app, threaded=True,
                                   request_handler=_QuietRequestHandler)
        self.port = int(self._server.server_port)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="play2nav-http", daemon=True)
        self._thread.start()
        return self.port

    def url(self):
        """给人看的访问地址（监听 0.0.0.0 时给出可路由的本机 IP）。"""
        host = self.host
        if host in ("0.0.0.0", "::", ""):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.connect(("10.255.255.255", 1))
                    host = sock.getsockname()[0]
            except Exception:
                host = "127.0.0.1"
        return f"http://{host}:{self.port}"

    def stop(self):
        """叫醒流读者并关闭服务。测试需要显式调用；正常退出依赖 daemon 线程。"""
        self._stop_event.set()
        self._frame.stop()
        self._topdown.stop()
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
            try:
                self._server.server_close()
            except Exception:
                pass
