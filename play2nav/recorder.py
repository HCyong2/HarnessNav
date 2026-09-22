"""每集产物落盘：流式 mp4、轨迹图、metrics.json、会话汇总。"""

import json
import os
import time

import numpy as np

from . import topdown as td

try:
    import imageio
except Exception:  # pragma: no cover
    imageio = None


def make_episode_dir(out_root, scene_id, episode_id, category, when=None):
    """创建本集输出目录 ``{out_root}/{时间}_{场景}_{epid}_{类别}/``。

    Args:
        out_root (str): 产物根目录。
        scene_id (str): 场景名。
        episode_id: 集号。
        category (str): 目标类别。
        when (float, optional): 时间戳；默认当前时间。

    Returns:
        str: 已创建的目录路径。
    """
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(when))
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(category))
    name = f"{stamp}_{scene_id}_{episode_id}_{safe}"
    path = os.path.join(out_root, name)
    suffix = 1
    while os.path.exists(path):
        suffix += 1
        path = os.path.join(out_root, f"{name}_{suffix}")
    os.makedirs(path, exist_ok=True)
    return path


class EpisodeRecorder:
    """一集的录像与指标写出器。``enabled=False`` 时全部为空操作。"""

    def __init__(self, out_dir, fps=10, topdown_video_size=512, enabled=True, quiet=False):
        """打开本集的两个 mp4 编码器。

        Args:
            out_dir (str): 本集目录。
            fps (int): 录像帧率。
            topdown_video_size (int): 俯视回放编码边长。
            enabled (bool): 是否真正写盘。
            quiet (bool): 是否抑制警告。
        """
        self.out_dir = out_dir
        self.fps = max(int(fps), 1)
        self.topdown_video_size = int(topdown_video_size)
        self.enabled = bool(enabled)
        self.quiet = quiet

        self.frames = 0
        self.closed = False
        self._rgb_writer = None
        self._topdown_writer = None
        self._warned = False

        if self.enabled:
            os.makedirs(self.out_dir, exist_ok=True)
            self._rgb_writer = self._open("rgb.mp4")
            self._topdown_writer = self._open("topdown.mp4")

    def _open(self, filename):
        """打开一个 mp4 写出器。

        Args:
            filename (str): 文件名。

        Returns:
            imageio 写出器，或打开失败时为 ``None``。
        """
        if imageio is None:
            self._warn("imageio 不可用，跳过 mp4 录制")
            return None
        path = os.path.join(self.out_dir, filename)
        try:
            return imageio.get_writer(path, fps=self.fps, quality=8,
                                      macro_block_size=None, ffmpeg_log_level="error")
        except Exception as exc:
            self._warn(f"打开 {filename} 失败：{exc}")
            return None

    def _warn(self, message):
        if not self._warned:
            print(f"[recorder] {message}")
            self._warned = True

    def _append(self, writer, frame):
        if writer is None:
            return
        try:
            writer.append_data(np.ascontiguousarray(frame, dtype=np.uint8))
        except Exception as exc:
            self._warn(f"写帧失败：{exc}")

    def add_step(self, rgb, topdown_canvas=None, caption=None):
        """记录一个动作步：RGB 原样入录像，俯视图先缩放到编码尺寸再压信息条。

        Args:
            rgb (np.ndarray): 第一视角 RGB。
            topdown_canvas (np.ndarray, optional): 俯视画布。
            caption: 压在俯视图上的文字。
        """
        if not self.enabled or self.closed:
            return
        self._append(self._rgb_writer, rgb)

        if topdown_canvas is not None:
            frame = td.resize_for_video(topdown_canvas, self.topdown_video_size)
            if caption:
                frame = td.draw_caption(frame, caption)
            self._append(self._topdown_writer, frame)

        self.frames += 1

    def finalize(self, metrics, trajectory_canvas=None):
        """关闭编码器，写出 metrics.json 与 trajectory.png。

        Args:
            metrics (dict): 本集指标。
            trajectory_canvas (np.ndarray, optional): 终帧俯视图。

        Returns:
            dict: 写出的文件路径；未启用则为 ``None``。
        """
        if not self.enabled:
            self.close()
            return None

        self.close()

        metrics = dict(metrics)
        metrics["video_fps"] = self.fps
        metrics["video_frames"] = self.frames
        metrics["out_dir"] = self.out_dir

        path = os.path.join(self.out_dir, "metrics.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)

        traj_path = None
        if trajectory_canvas is not None:
            traj_path = td.save_image_rgb(
                os.path.join(self.out_dir, "trajectory.png"), trajectory_canvas)

        return {"metrics": path, "trajectory": traj_path}

    def close(self):
        """幂等地关闭两个编码器。"""
        if self.closed:
            return
        self.closed = True
        for attr in ("_rgb_writer", "_topdown_writer"):
            writer = getattr(self, attr, None)
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:
                    self._warn(f"关闭编码器失败：{exc}")
                setattr(self, attr, None)


def write_session_summary(out_root, records):
    """写出会话级汇总 JSON。

    Args:
        out_root (str): 产物根目录。
        records (list): 各集指标。

    Returns:
        str: ``session_summary.json`` 路径；无记录则为 ``None``。
    """
    if not records:
        return None

    def mean(key):
        vals = [r[key] for r in records if isinstance(r.get(key), (int, float))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_episodes": len(records),
        "episodes": records,
        "mean": {k: mean(k) for k in ("success", "spl", "soft_spl",
                                      "distance_to_goal", "path_length_m", "num_steps")},
        "num_success": int(sum(1 for r in records if r.get("success"))),
    }
    path = os.path.join(out_root, "session_summary.json")
    os.makedirs(out_root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return path
