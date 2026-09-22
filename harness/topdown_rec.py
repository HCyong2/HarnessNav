"""把 Habitat ``top_down_map`` 度量录成 topdown.mp4。"""

import os

import numpy as np

from play2nav import topdown as td

try:
    import imageio
except Exception:
    imageio = None


class TopdownRecorder:
    """每步抓一帧 habitat 俯视度量。"""

    def __init__(self, env, out_dir, fps=5, size=512):
        """打开写出器。

        Args:
            env: ``habitat.Env``。
            out_dir (str): 本集目录。
            fps (int): 帧率。
            size (int): 编码长边。
        """
        self.env = env
        self.path = os.path.join(out_dir, "topdown.mp4")
        self.size = int(size)
        self.frames = 0
        self._writer = None
        if imageio is None:
            return
        os.makedirs(out_dir, exist_ok=True)
        try:
            self._writer = imageio.get_writer(
                self.path, fps=max(int(fps), 1), quality=8,
                macro_block_size=None, ffmpeg_log_level="error")
        except Exception:
            self._writer = None

    def capture(self):
        """读当前 ``top_down_map`` 上色写入。"""
        if self._writer is None:
            return
        try:
            canvas, info = td.build_topdown_image(self.env)
            canvas = td.draw_agents(canvas, info)
            frame = td.resize_for_video(canvas, self.size)
            self._writer.append_data(np.ascontiguousarray(frame, dtype=np.uint8))
            self.frames += 1
        except Exception:
            return

    def close(self):
        """关闭文件。"""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        rec = getattr(self.env, "_hn_topdown_rec", None)
        if rec is self:
            self.env._hn_topdown_rec = None


def attach_step_capture(env, recorder):
    """在 ``env.step`` 返回后抓一帧；同一 env 多集复用原 step。

    Args:
        env: ``habitat.Env``。
        recorder (TopdownRecorder): 本集录像器。
    """
    if not hasattr(env, "_hn_step_orig"):
        orig = env.step

        def hooked(action):
            obs = orig(action)
            rec = getattr(env, "_hn_topdown_rec", None)
            if rec is not None:
                rec.capture()
            return obs

        env._hn_step_orig = orig
        env.step = hooked
    env._hn_topdown_rec = recorder
