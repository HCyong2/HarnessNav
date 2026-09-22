"""压低 Habitat 加载场景时的终端噪音。

须在 ``import habitat`` 之前被导入：环境变量只在仿真库尚未加载时生效。
语义网格加载失败走 C 层标准错误，须在文件描述符上拦截。
"""

import logging
import os
import sys
import threading
import warnings

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("HABITAT_LAB_LOG", str(logging.ERROR))
os.environ.setdefault("GLOG_minloglevel", "2")

# 缺少语义网格、Gymnasium 迁移提示等。
warnings.filterwarnings("ignore", message=r".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=r".*[Ss]emantic [Ss]cene.*")
warnings.filterwarnings("ignore", message=r".*semantic information.*")
warnings.filterwarnings("ignore", message=r".*No semantic.*")

_DROP_SUBSTR = (
    "initializing dataset",
    "initializing sim",
    "initializing task",
    "semantic scene",
    "semantic information",
    "no semantic information",
    "ssd load",
    "ssd file",
    "semanticattributes",
    "loadsemanticscenedescriptor",
    "semanticscene.cpp",
    "exists but failed to load",
    "gym has been unmaintained",
    "please upgrade to gymnasium",
    "migration_guide",
)

_fd_installed = set()


class _FilterStream:
    """丢掉 Habitat 加载场景与缺少语义网格的行，其余原样写出。"""

    def __init__(self, inner):
        self._inner = inner
        self._buf = ""

    def write(self, text):
        if not text:
            return 0
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not _should_drop(line):
                self._inner.write(line + "\n")
        return len(text)

    def flush(self):
        if self._buf:
            if not _should_drop(self._buf):
                self._inner.write(self._buf)
            self._buf = ""
        self._inner.flush()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _should_drop(line):
    """该行是否属于应过滤的 Habitat 加载或语义告警。

    Args:
        line (str): 尚未带换行的一行。

    Returns:
        bool: 应丢弃为 True。
    """
    low = str(line).lower()
    return any(token in low for token in _DROP_SUBSTR)


def _install_fd_filter(fd):
    """拦截 C 层对给定文件描述符的写入并丢掉语义网格失败行。

    Args:
        fd (int): 一般为标准错误 ``2``。
    """
    if fd in _fd_installed:
        return
    try:
        saved = os.dup(fd)
    except OSError:
        return
    try:
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        os.close(write_fd)
    except OSError:
        os.close(saved)
        return

    def pump():
        buf = b""
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
                buf += chunk.replace(b"\r", b"\n")
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace")
                    if _should_drop(text):
                        continue
                    try:
                        os.write(saved, line + b"\n")
                    except OSError:
                        return
            if buf:
                text = buf.decode("utf-8", errors="replace")
                if not _should_drop(text):
                    try:
                        os.write(saved, buf)
                    except OSError:
                        pass
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass
            try:
                os.close(saved)
            except OSError:
                pass

    threading.Thread(
        target=pump, name="habitat-log-filter", daemon=True).start()
    _fd_installed.add(fd)


def redirect_c_stdio(fileobj):
    """把 C 层标准输出与标准错误接到 ``fileobj``，避免仿真库写到终端。

    Args:
        fileobj: 已打开的文本文件。
    """
    fileno = fileobj.fileno()
    os.dup2(fileno, 1)
    os.dup2(fileno, 2)
    _fd_installed.discard(2)


def apply_io_filter():
    """过滤 Python 与 C 层标准错误中的 Habitat 加载日志。"""
    if not isinstance(sys.stdout, _FilterStream):
        sys.stdout = _FilterStream(sys.stdout)
    if not isinstance(sys.stderr, _FilterStream):
        sys.stderr = _FilterStream(sys.stderr)
    _install_fd_filter(2)
    try:
        from habitat.core.logging import logger
        logger.setLevel(logging.ERROR)
    except Exception:
        pass


apply_io_filter()
