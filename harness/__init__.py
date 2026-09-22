"""HarnessNav 双系统导航编排。"""

from harness.state import NavState

__all__ = ["NavState", "Harness"]


def __getattr__(name):
    """延迟加载 Harness，避免与 vlm 循环 import。"""
    if name == "Harness":
        from harness.loop import Harness
        return Harness
    raise AttributeError(name)
