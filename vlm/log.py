"""终端与 debug.txt 共用的时间戳流水。"""

import os
from datetime import datetime


class RunLog:
    """打印 ``HH:MM:SS.mmm [Component] 消息``，并追加到文件。"""

    def __init__(self, path=None):
        """初始化。

        Args:
            path (str, optional): debug.txt 路径。
        """
        self.path = path
        self.lines = []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("")

    def bind(self, path):
        """改绑输出文件并清空。"""
        self.path = path
        self.lines = []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("")

    def emit(self, component, message):
        """打一行并落盘。

        Args:
            component (str): Planner / Depth / Mover 等。
            message (str): 动作或返回摘要。
        """
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"{stamp} [{component}] {message}"
        print(line, flush=True)
        self.lines.append(line)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return line
