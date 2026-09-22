"""按键状态机：轻触出一步，按住满门槛后持续输出。不依赖界面，可单测。"""

from typing import Optional

# 键位 -> habitat 动作名。
KEY_ACTION = {
    "w": "move_forward",
    "a": "turn_left",
    "d": "turn_right",
    "q": "look_up",
    "e": "look_down",
}

# 多键同时按住时每个 tick 只发一个动作，按此优先级取。
HOLD_PRIORITY = ("w", "a", "d", "q", "e")

# 松手后仍视为按住的宽限期（秒），用来覆盖心跳与松手交错到达。
RELEASE_GRACE_S = 0.06

# 连续按住达到该时长后转为持续输出；轻触只出一步。``--hold-delay 0`` 表示按下即持续。
HOLD_DELAY_S = 1.0

# 轻触那一步的有效期（秒）。不要求取键时键盘仍按着。
TAP_PENDING_S = 2.0


def normalize(keysym: str) -> Optional[str]:
    """把界面层给的键名归一化成小写单字符；无法识别则返回 ``None``。

    Args:
        keysym (str): 浏览器或界面给出的键名。

    Returns:
        Optional[str]: 小写单字符，或 ``None``。
    """
    if not keysym or len(keysym) != 1:
        return None
    return keysym.lower()


class KeyState:
    """跟踪当前按住的移动键，并决定本 tick 该发哪个动作。"""

    def __init__(self, grace: float = RELEASE_GRACE_S):
        """初始化按键状态。

        Args:
            grace (float): 松手宽限期，秒。
        """
        self._grace = float(grace)
        self._pressed_at: dict = {}
        self._released_at: dict = {}
        # False 表示按下后那一步还没发出去。
        self._tap_used: dict = {}

    def press(self, keysym: str, now: float, heartbeat: bool = False) -> Optional[str]:
        """记录一次按下。心跳不改计时、不补发轻触步、不复活已松开的键。

        Args:
            keysym (str): 原始键名。
            now (float): 当前时间戳，秒。
            heartbeat (bool): True 表示前端重申「仍按着」，不是新的一次按下。

        Returns:
            Optional[str]: 归一化键名；无法识别则 ``None``。
        """
        key = normalize(keysym)
        if key is None or key not in KEY_ACTION:
            return None
        if heartbeat:
            return key if self.is_held(key, now) else None
        self._pressed_at[key] = now
        self._tap_used[key] = False
        self._released_at.pop(key, None)
        return key

    def release(self, keysym: str, now: float) -> Optional[str]:
        """记录一次松手。

        Args:
            keysym (str): 原始键名。
            now (float): 当前时间戳，秒。

        Returns:
            Optional[str]: 归一化键名；无法识别则 ``None``。
        """
        key = normalize(keysym)
        if key is None or key not in KEY_ACTION:
            return None
        self._released_at[key] = now
        return key

    def is_held(self, key: str, now: float) -> bool:
        """判断 ``key`` 此刻是否算按住（含松手宽限期）。

        Args:
            key (str): 归一化键名。
            now (float): 当前时间戳，秒。

        Returns:
            bool: 按住为 True。
        """
        if key not in self._pressed_at:
            return False
        released = self._released_at.get(key)
        return released is None or (now - released) < self._grace

    def held_keys(self, now: float) -> list:
        """返回当前按住的键，按 :data:`HOLD_PRIORITY` 排序。

        Args:
            now (float): 当前时间戳，秒。

        Returns:
            list: 键名列表。
        """
        return [k for k in HOLD_PRIORITY if self.is_held(k, now)]

    def hold_elapsed(self, key: str, now: float) -> Optional[float]:
        """返回 ``key`` 已连续按住的时长。

        Args:
            key (str): 归一化键名。
            now (float): 当前时间戳，秒。

        Returns:
            Optional[float]: 秒；未按住则为 ``None``。
        """
        if key not in self._pressed_at or not self.is_held(key, now):
            return None
        return max(now - self._pressed_at[key], 0.0)

    def active_key(self, now: float, hold_delay: float = 0.0) -> Optional[str]:
        """取出本 tick 该发动作的键。会把轻触步标记为已发出。

        Args:
            now (float): 当前时间戳，秒。
            hold_delay (float): 转为持续输出所需的按住时长；0 表示按下即持续。

        Returns:
            Optional[str]: 键名；没有可发动作则为 ``None``。
        """
        for key in HOLD_PRIORITY:
            if self.pending_tap(key, now):
                self._tap_used[key] = True
                return key
            if self.is_held(key, now) and (now - self._pressed_at[key]) >= hold_delay:
                return key
        return None

    def pending_tap(self, key: str, now: float) -> bool:
        """判断 ``key`` 是否还欠着一次未发出的轻触步。

        Args:
            key (str): 归一化键名。
            now (float): 当前时间戳，秒。

        Returns:
            bool: 仍欠轻触步为 True。
        """
        if self._tap_used.get(key, True):
            return False
        pressed = self._pressed_at.get(key)
        return pressed is not None and (now - pressed) <= TAP_PENDING_S

    def waiting_key(self, now: float, hold_delay: float) -> Optional[str]:
        """返回正在等待「转连续」门槛的键，供界面显示进度。

        Args:
            now (float): 当前时间戳，秒。
            hold_delay (float): 持续输出门槛，秒。

        Returns:
            Optional[str]: 键名；没有则为 ``None``。
        """
        if hold_delay <= 0:
            return None
        for key in HOLD_PRIORITY:
            if not self.is_held(key, now):
                continue
            if not self._tap_used.get(key, True):
                continue
            if (now - self._pressed_at[key]) < hold_delay:
                return key
        return None

    def clear(self) -> None:
        """清空全部按键状态。"""
        self._pressed_at.clear()
        self._released_at.clear()
        self._tap_used.clear()

    def debug(self, now: float, hold_delay: float = 0.0) -> str:
        """返回一行可读状态，供 ``--key-debug`` 打印。

        Args:
            now (float): 当前时间戳，秒。
            hold_delay (float): 持续输出门槛，秒。

        Returns:
            str: 单行状态。
        """
        held = self.held_keys(now)
        parts = []
        for key in held:
            elapsed = self.hold_elapsed(key, now)
            if hold_delay > 0 and elapsed < hold_delay:
                stage = "待连续" if self._tap_used.get(key, True) else "单步待发"
                parts.append(f"{key}:{elapsed:.1f}/{hold_delay:.1f}s({stage})")
            else:
                parts.append(f"{key}:moving")
        stale = [k for k in HOLD_PRIORITY if self.pending_tap(k, now) and not self.is_held(k, now)]
        tail = f"  待发(已松手)=[{''.join(stale)}]" if stale else ""
        return f"held=[{' '.join(parts)}]  pressed={sorted(self._pressed_at)}{tail}"
