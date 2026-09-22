#!/usr/bin/env python
"""``KeyState`` 单元测试：不依赖显示器、habitat 或 HTTP。

    python play2nav/test_keystate.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from play2nav.keystate import (  # noqa: E402
    HOLD_DELAY_S,
    HOLD_PRIORITY,
    KEY_ACTION,
    RELEASE_GRACE_S,
    TAP_PENDING_S,
    KeyState,
    normalize,
)

GRACE = RELEASE_GRACE_S
HOLD_DELAY = HOLD_DELAY_S


# --------------------------------------------------------------------------- #
# 键名归一化
# --------------------------------------------------------------------------- #

def test_normalize_lowercases_single_chars():
    """单字符键名统一成小写。"""
    assert normalize("w") == "w"
    assert normalize("W") == "w"
    assert normalize("A") == "a"
    assert normalize("Q") == "q"


def test_normalize_rejects_non_single_char_keysyms():
    """多字符 keysym 不是移动键。"""
    for keysym in ("", "space", "Shift_L", "Control_L", "Escape", "F5", "Up"):
        assert normalize(keysym) is None, keysym


def test_press_returns_none_for_unmapped_keys():
    state = KeyState()
    assert state.press("z", 0.0) is None
    assert state.press("Shift_L", 0.0) is None
    assert state.press("p", 0.0) is None          # p 是事件，由 app.py 直接绑，不进状态机
    assert state.active_key(0.0) is None


# --------------------------------------------------------------------------- #
# 核心：按住 = 按下集合减去松手（浏览器按住期间只发一次按下）
# --------------------------------------------------------------------------- #

def test_press_alone_counts_as_held_indefinitely():
    """只发一次按下、没有松手时，键一直算按住。"""
    state = KeyState()
    state.press("w", 0.0)
    for t in (0.05, 1.0, 10.0, 600.0):
        assert state.is_held("w", t), t
        assert state.active_key(t) == "w", t


def test_repeated_keypress_while_holding_stays_held():
    """按住期间的心跳不改变按住状态。"""
    state = KeyState()
    state.press("w", 0.0)                     # 真的按下（心跳之前必有一次真实按下）
    for i in range(1, 50):
        state.press("w", i * 0.03, heartbeat=True)
        assert state.active_key(i * 0.03) == "w"
        assert state.held_keys(i * 0.03) == ["w"]


def test_first_heartbeat_alone_does_not_press_the_key():
    """没有真实按下时，单独的心跳不能按下键。"""
    state = KeyState()
    assert state.press("w", 0.0, heartbeat=True) is None
    assert not state.is_held("w", 0.0)
    assert state.active_key(0.0, HOLD_DELAY) is None
    assert state.active_key(10.0, HOLD_DELAY) is None


def test_release_takes_effect_after_grace_window():
    """松手后宽限期内仍算按住，宽限期结束后失效。"""
    state = KeyState()
    state.press("w", 0.0)
    state.release("w", 1.0)
    state.active_key(0.0, HOLD_DELAY)                   # 轻触那一步：早已出手

    assert state.is_held("w", 1.0)                      # 松手当刻
    assert state.is_held("w", 1.0 + GRACE * 0.5)        # 宽限期内
    assert not state.is_held("w", 1.0 + GRACE)          # 边界：< 而非 <=
    assert not state.is_held("w", 1.0 + GRACE + 1e-6)
    assert state.active_key(1.0 + GRACE + 0.5) is None


def test_gapped_repeat_pairs_survive_grace_window():
    """心跳与松手交错到达时，键仍保持按住。"""
    state = KeyState()
    now = 0.0
    state.press("w", now)                     # 最早那一次是真的按下
    for _ in range(200):                      # 模拟 2 秒的按压重复序列
        state.press("w", now, heartbeat=True)
        now += 0.001
        state.release("w", now)
        assert state.active_key(now) == "w", now
        now += 0.001
        assert state.active_key(now) == "w", now


def test_repress_cancels_previous_release():
    """宽限期内再次按下，取消上一次松手。"""
    state = KeyState()
    state.press("w", 0.0)
    state.release("w", 1.0)
    assert not state.is_held("w", 1.0 + GRACE + 0.01)   # 先确认已经失效
    state.press("w", 1.5)
    assert state.is_held("w", 100.0)                    # 重新按下后长期有效
    assert state.active_key(100.0) == "w"


# --------------------------------------------------------------------------- #
# 多键同按的优先级
# --------------------------------------------------------------------------- #

def test_highest_priority_key_wins():
    """同时按住多个键时每 tick 只发一个动作：w > a > d > q > e。"""
    state = KeyState()
    state.press("a", 0.0)
    state.press("d", 0.0)
    assert state.active_key(0.0) == "a"

    state.press("w", 0.0)                     # w 优先级最高，后来居上
    assert state.active_key(0.0) == "w"

    state.release("w", 1.0)
    assert state.active_key(1.0 + GRACE + 0.01) == "a"   # w 失效后回落到 a


def test_priority_order_matches_key_action_table():
    """优先级表覆盖全部移动键且无重复。"""
    assert set(HOLD_PRIORITY) == set(KEY_ACTION)
    assert len(HOLD_PRIORITY) == len(set(HOLD_PRIORITY))


def test_low_priority_keys_alone_still_work():
    for key in HOLD_PRIORITY:
        state = KeyState()
        state.press(key, 0.0)
        assert state.active_key(0.0) == key, key
        assert state.held_keys(0.0) == [key], key


# --------------------------------------------------------------------------- #
# 轻触一步 + 按住 HOLD_DELAY_S 转持续
# --------------------------------------------------------------------------- #

def test_tap_gives_exactly_one_step():
    """轻触只出一步，松手后不再补发。"""
    state = KeyState()
    state.press("w", 0.0)
    assert state.active_key(0.0, HOLD_DELAY) == "w"        # 按下即出这一步
    assert state.active_key(0.01, HOLD_DELAY) is None      # 取走就没了
    assert state.active_key(0.5, HOLD_DELAY) is None
    state.release("w", 0.05)
    assert state.active_key(0.05, HOLD_DELAY) is None      # 宽限期内也不补第二步
    assert state.active_key(HOLD_DELAY + 10.0, HOLD_DELAY) is None


def test_tap_step_of_a_lower_priority_key_is_not_lost():
    """同时按下时，低优先级键的轻触步在高优先级取走后仍会发出。"""
    state = KeyState()
    state.press("w", 0.0)
    state.press("a", 0.0)
    assert state.active_key(0.0, HOLD_DELAY) == "w"
    assert state.active_key(0.1, HOLD_DELAY) == "a"   # w 那一步已用过且没到门槛 → 轮到 a
    assert state.active_key(0.2, HOLD_DELAY) is None


def test_tap_step_survives_an_early_release():
    """按下后立刻松手，轻触步仍然欠着，可在之后取走。"""
    state = KeyState()
    state.press("w", 0.0)
    state.release("w", 0.05)                          # 用户这一下早就松手了
    assert state.active_key(0.23, HOLD_DELAY) == "w"  # 那一步照样发得出去
    assert state.active_key(0.23, HOLD_DELAY) is None # 而且只有一步
    assert not state.is_held("w", 0.23)


def test_pending_tap_expires_after_its_window():
    """未发出的轻触步超过 ``TAP_PENDING_S`` 后不再补发。"""
    state = KeyState()
    state.press("w", 0.0)
    state.release("w", 0.05)
    assert state.pending_tap("w", TAP_PENDING_S)      # 边界含等号
    assert state.active_key(TAP_PENDING_S, HOLD_DELAY) == "w"

    state = KeyState()
    state.press("w", 0.0)
    state.release("w", 0.05)
    assert not state.pending_tap("w", TAP_PENDING_S + 1e-6)
    assert state.active_key(TAP_PENDING_S + 1e-6, HOLD_DELAY) is None


def test_second_tap_inside_the_grace_window_is_a_new_press():
    """宽限期内的第二次按下算新的轻触，走出第二步。"""
    state = KeyState()
    state.press("w", 0.0)
    assert state.active_key(0.0, HOLD_DELAY) == "w"   # 第一下
    state.release("w", 0.05)
    state.press("w", 0.08)                            # 第二下：距松手 0.03 < 宽限期
    assert state.active_key(0.08, HOLD_DELAY) == "w"  # 第二下也要出一步
    assert state.active_key(0.20, HOLD_DELAY) is None
    assert state.hold_elapsed("w", 0.5) == 0.5 - 0.08
    assert state.active_key(0.08 + HOLD_DELAY, HOLD_DELAY) == "w"


def test_heartbeat_does_not_resurrect_a_released_key():
    """已松开且宽限期已过的键，迟到心跳不能把它重新按下。"""
    state = KeyState()
    state.press("w", 0.0)
    assert state.active_key(0.0, HOLD_DELAY) == "w"
    state.release("w", 0.1)
    dead = 0.1 + GRACE + 0.01
    assert state.press("w", dead, heartbeat=True) is None
    assert not state.is_held("w", dead)
    assert state.active_key(dead, HOLD_DELAY) is None

    state = KeyState()
    state.press("w", 0.0)
    assert state.active_key(0.0, HOLD_DELAY) == "w"
    state.release("w", 0.1)
    assert state.press("w", 0.1, heartbeat=True) == "w"
    assert state.active_key(0.1, HOLD_DELAY) is None
    assert not state.is_held("w", 0.1 + GRACE)


def test_debug_lists_the_tap_that_is_still_owed():
    """debug 字符串能标出已松手但仍欠着的轻触步。"""
    state = KeyState()
    state.press("w", 0.0)
    state.release("w", 0.05)
    later = 0.05 + GRACE + 0.05               # 宽限期已过：只剩那一步还欠着
    assert "待发(已松手)=[w]" in state.debug(later, HOLD_DELAY)
    state.active_key(later, HOLD_DELAY)
    assert "待发" not in state.debug(later, HOLD_DELAY)


def test_hold_delay_starts_continuous_output_after_threshold():
    """轻触那一步之后要一直按住满 ``HOLD_DELAY_S`` 才转连续，松开立刻停。"""
    state = KeyState()
    state.press("w", 0.0)
    assert state.active_key(0.0, HOLD_DELAY) == "w"                 # 轻触的一步
    assert state.active_key(HOLD_DELAY * 0.99, HOLD_DELAY) is None  # 门槛之前没有第二步
    assert state.active_key(HOLD_DELAY, HOLD_DELAY) == "w"          # 边界含等号
    assert state.active_key(HOLD_DELAY + 5.0, HOLD_DELAY) == "w"    # 按住期间一直出

    state.release("w", HOLD_DELAY + 5.0)
    assert state.active_key(HOLD_DELAY + 5.1, HOLD_DELAY) is None   # 松手即停


def test_heartbeat_does_not_reset_hold_clock():
    """心跳不重置按住计时，也不补发轻触步。"""
    state = KeyState()
    state.press("w", 0.0)
    assert state.active_key(0.0, HOLD_DELAY) == "w"   # 轻触那一步先走掉，下面看的是连续输出
    for i in range(1, 6):                     # 5 次心跳，都在门槛之前
        hb = i * 0.1
        state.press("w", hb, heartbeat=True)
        assert state.hold_elapsed("w", hb) == hb, hb
        assert state.active_key(hb, HOLD_DELAY) is None, hb   # 心跳不补步
    assert state.active_key(0.1 * 9, HOLD_DELAY) is None
    assert state.active_key(HOLD_DELAY, HOLD_DELAY) == "w"   # 起始时刻仍是第一次按下


def test_repress_after_release_restarts_the_clock():
    """真正松手（超过宽限期）之后再按下，计时从头开始，且重新拿到一次轻触步。"""
    state = KeyState()
    state.press("w", 0.0)
    assert state.active_key(0.0, HOLD_DELAY) == "w"
    state.release("w", 0.2)
    assert state.hold_elapsed("w", 0.2 + GRACE + 0.01) is None
    state.press("w", 2.0)
    assert state.active_key(2.0, HOLD_DELAY) == "w"                  # 新一次轻触
    assert state.active_key(2.0 + HOLD_DELAY * 0.5, HOLD_DELAY) is None
    assert state.active_key(2.0 + HOLD_DELAY, HOLD_DELAY) == "w"


def test_gapped_repeat_survives_grace_and_keeps_the_clock():
    """心跳与松手交错时，按住不断、计时也不回零。"""
    state = KeyState()
    state.press("w", 0.0)
    assert state.active_key(0.0, HOLD_DELAY) == "w"   # 轻触那一步先走掉
    now = 0.0
    for _ in range(200):                      # 0.4 s，还没到门槛
        state.press("w", now, heartbeat=True); now += 0.001
        state.release("w", now)
        now += 0.001
    assert state.hold_elapsed("w", now) == now      # 起始时刻仍是最早那次
    assert state.active_key(now, HOLD_DELAY) is None
    for _ in range(400):                      # 累计 1.2 s，越过门槛
        state.press("w", now, heartbeat=True); now += 0.001
        state.release("w", now)
        now += 0.001
    assert state.active_key(now, HOLD_DELAY) == "w"


def test_hold_elapsed_is_none_when_not_held():
    state = KeyState()
    assert state.hold_elapsed("w", 0.0) is None
    state.press("w", 0.0)
    state.release("w", 1.0)
    assert state.hold_elapsed("w", 1.0 + GRACE + 0.01) is None

def test_waiting_key_reports_pending_progress():
    """门槛未到且轻触步已发出时，waiting_key 报告进度。"""
    state = KeyState()
    state.press("w", 0.0)
    assert state.waiting_key(0.5, HOLD_DELAY) is None        # 轻触那一步还没出手
    assert state.active_key(0.0, HOLD_DELAY) == "w"
    assert state.waiting_key(0.5, HOLD_DELAY) == "w"
    assert state.waiting_key(HOLD_DELAY, HOLD_DELAY) is None
    assert state.waiting_key(0.5, 0.0) is None               # 门槛为 0 时没有等待态


def test_hold_delay_zero_is_the_old_behaviour():
    """``hold_delay=0`` 时按下即持续输出。"""
    state = KeyState()
    state.press("w", 0.0)
    assert state.active_key(0.0, 0.0) == "w"
    assert state.active_key(0.0) == "w"                      # 默认参数也是 0


# --------------------------------------------------------------------------- #
# 失焦清空（不处理的话 agent 会一直往前走）
# --------------------------------------------------------------------------- #

def test_clear_drops_everything():
    """clear 清空全部按键，之后仍可重新按下。"""
    state = KeyState()
    state.press("w", 0.0)
    state.press("a", 0.0)
    assert state.active_key(0.0) == "w"

    state.clear()
    assert state.active_key(0.0) is None
    assert state.held_keys(0.0) == []
    state.press("d", 1.0)
    assert state.active_key(1.0) == "d"


def test_debug_string_is_a_single_line():
    state = KeyState()
    state.press("w", 0.0)
    state.press("a", 0.0)
    text = state.debug(0.0)
    assert "\n" not in text
    assert "w" in text and "a" in text


# --------------------------------------------------------------------------- #
# runner（本仓库没装 pytest）
# --------------------------------------------------------------------------- #

def main():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed.append((name, exc))
            print(f"FAIL  {name}  {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"ERROR {name}  {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")

    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
