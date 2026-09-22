"""Look Skill：原地一步转/俯仰。"""

from nav.occupancy import HAB_LEFT, HAB_LOOK_DOWN, HAB_LOOK_UP, HAB_RIGHT

LOOK_TO_ACTION = {
    "up": HAB_LOOK_UP,
    "down": HAB_LOOK_DOWN,
    "left": HAB_LEFT,
    "right": HAB_RIGHT,
}


def run_look(env, action, pitch_steps, pitch_limit=2):
    """执行一步 Look。俯仰超出 ``pitch_limit`` 则拒绝。

    Args:
        env: ``habitat.Env``。
        action (str): ``up|down|left|right``。
        pitch_steps (int): 当前相对平视的低头步数（正为 down）。
        pitch_limit (int): 允许的最大 |pitch| 步数。

    Returns:
        tuple: ``(result_dict, new_pitch_steps)``。
    """
    if env.episode_over:
        return {"ok": False, "error": "episode_over"}, pitch_steps
    if action not in LOOK_TO_ACTION:
        return {"ok": False, "error": "bad_action"}, pitch_steps
    if action == "down" and pitch_steps >= pitch_limit:
        return {"ok": False, "error": "pitch_limit", "action": action}, pitch_steps
    if action == "up" and pitch_steps <= -pitch_limit:
        return {"ok": False, "error": "pitch_limit", "action": action}, pitch_steps
    env.step(LOOK_TO_ACTION[action])
    if action == "down":
        pitch_steps += 1
    elif action == "up":
        pitch_steps -= 1
    return {"ok": True, "action": action, "image_label": f"Look {action}"}, pitch_steps


def restore_pitch(env, pitch_steps):
    """把俯仰抬回平视。

    Args:
        env: ``habitat.Env``。
        pitch_steps (int): 当前低头步数。

    Returns:
        int: 恒为 0。
    """
    while pitch_steps > 0 and not env.episode_over:
        env.step(HAB_LOOK_UP)
        pitch_steps -= 1
    while pitch_steps < 0 and not env.episode_over:
        env.step(HAB_LOOK_DOWN)
        pitch_steps += 1
    return 0
