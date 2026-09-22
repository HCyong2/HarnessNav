"""Verify：侧移一步取第二视角，交给 VLM 判断是否同一物体。"""

import numpy as np

from nav.goto import STUCK_EPS_M, body_position, face_pano
from nav.occupancy import HAB_FORWARD, HAB_LEFT, HAB_RIGHT, to_rgb_uint8


def _xz(env):
    """机身水平位置。"""
    p = body_position(env)
    return np.array([float(p[0]), float(p[2])], dtype=np.float64)


def _run_acts(env, occ, acts):
    """依次 step；episode 结束则停。"""
    for act in acts:
        if env.episode_over:
            return
        env.step(act)
        occ.integrate_from_env(env)


def _try_move(env, occ, acts):
    """执行一组动作，位移超过阈值则成功。"""
    start = _xz(env)
    _run_acts(env, occ, acts)
    if env.episode_over:
        return False
    return float(np.linalg.norm(_xz(env) - start)) >= STUCK_EPS_M


def run_verify(env, occ, min_d, max_d, query, pano_id, on_log=None):
    """对准扇区后先左侧移，失败再前进/转头取第二张图。

    Args:
        env: ``habitat.Env``。
        occ: 占用图。
        min_d, max_d (float): 深度量程（兼容调用方，本流程不用）。
        query (str): 目标类名。
        pano_id (int): 可疑目标所在扇区。
        on_log (callable, optional): ``on_log(code, message)``。

    Returns:
        dict: 含两张图、``need_make_plan``。
    """
    del min_d, max_d

    def note(code, msg):
        if on_log is not None:
            on_log(code, msg)

    face_pano(env, pano_id)
    note("1", f"转向扇区 pano_id={pano_id}")
    obs = env.sim.get_sensor_observations()
    rgb0 = to_rgb_uint8(obs["rgb"])
    note("2", "视点0 已采集")

    how = None
    rgb1 = None
    if _try_move(env, occ, [HAB_LEFT, HAB_LEFT, HAB_FORWARD, HAB_RIGHT, HAB_RIGHT]):
        how = "side"
        note("3", "向左移一步（左转两次、前进、右转两次）")
        obs = env.sim.get_sensor_observations()
        rgb1 = to_rgb_uint8(obs["rgb"])
    else:
        note("3", "侧移未动，尝试前进一步")
        if _try_move(env, occ, [HAB_FORWARD]):
            how = "forward"
            obs = env.sim.get_sensor_observations()
            rgb1 = to_rgb_uint8(obs["rgb"])
        else:
            note("3", "前进未动，取左转视角")
            _run_acts(env, occ, [HAB_LEFT])
            if not env.episode_over:
                how = "turn"
                obs = env.sim.get_sensor_observations()
                rgb1 = to_rgb_uint8(obs["rgb"])

    if rgb1 is None:
        note("3", "无法取得第二视角，Verify 失败，应 MakePlan")
        return {
            "ok": True, "consistency": False, "need_make_plan": True,
            "side": "left", "how": None, "views": [rgb0], "compare_views": [rgb0],
            "instance_id": f"{query}_1",
        }

    note("4", f"视点1 how={how}")
    return {
        "ok": True,
        "consistency": False,
        "need_make_plan": False,
        "side": "left",
        "how": how,
        "views": [rgb0, rgb1],
        "compare_views": [rgb0, rgb1],
        "instance_id": f"{query}_1",
    }
