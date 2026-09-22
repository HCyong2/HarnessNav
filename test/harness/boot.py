"""测试脚本共用的环境启动。"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness import quiet_sim  # noqa: F401  须在 habitat 之前

import habitat
from habitat.config.read_write import read_write

from play2nav.config import check_split, play_config
from nav.goto import body_yaw_env, face_pano
from nav.occupancy import EVEN_PANO_INDICES, to_rgb_uint8
from perception.base import empty_result, nms_seg_result

# depth / mover 可视化优先用这些词，出生点朝向经常没有 chair。
GLEE_TEST_QUERIES = ("door", "bed", "window")


def boot_env(seed=5, gpu_id=0, episodes=1, stage="val"):
    """构造并 reset 一个 HM3D val env。

    Args:
        seed (int): 数据集种子。
        gpu_id (int): GPU。
        episodes (int): episode 数。
        stage (str): split。

    Returns:
        tuple: ``(env, config)``。
    """
    config, fixed, _ = play_config(stage=stage, episodes=episodes, seed=seed, gpu_id=gpu_id)
    problems = check_split(stage, config)
    if problems:
        raise SystemExit("数据路径对不上:\n  " + "\n  ".join(problems))
    with read_write(config):
        config.habitat.environment.iterator_options.shuffle = False
    env = habitat.Env(config)
    env.reset()
    return env, config


def maybe_glee(device="cuda:0", threshold=0.15):
    """尝试加载 GLEE；失败则返回 ``None``。

    Args:
        device (str): 设备。
        threshold (float): 分数阈值。

    Returns:
        SegBackend: 或 ``None``。
    """
    try:
        from perception.glee_backend import GLEEBackend
        return GLEEBackend(device=device, score_threshold=threshold, num_inst_select=8)
    except Exception as exc:
        print(f"GLEE 未加载: {exc}")
        return None


def find_glee_view(env, backend, queries, node_yaw=None):
    """转偶序号扇区，直到某个 query 分出实例。

    Args:
        env: ``habitat.Env``。
        backend: 分割后端。
        queries (sequence): 文本列表，按顺序试。
        node_yaw (float, optional): 扇区 0 的 yaw；缺省为当前朝向。

    Returns:
        tuple: ``(rgb, query, result, pano_id)``；找不到则为 rgb 仍当前帧、其余 None。
    """
    if node_yaw is None:
        node_yaw = body_yaw_env(env)
    queries = [q for q in queries if q]
    if backend is None:
        rgb = to_rgb_uint8(env.sim.get_sensor_observations()["rgb"])
        h, w = rgb.shape[:2]
        return rgb, None, empty_result(height=h, width=w), None

    best = None
    for pid in EVEN_PANO_INDICES:
        face_pano(env, pid, node_yaw=node_yaw)
        rgb = to_rgb_uint8(env.sim.get_sensor_observations()["rgb"])
        for query in queries:
            result = nms_seg_result(backend.segment(rgb, query), iou_thresh=0.5, max_n=3)
            if len(result) == 0:
                continue
            if best is None or float(result.scores[0]) > float(best[2].scores[0]):
                best = (rgb, query, result, pid)
    if best is None:
        rgb = to_rgb_uint8(env.sim.get_sensor_observations()["rgb"])
        h, w = rgb.shape[:2]
        return rgb, None, empty_result(height=h, width=w), None
    rgb, query, result, pid = best
    face_pano(env, pid, node_yaw=node_yaw)
    rgb = to_rgb_uint8(env.sim.get_sensor_observations()["rgb"])
    result = nms_seg_result(backend.segment(rgb, query), iou_thresh=0.5, max_n=3)
    return rgb, query, result, pid
