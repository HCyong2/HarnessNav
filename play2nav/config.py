"""在 ``hm3d_config`` 之上覆盖 play2nav 需要的交互项。"""

import os

from habitat.config.read_write import read_write

from nav.habitat_config import hm3d_config

_SCENE_ROOT = "/DATA_HDD/hc/dataset-3d/data/scene_datasets"
_HM3D_DIR = os.path.join(_SCENE_ROOT, "hm3d_v0.2")

# HM3D ObjectNav 标准成功距离。
DEFAULT_SUCCESS_DISTANCE = 1.0


def _scene_dataset_config(stage: str) -> str:
    """返回该 split 下实际存在的 scene_dataset 索引路径。

    Args:
        stage (str): 数据集 split，如 ``val``。

    Returns:
        str: 存在的配置文件路径；找不到则空串。
    """
    candidates = [
        os.path.join(_HM3D_DIR, stage, f"hm3d_annotated_{stage}_basis.scene_dataset_config.json"),
        os.path.join(_HM3D_DIR, stage, f"hm3d_{stage}_basis.scene_dataset_config.json"),
        os.path.join(_SCENE_ROOT, f"hm3d_v0.2/{stage}", "hm3d_annotated_basis.scene_dataset_config.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def play_config(stage="val", episodes=10, seed=None, gpu_id=0,
                success_distance=DEFAULT_SUCCESS_DISTANCE,
                draw_shortest_path=False, shuffle=False):
    """构造交互界面用的 habitat 配置。

    Args:
        stage (str): 数据集 split。
        episodes (int): 会话里预备的 episode 数量。
        seed (int, optional): 数据集随机种子。
        gpu_id (int): 模拟器 GPU。
        success_distance (float): 成功距离，米。
        draw_shortest_path (bool): 是否让 habitat 自己画最短路径。
        shuffle (bool): 是否打乱 episode 顺序。

    Returns:
        tuple: ``(config, 修正后的 scene_dataset 路径或空串, YAML 原 success_distance)``。
    """
    config = hm3d_config(stage=stage, episodes=episodes, gpu_id=gpu_id)
    yaml_success = _read_success_distance(config)

    with read_write(config):
        if seed is not None:
            config.habitat.seed = seed

        config.habitat.environment.iterator_options.shuffle = bool(shuffle)
        config.habitat.task.measurements.success.success_distance = float(success_distance)

        if "top_down_map" in config.habitat.task.measurements:
            config.habitat.task.measurements.top_down_map.draw_shortest_path = bool(
                draw_shortest_path
            )

        fixed = _scene_dataset_config(stage)
        if fixed:
            config.habitat.simulator.scene_dataset = fixed

    return config, fixed, yaml_success


def _read_success_distance(config):
    """读取配置里当前的 success_distance。

    Args:
        config: habitat 配置。

    Returns:
        float: 成功距离；读不到则为 nan。
    """
    try:
        return float(config.habitat.task.measurements.success.success_distance)
    except Exception:
        return float("nan")


def check_split(stage, config):
    """检查本机该 split 的数据路径是否齐全。

    Args:
        stage (str): 数据集 split。
        config: habitat 配置。

    Returns:
        list: 问题描述；空列表表示可以跑。
    """
    problems = []
    scene_dataset = str(config.habitat.simulator.scene_dataset)
    if not os.path.exists(scene_dataset):
        problems.append(f"场景索引不存在: {scene_dataset}")

    data_path = str(config.habitat.dataset.data_path).format(split=stage)
    if not os.path.exists(data_path):
        problems.append(f"episode 数据不存在: {data_path}")

    scene_root = os.path.join(str(config.habitat.dataset.scenes_dir), "hm3d_v0.2", stage)
    if not os.path.isdir(scene_root):
        problems.append(
            f"场景目录不存在: {scene_root}\n"
            f"      （episode 里的 scene_id 形如 "
            f"'hm3d_v0.2/{stage}/<scene>/<scene>.basis.glb'，需要在 scenes_dir 下能对上）")

    return problems


def summary_lines(config, stage, episodes, success_distance, fixed_scene_dataset, yaml_success):
    """生成启动时打印的配置摘要。

    Args:
        config: habitat 配置。
        stage (str): 数据集 split。
        episodes (int): 预备集数。
        success_distance (float): 实际使用的成功距离。
        fixed_scene_dataset (str): 修正后的 scene_dataset 路径。
        yaml_success (float): YAML 里原来的成功距离。

    Returns:
        list: 可供逐行打印的字符串。
    """
    sim = config.habitat.simulator
    rgb = sim.agents.main_agent.sim_sensors.rgb_sensor
    shuffle = bool(config.habitat.environment.iterator_options.shuffle)
    return [
        f"stage      : {stage}  (num_episode_sample={episodes})",
        f"iterator   : shuffle={shuffle}"
        f"{'  (数据集顺序，连着几集常是同一场景)' if shuffle else '  (「进入下一个场景」确定性推进)'}",
        f"scene data : {config.habitat.dataset.scenes_dir}",
        f"episodes   : {config.habitat.dataset.data_path}",
        f"scene index: {config.habitat.simulator.scene_dataset}",
        f"             {'(已修正为存在的路径)' if fixed_scene_dataset else '(沿用 hm3d_config 原值)'}",
        f"rgb sensor : {rgb.width}x{rgb.height}  hfov={rgb.hfov}",
        f"turn_angle : {sim.turn_angle} deg",
        f"success    : within {success_distance:.2f} m "
        f"(HM3D ObjectNav standard; YAML 里是 {yaml_success:.2f})",
    ]
