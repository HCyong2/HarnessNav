"""HarnessNav 的 Habitat 配置：读 YAML、补数据路径、预检 split。"""

import logging
import os

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("HABITAT_LAB_LOG", str(logging.ERROR))

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.config.read_write import read_write

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_PATH = os.path.join(REPO_ROOT, "config", "harness_hm3d.yaml")

SCENE_ROOT = "/DATA_HDD/hc/dataset-3d/data/scene_datasets"
HM3D_DIR = os.path.join(SCENE_ROOT, "hm3d_v0.2")
EPISODE_PATH = (
    "/DATA_HDD/hc/dataset-3d/data/datasets/objectnav_hm3d_v2/{split}/{split}.json.gz"
)
DEFAULT_SUCCESS_DISTANCE = 1.0


def _scene_dataset_config(stage):
    """该 split 下实际存在的 scene_dataset 索引。

    Args:
        stage (str): 数据集 split。

    Returns:
        str: 存在的路径；找不到则空串。
    """
    candidates = [
        os.path.join(HM3D_DIR, stage, f"hm3d_annotated_{stage}_basis.scene_dataset_config.json"),
        os.path.join(HM3D_DIR, stage, f"hm3d_{stage}_basis.scene_dataset_config.json"),
        os.path.join(HM3D_DIR, f"hm3d_annotated_{stage}_basis.scene_dataset_config.json"),
        os.path.join(SCENE_ROOT, f"hm3d_v0.2/{stage}", "hm3d_annotated_basis.scene_dataset_config.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def check_split(stage, config):
    """构造 Env 前检查本机数据路径。

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
            f"      （episode 的 scene_id 形如 "
            f"'hm3d_v0.2/{stage}/<scene>/<scene>.basis.glb'）"
        )
    return problems


def load_config(stage="val", episodes=10, seed=None, gpu_id=0,
                success_distance=DEFAULT_SUCCESS_DISTANCE, yaml_path=YAML_PATH):
    """构造评测用 ``habitat.Env`` 配置。

    Args:
        stage (str): 数据集 split。本机通常只有 ``val`` 路径齐全。
        episodes (int): 采样集数。
        seed (int, optional): 数据集种子。
        gpu_id (int): 模拟器 GPU。
        success_distance (float): 成功距离，米。
        yaml_path (str): Habitat YAML。

    Returns:
        tuple: ``(config, 实际 scene_dataset 路径, YAML 里的 success_distance)``。
    """
    config = habitat.get_config(yaml_path)
    try:
        yaml_success = float(config.habitat.task.measurements.success.success_distance)
    except Exception:
        yaml_success = float("nan")

    with read_write(config):
        if seed is not None:
            config.habitat.seed = seed
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(gpu_id)
        config.habitat.dataset.split = stage
        config.habitat.dataset.scenes_dir = SCENE_ROOT
        config.habitat.dataset.data_path = EPISODE_PATH
        config.habitat.environment.iterator_options.num_episode_sample = int(episodes)
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.task.measurements.success.success_distance = float(success_distance)
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 5.0
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
        config.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=False,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=90,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        fixed = _scene_dataset_config(stage)
        if fixed:
            config.habitat.simulator.scene_dataset = fixed
        else:
            config.habitat.simulator.scene_dataset = os.path.join(
                HM3D_DIR, f"hm3d_annotated_{stage}_basis.scene_dataset_config.json"
            )

    return config, fixed, yaml_success
