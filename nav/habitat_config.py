"""构建 habitat HM3D 环境配置。

数据路径、传感器参数、topdown 度量与碰撞度量都在这里设置。``HM3D_CONFIG_PATH`` 指向本
仓库 ``config/`` 下的 YAML 副本，因此不依赖 StateNav 的检出。
"""

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

HM3D_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "objectnav_hm3d_step_2000.yaml")


def hm3d_config(path: str = HM3D_CONFIG_PATH, stage: str = 'val', episodes=200, gpu_id=0):
    """构造 HM3D ObjectNav 的环境配置。

    Args:
        path (str): habitat 环境 YAML 的路径。
        stage (str): 数据集 split，如 ``val``。
        episodes (int): 采样多少个 episode。
        gpu_id (int): 显式指定 GPU，防止某些版本的 habitat 回跳到物理 GPU 0。

    Returns:
        omegaconf.DictConfig: 可直接传给 ``habitat.Env`` 的配置。
    """
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        # 显式指定 GPU，防止某些版本的 Habitat 回跳到物理 GPU 0
        habitat_config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id

        habitat_config.habitat.dataset.split = stage

        # 1. 3D 资产实体根目录
        habitat_config.habitat.dataset.scenes_dir = "/DATA_HDD/hc/dataset-3d/data/scene_datasets"
        # 2. 关卡任务文件路径
        habitat_config.habitat.dataset.data_path = "/DATA_HDD/hc/dataset-3d/data/datasets/objectnav_hm3d_v2/{split}/{split}.json.gz"
        # 3. 场景总索引 JSON 的绝对路径
        habitat_config.habitat.simulator.scene_dataset = f"/DATA_HDD/hc/dataset-3d/data/scene_datasets/hm3d_v0.2/hm3d_annotated_{stage}_basis.scene_dataset_config.json"

        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes

        habitat_config.habitat.task.measurements.update({
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
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25

    return habitat_config
