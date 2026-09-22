"""habitat 相机内参，以及 habitat 世界系与 mapper 局部系之间的位姿换算。

Y/Z 轴置换都集中在这里：``habitat_translation`` 把 habitat 世界坐标 ``(x, y, z)`` 换成
mapper 局部系的 ``(x, z, y)``；``habitat_rotation`` 在四元数的旋转矩阵左侧乘上 Y/Z 置换
矩阵，使结果可以直接喂给 ``geometry.translate_to_world``。
"""

import numpy as np
import quaternion


def habitat_camera_intrinsic(config):
    """从深度传感器的 width / height / hfov 构造针孔相机内参。

    会断言 rgb 与 depth 三项一致——两者共用同一个内参矩阵做反投影，必须一致。

    Returns:
        np.ndarray: ``3x3`` 内参矩阵，``float32``。
    """
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width, 'The configuration of the depth camera should be the same as rgb camera.'
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height, 'The configuration of the depth camera should be the same as rgb camera.'
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov, 'The configuration of the depth camera should be the same as rgb camera.'
    width = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width
    height = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height
    hfov = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
    intrinsic_matrix = np.array([[f, 0, xc],
                                 [0, f, zc],
                                 [0, 0, 1]], np.float32)
    return intrinsic_matrix


def habitat_translation(position):
    """habitat 世界坐标 ``(x, y, z)`` -> mapper 局部系 ``(x, z, y)``。"""
    return np.array([position[0], position[2], position[1]])


def habitat_rotation(rotation):
    """habitat 旋转四元数 -> mapper 局部系的旋转矩阵。"""
    rotation_matrix = quaternion.as_rotation_matrix(rotation)
    transform_matrix = np.array([[1, 0, 0],
                                 [0, 0, 1],
                                 [0, 1, 0]])
    rotation_matrix = np.matmul(transform_matrix, rotation_matrix)
    return rotation_matrix


def as_quaternion(rotation):
    """把仿真器给出的旋转统一成 ``quaternion.quaternion``。

    habitat-sim 的 ``AgentState.rotation`` 是 ``[x, y, z, w]`` 数组，而 ``quaternion``
    库按 ``[w, x, y, z]`` 解释，混用会静默得到错误的旋转矩阵；已经是
    ``quaternion.quaternion`` 的按原样返回。

    Args:
        rotation: ``[x, y, z, w]`` 数组或 ``quaternion.quaternion``。

    Returns:
        quaternion.quaternion: 归一化后的旋转。
    """
    if hasattr(rotation, "w") and hasattr(rotation, "x"):
        return rotation
    q = np.asarray(rotation, dtype=np.float64).reshape(4)      # [x, y, z, w]
    return quaternion.from_float_array(np.array([q[3], q[0], q[1], q[2]]))


def mapper_local_to_world(local_pos, initial_position):
    """把 mapper 局部系 ``(x, z, y)`` 转回 habitat 世界 ``(x, y, z)``。

    Args:
        local_pos: 局部系坐标，``(3,)`` 或 ``(N, 3)``。
        initial_position: 起始位姿，必须与 ``local_pos`` 同为 mapper 局部
            ``(x, z, y)`` 顺序，即 ``habitat_translation(origin_world)``。置换只作用
            于偏移量、不作用于原点，传 habitat 世界顺序的原点会把原点的 y/z 一起
            换掉，得到一个错误但不报错的点。

    Returns:
        np.ndarray: habitat 世界坐标，形状与 ``local_pos`` 对应。
    """
    local_pos = np.asarray(local_pos)
    initial_position = np.asarray(initial_position)

    if local_pos.ndim == 1:
        # 单个点
        return np.array([
            local_pos[0] + initial_position[0],  # x
            local_pos[2] + initial_position[2],  # y (note: local y -> world y)
            local_pos[1] + initial_position[1],  # z (note: local z -> world z)
        ], dtype=np.float64)
    else:
        # 多个点 [N, 3]
        return np.array([
            local_pos[:, 0] + initial_position[0],  # x
            local_pos[:, 2] + initial_position[2],  # y
            local_pos[:, 1] + initial_position[1],  # z
        ], dtype=np.float64).T
