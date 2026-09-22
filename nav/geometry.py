"""深度图到点云的反投影，以及相机系到世界系的换算。

坐标系约定：``get_pointcloud_from_depth*`` 返回相机系点，按 ``[x, z, -y]`` 堆叠，即第 0
列是 x，第 1 列是 z（图像行方向取反，第 0 行对应最大的 z），第 2 列是 ``-depth``。
``translate_to_world`` 做的是普通的 ``R @ p + t``，Y/Z 轴的置换由调用方传入的
``rotation`` / ``position`` 承担（见 ``nav.transform``）。
"""

import cv2
import numpy as np


def preprocess_depth(depth: np.ndarray, lower_bound: float = 0.1, upper_bound: float = 4.9):
    """把 ``[lower_bound, upper_bound]`` 区间之外的深度值置零，单位米。

    Args:
        depth (np.ndarray): 深度图，单位为米。
        lower_bound (float): 区间下界，小于它的值置零。
        upper_bound (float): 区间上界，大于它的值置零。

    Returns:
        np.ndarray: 置零后的深度图，与输入是同一个数组（原地修改）；需要保留原值
        时请先自行 copy。
    """
    depth[np.where((depth < lower_bound) | (depth > upper_bound))] = 0
    return depth


def get_pointcloud_from_depth(rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray):
    """把每个有效深度像素反投影到相机系。

    Args:
        rgb (np.ndarray): 图像，``(H, W, 3)`` RGB 或 ``(H, W)`` 灰度。
        depth (np.ndarray): 深度图，``(H, W)`` 或 ``(H, W, 1)``，无效像素为 0。
        intrinsic (np.ndarray): ``3x3`` 相机内参。

    Returns:
        tuple: ``(point_values, color_values)``，两者形状均为 ``(N, 3)``。
    """
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    if len(rgb.shape) == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    elif rgb.shape[-1] == 4:
        # RGBA 只取前三个通道
        rgb = rgb[:, :, :3]
    filter_z, filter_x = np.where(depth > 0)
    depth_values = depth[filter_z, filter_x]
    pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2]) * depth_values / intrinsic[0][0]
    pixel_y = depth_values
    color_values = rgb[filter_z, filter_x]
    point_values = np.stack([pixel_x, pixel_z, -pixel_y], axis=-1)
    return point_values, color_values


def get_pointcloud_from_depth_mask(depth: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray):
    """同 ``get_pointcloud_from_depth``，但只取 ``mask > 0`` 的像素，且只返回点。

    Args:
        depth (np.ndarray): 深度图，``(H, W)`` 或 ``(H, W, 1)``，无效像素为 0。
        mask (np.ndarray): 布尔掩码，``(H, W)``。
        intrinsic (np.ndarray): ``3x3`` 相机内参。

    Returns:
        np.ndarray: 相机系点，``(N, 3)``。
    """
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    filter_z, filter_x = np.where((depth > 0) & (mask > 0))
    depth_values = depth[filter_z, filter_x]
    pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2]) * depth_values / intrinsic[0][0]
    pixel_y = depth_values
    point_values = np.stack([pixel_x, pixel_z, -pixel_y], axis=-1)
    return point_values


def translate_to_world(pointcloud, position, rotation):
    """把相机系点按外参 ``[R | t]`` 变换到世界系。

    Args:
        pointcloud (np.ndarray): 相机系点，``(N, 3)``。
        position (np.ndarray): 平移量 ``t``。
        rotation (np.ndarray): 旋转矩阵 ``R``，``3x3``。

    Returns:
        np.ndarray: 世界系点，``(N, 3)``。
    """
    extrinsic = np.eye(4)
    extrinsic[0:3, 0:3] = rotation
    extrinsic[0:3, 3] = position
    world_points = np.matmul(
        extrinsic, np.concatenate((pointcloud, np.ones((pointcloud.shape[0], 1))), axis=-1).T
    ).T
    return world_points[:, 0:3]
