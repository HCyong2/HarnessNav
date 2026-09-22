"""Depth Skill：分割反投影后返回占用图测地距离。"""

import math

import numpy as np

from perception.base import mask_center_pixel, mean_depth_window, segment_relax

from nav.goto import foothold_from_hit, occupancy_path_length, pixel_to_world
from nav.occupancy import body_position, clean_depth, sensor_pose, to_rgb_uint8


def run_depth(env, backend, query, instance_id, min_depth, max_depth,
              rgb=None, depth=None, occ=None, intrinsic=None):
    """在当前（或给定）观测上量实例占用图路径距离。

    深度图仅用于反投影；主读数为 ``geodesic_m``（机身到落脚点的占用图 A* 长）。

    Args:
        env: ``habitat.Env``。
        backend: 分割后端，可为 ``None``。
        query (str): 物体词。
        instance_id (str): 指定实例时只填该条。
        min_depth (float): 传感器近端。
        max_depth (float): 传感器远端。
        rgb: 可选 RGB。
        depth: 可选原始深度。
        occ: ``OccupancyMap``；缺省则不填测地。
        intrinsic: 相机内参；与 ``occ`` 一起用于反投影。

    Returns:
        dict: Skill 输出。
    """
    obs = env.sim.get_sensor_observations()
    if rgb is None:
        rgb = to_rgb_uint8(obs["rgb"])
    if depth is None:
        depth = obs["depth"]
    depth = clean_depth(depth, min_depth, max_depth)
    if backend is None:
        return {"ok": False, "error": "no_backend", "instances": []}
    result, _thr = segment_relax(backend, rgb, query)
    pos, rot = sensor_pose(env)
    body = body_position(env)
    instances = []
    for i in range(len(result)):
        uv = mask_center_pixel(result.masks[i], depth)
        if uv is None:
            continue
        d = mean_depth_window(depth, uv)
        if d is None:
            valid = result.masks[i] & (depth > 0)
            if not np.any(valid):
                continue
            d = float(np.median(depth[valid]))
        inst_id = f"{query}_{i + 1}"
        if instance_id is not None and inst_id != instance_id:
            continue
        geo = None
        foothold = None
        if occ is not None and intrinsic is not None:
            world = pixel_to_world(uv[0], uv[1], depth, intrinsic, pos, rot)
            if world is not None:
                foothold = foothold_from_hit(occ, pos, world, body)
                if foothold is not None:
                    length = occupancy_path_length(occ, body, foothold,
                                                   success_dist=0.35)
                    if math.isfinite(length):
                        geo = float(length)
        item = {
            "id": inst_id,
            "uv": [int(uv[0]), int(uv[1])],
            "depth_m": d,
            "geodesic_m": geo,
            "score": float(result.scores[i]),
        }
        if foothold is not None:
            item["foothold_xyz"] = np.asarray(foothold, dtype=np.float64).reshape(3).tolist()
        instances.append(item)
    return {"ok": True, "instances": instances}
