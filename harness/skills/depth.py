"""Depth Skill：GLEE 实例中心深度，不把深度图送进上下文。"""

import numpy as np

from perception.base import mask_center_pixel, mean_depth_window, segment_relax

from nav.occupancy import clean_depth, to_rgb_uint8


def run_depth(env, backend, query, instance_id, min_depth, max_depth, rgb=None, depth=None):
    """在当前（或给定）观测上量实例深度。

    深度取 mask 中心周围 9×9 有效像素均值。

    Args:
        env: ``habitat.Env``。
        backend: 分割后端，可为 ``None``。
        query (str): 物体词。
        instance_id (str): 指定实例时只填该条。
        min_depth (float): 传感器近端。
        max_depth (float): 传感器远端。
        rgb: 可选 RGB。
        depth: 可选原始深度。

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
        instances.append({
            "id": inst_id,
            "uv": [int(uv[0]), int(uv[1])],
            "depth_m": d,
            "score": float(result.scores[i]),
        })
    return {"ok": True, "instances": instances}
