"""System 2 输入图：在当前第一视角上标分割实例或探索点。"""

import math

import cv2
import numpy as np

from perception.base import SegResult, mask_center_pixel, mean_depth_window, nms_seg_result
from perception.render import PALETTE, overlay_instances

from harness.protocol import round_sig
from nav.goto import face_pano, project_world_to_uv
from nav.occupancy import EVEN_PANO_INDICES, clean_depth, sensor_pose, to_rgb_uint8

OCCLUDED_M = 0.4
# 旧提取曾把失败路径写成 1000+欧氏；室内合法路径不会到这个量级。
UNREACHABLE_GEO_M = 1000.0
# 画幅外但相机前方的探索点夹到内边距，保证扇区边缘圆点可见。
FRONTIER_CLAMP_MARGIN = 12


def path_geodesic_ok(geo):
    """占用图路径长是否有限且不是失败哨兵。"""
    if geo is None:
        return False
    try:
        g = float(geo)
    except (TypeError, ValueError):
        return False
    return math.isfinite(g) and g < UNREACHABLE_GEO_M


def semantic_candidates(result, depth, query, max_n=3, iou_thresh=0.5):
    """GLEE 结果 -> 从左到右的 ``{query}_k`` 候选。

    先按 mask IoU 做 NMS，再最多保留 ``max_n`` 个（默认 3）。

    Args:
        result: ``SegResult``。
        depth (np.ndarray): ``(H, W)``，0 无效。
        query (str): 查询词。
        max_n (int): 最多几个。
        iou_thresh (float): NMS 重叠阈值。

    Returns:
        tuple: ``(candidates, kept_result)``。
    """
    result = nms_seg_result(result, iou_thresh=iou_thresh, max_n=max_n)
    items = []
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
        items.append((uv[0], i, {
            "id": None,
            "uv": [int(uv[0]), int(uv[1])],
            "depth_m": d,
            "score": float(result.scores[i]),
            "index": i,
        }))
    items.sort(key=lambda t: t[0])
    cands = []
    keep = []
    for k, (_, i, c) in enumerate(items[:max_n], start=1):
        c["id"] = f"{query}_{k}"
        cands.append({k2: c[k2] for k2 in ("id", "uv", "depth_m", "score")})
        keep.append(i)
    if not keep:
        return [], result.top(0)
    kept = SegResult(
        boxes=result.boxes[keep],
        masks=result.masks[keep],
        scores=result.scores[keep],
        labels=[c["id"] for c in cands],
        time_ms=result.time_ms,
    )
    return cands, kept


def frontier_candidates(frontiers, intrinsic, sensor_pos, sensor_rot, image_hw,
                        max_n=3, depth=None, drop_occluded=False, occ=None,
                        agent_xyz=None):
    """把探索点投到画面上，按深度升序编号 F1…。

    不再做占用图直线通视过滤；画幅外但相机前方的点夹到内边距后直接画。
    默认不因像素深度丢掉占用图点（门框像素会误杀门缝点）。
    ``occ`` / ``agent_xyz`` 保留兼容，当前不参与过滤。

    Args:
        frontiers (list): 探索点。
        intrinsic: 内参。
        sensor_pos: 相机位置。
        sensor_rot: 相机旋转。
        image_hw (tuple): ``(H, W)``。
        max_n (int): 最多几个。
        depth (np.ndarray, optional): 清洗后深度，0 无效。
        drop_occluded (bool): 为真时，像素比几何近出 0.4 米则丢弃。
        occ: 兼容保留，忽略。
        agent_xyz: 兼容保留，忽略。

    Returns:
        list: 带像素坐标的候选。
    """
    vis = []
    h, w = int(image_hw[0]), int(image_hw[1])
    depth_img = None if depth is None else np.asarray(depth)
    for fr in frontiers:
        xyz = fr["xyz"]
        proj = project_world_to_uv(
            xyz, intrinsic, sensor_pos, sensor_rot, image_hw,
            clamp_margin=FRONTIER_CLAMP_MARGIN)
        if proj is None:
            continue
        u, v, d = proj
        if drop_occluded and depth_img is not None and 0 <= v < h and 0 <= u < w:
            z_img = float(depth_img[v, u])
            if z_img > 0 and (float(d) - z_img) > OCCLUDED_M:
                continue
        vis.append({
            "id": None,
            "uv": [u, v],
            "depth_m": float(d),
            "score": 1.0 / max(float(d), 0.1),
            "xyz": fr["xyz"],
            "fid": fr.get("fid"),
            "geodesic_m": fr.get("geodesic_m"),
        })
    vis.sort(key=lambda c: c["depth_m"])
    out = []
    for k, c in enumerate(vis[:max_n], start=1):
        c["id"] = f"F{k}"
        out.append(c)
    return out


def pick_mover_candidate(mode, cands):
    """按规则从候选里选一个：探索取可走路径最远；物体取置信度为主、距离为辅。

    Args:
        mode (str): ``frontier`` 或 ``semantic``。
        cands (list): 候选。

    Returns:
        dict: 选中的候选；空列表或探索点均不可达则 None。
    """
    if not cands:
        return None
    if mode == "frontier":
        walkable = [c for c in cands if path_geodesic_ok(c.get("geodesic_m"))]
        if not walkable:
            return None
        return max(walkable, key=lambda c: float(c["geodesic_m"]))
    depths = [max(float(c.get("depth_m") or 0.0), 1e-6) for c in cands]
    dmax = max(depths)
    best, best_s, best_d = None, None, -1.0
    for c, d in zip(cands, depths):
        conf = float(c.get("score") or 0.0)
        score = 0.7 * conf + 0.3 * (d / dmax)
        if (best is None or score > best_s + 1e-9
                or (abs(score - best_s) <= 1e-9 and d > best_d)):
            best, best_s, best_d = c, score, d
    if best is not None:
        best = dict(best)
        best["pick_score"] = float(best_s)
    return best


def depth_text_map(candidates):
    """把候选深度写成 ``{F1: 2.5m, ...}`` 文本（3 位有效数字）。"""
    parts = []
    for c in candidates:
        cid = c.get("id")
        if not cid:
            continue
        d = round_sig(c.get("depth_m"))
        if d is None:
            continue
        parts.append(f"{cid}: {d}m")
    return "{" + ", ".join(parts) + "}"


def face_best_frontier_view(env, frontiers, intrinsic, node_yaw, max_n=3, occ=None):
    """转到 in_view 前沿最多的偶序号扇区。

    Args:
        env: ``habitat.Env``。
        frontiers (list): 全局前沿。
        intrinsic: 内参。
        node_yaw (float): ScanNode 时的 yaw。
        max_n (int): 最多候选。
        occ: 兼容保留，忽略。

    Returns:
        tuple: ``(rgb, candidates, pano_id)``。
    """
    best_cands, best_pid = [], None
    for pid in EVEN_PANO_INDICES:
        face_pano(env, pid, node_yaw=node_yaw)
        obs = env.sim.get_sensor_observations()
        rgb = to_rgb_uint8(obs["rgb"])
        depth = clean_depth(obs["depth"], 0.5, 5.0)
        pos, rot = sensor_pose(env)
        cands = frontier_candidates(
            frontiers, intrinsic, pos, rot, rgb.shape[:2], max_n=max_n,
            depth=depth, occ=occ, agent_xyz=pos)
        if len(cands) > len(best_cands):
            best_cands, best_pid = cands, pid
    if best_pid is None:
        rgb = to_rgb_uint8(env.sim.get_sensor_observations()["rgb"])
        return rgb, [], None
    face_pano(env, best_pid, node_yaw=node_yaw)
    obs = env.sim.get_sensor_observations()
    rgb = to_rgb_uint8(obs["rgb"])
    depth = clean_depth(obs["depth"], 0.5, 5.0)
    pos, rot = sensor_pose(env)
    cands = frontier_candidates(
        frontiers, intrinsic, pos, rot, rgb.shape[:2], max_n=max_n,
        depth=depth, occ=occ, agent_xyz=pos)
    return rgb, cands, best_pid


def draw_annotated(rgb, mode, candidates, seg_kept=None, write_depth=False):
    """画 ``Ego annotated``。

    Args:
        rgb: 当前第一视角。
        mode (str): ``semantic`` 或 ``frontier``。
        candidates (list): 候选。
        seg_kept: semantic 时的 ``SegResult``。
        write_depth (bool): 是否在图上写深度；探索模式默认不写。

    Returns:
        np.ndarray: 标注图。
    """
    canvas = to_rgb_uint8(rgb)
    if mode == "semantic" and seg_kept is not None and len(seg_kept) > 0:
        canvas = overlay_instances(canvas, seg_kept)
        for i, c in enumerate(candidates):
            u, v = c["uv"]
            if write_depth:
                text = f"{c['id']} {c['depth_m']:.2f}m"
            else:
                text = str(c["id"])
            cv2.putText(canvas, text, (u, max(18, v - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, PALETTE[i % len(PALETTE)], 2, cv2.LINE_AA)
        return canvas
    for i, c in enumerate(candidates):
        u, v = int(c["uv"][0]), int(c["uv"][1])
        color = (0, 220, 0)
        cv2.circle(canvas, (u, v), 8, color, 2, cv2.LINE_AA)
        if write_depth:
            label = f"{c['id']} {c['depth_m']:.2f}m"
        else:
            label = str(c["id"])
        cv2.putText(canvas, label, (u + 10, v),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas
