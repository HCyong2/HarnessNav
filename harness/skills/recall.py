"""Recall Skill：取回旧节点全景，不走路。"""

import os

import cv2
import numpy as np

from nav.occupancy import to_rgb_uint8


def run_recall(graph, node_id, pano_id=None, pano_ids=None, query=""):
    """读节点存盘 RGB。

    Args:
        graph: ``NodeGraph``。
        node_id (int): 节点。
        pano_id (int, optional): 单个朝向。
        pano_ids (list, optional): 多个朝向。
        query (str): 查询文本，原样回传。

    Returns:
        dict: Skill 输出，含 ``images`` 列表（RGB）。
    """
    node = graph.nodes.get(int(node_id))
    if node is None:
        return {"ok": False, "error": "unknown_node"}
    ids = []
    if pano_ids:
        ids = [int(x) for x in pano_ids]
    elif pano_id is not None:
        ids = [int(pano_id)]
    else:
        last = node.get("last_plan") or {}
        ids = [int(last.get("pano_id") or 0)]
    images = []
    labels = []
    for pid in ids:
        meta = (node.get("pano") or {}).get(str(pid), {})
        rgb = meta.get("rgb")
        if rgb is not None:
            rgb = to_rgb_uint8(rgb)
        else:
            path = meta.get("rgb_path")
            rgb = None
            if path and os.path.isfile(path):
                bgr = cv2.imread(path)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb = to_rgb_uint8(rgb)
        images.append(rgb)
        labels.append(f"Recall node={node_id} dir={pid}")
    public = {k: node[k] for k in ("node_id", "xyz", "yaw", "visit_count",
                                    "summary", "leftover", "last_plan", "views",
                                    "explored_dirs")
              if k in node}
    return {
        "ok": True,
        "node_id": int(node_id),
        "query": query,
        "image_labels": labels,
        "images": images,
        "node_public": public,
    }
