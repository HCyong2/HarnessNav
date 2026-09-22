"""文本驱动分割后端的公共接口。

各后端接收一张 RGB uint8 图像和一个自由文本提示，返回逐实例的 box、mask 与分数。本包
统一约定：

* 图像为 **RGB** ``np.uint8``，形状 ``(H, W, 3)``
* box 为 **xyxy** 像素坐标
* mask 为布尔 ``(H, W)``，与图像同分辨率
* ``mask_center_pixel`` 是后续反投影到 3D 的那个点
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SegResult:
    """后端返回的实例集合，按分数降序排列。"""

    boxes: np.ndarray                      # (N, 4) float32，xyxy 像素
    masks: np.ndarray                      # (N, H, W) bool
    scores: np.ndarray                     # (N,) float32
    labels: List[str] = field(default_factory=list)   # 长度 N
    time_ms: float = 0.0                   # 推理耗时（毫秒）

    def __len__(self) -> int:
        """实例个数。"""
        return int(self.boxes.shape[0])

    def top(self, k: int = 1) -> "SegResult":
        """返回只保留分数最高的 ``k`` 个实例的副本。"""
        k = min(k, len(self))
        return SegResult(
            boxes=self.boxes[:k],
            masks=self.masks[:k],
            scores=self.scores[:k],
            labels=self.labels[:k],
            time_ms=self.time_ms,
        )


def empty_result(time_ms: float = 0.0, height: int = 0, width: int = 0) -> SegResult:
    """构造零实例的结果，各数组形状正确。"""
    return SegResult(
        boxes=np.zeros((0, 4), dtype=np.float32),
        masks=np.zeros((0, height, width), dtype=bool),
        scores=np.zeros((0,), dtype=np.float32),
        labels=[],
        time_ms=time_ms,
    )


def mask_center_pixel(mask: np.ndarray, depth: Optional[np.ndarray] = None):
    """选出代表该实例的像素，用于反投影到 3D。

    默认取 mask 质心；给定深度图（0 表示无效）时，质心若落在无效深度上则退化为 mask
    内距质心最近的有效像素——空心的或被遮挡的 mask，质心常常落在背景上。

    Args:
        mask (np.ndarray): 布尔掩码，``(H, W)``。
        depth (np.ndarray, optional): 深度图，0 表示无效。

    Returns:
        tuple: 整数像素坐标 ``(x, y)``；mask 为空或无有效深度时返回 ``None``。
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    cx, cy = float(xs.mean()), float(ys.mean())

    if depth is None:
        return int(round(cx)), int(round(cy))

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    valid = depth[ys, xs] > 0
    if not valid.any():
        return None

    # 常见情况：质心本身就落在有效深度上。
    icy, icx = int(round(cy)), int(round(cx))
    if 0 <= icy < depth.shape[0] and 0 <= icx < depth.shape[1] and depth[icy, icx] > 0:
        return icx, icy

    # 否则取 mask 内距质心最近的有效像素。
    vxs, vys = xs[valid], ys[valid]
    best = np.argmin((vxs - cx) ** 2 + (vys - cy) ** 2)
    return int(vxs[best]), int(vys[best])


DEPTH_WINDOW = 9


def mean_depth_window(depth, uv, size=DEPTH_WINDOW):
    """选中像素周围窗口内有效深度的均值。

    Args:
        depth (np.ndarray): 深度图，0 表示无效；可为 ``(H, W)`` 或 ``(H, W, 1)``。
        uv (tuple): 中心像素 ``(x, y)``。
        size (int): 窗口边长，默认 9。

    Returns:
        float: 有效像素均值；窗口内全无效为 ``None``。
    """
    img = np.asarray(depth)
    if img.ndim == 3:
        img = img[:, :, 0]
    u, v = int(uv[0]), int(uv[1])
    half = max(int(size), 1) // 2
    h, w = img.shape[:2]
    y0 = max(0, v - half)
    y1 = min(h, v + half + 1)
    x0 = max(0, u - half)
    x1 = min(w, u + half + 1)
    patch = img[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(valid.mean())


def _mask_iou(a, b):
    """两个布尔 mask 的 IoU。"""
    inter = float(np.logical_and(a, b).sum())
    if inter <= 0:
        return 0.0
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _mask_cover(a, b):
    """较小 mask 被较大 mask 盖住的比例 ``inter / min(area)``。"""
    inter = float(np.logical_and(a, b).sum())
    if inter <= 0:
        return 0.0
    smaller = min(float(a.sum()), float(b.sum()))
    return inter / smaller if smaller > 0 else 0.0


SEG_SCORE_TRIES = (0.5, 0.4, 0.3)


def filter_by_score(result, score_threshold):
    """只保留分数严格高于阈值的实例。

    Args:
        result (SegResult): 分割结果。
        score_threshold (float): 分数下限。

    Returns:
        SegResult: 过滤后的副本；可能为空。
    """
    if len(result) == 0:
        return result
    keep = np.asarray(result.scores, dtype=np.float64) > float(score_threshold)
    if not np.any(keep):
        h = int(result.masks.shape[1]) if result.masks.ndim == 3 else 0
        w = int(result.masks.shape[2]) if result.masks.ndim == 3 else 0
        return empty_result(result.time_ms, h, w)
    idx = np.flatnonzero(keep)
    labels = [result.labels[i] for i in idx] if result.labels else []
    return SegResult(
        boxes=result.boxes[idx],
        masks=result.masks[idx],
        scores=result.scores[idx],
        labels=labels,
        time_ms=result.time_ms,
    )


def segment_relax(backend, image_rgb, text, thresholds=SEG_SCORE_TRIES,
                  iou_thresh=0.5, max_n=3):
    """一次分割，从高到低最多 3 档阈值，直到 NMS 后仍有实例。

    推理按最低档跑，再依次用 0.5 → 0.4 → 0.3 过滤，避免重复前向。

    Args:
        backend: ``SegBackend``。
        image_rgb (np.ndarray): RGB uint8。
        text (str): 提示词。
        thresholds (tuple): 从高到低的分数阈值。
        iou_thresh (float): NMS IoU。
        max_n (int): NMS 最多保留。

    Returns:
        tuple: ``(SegResult, 实际采用的阈值)``。
    """
    floor = float(min(thresholds))
    raw = backend.segment(image_rgb, text, score_threshold=floor)
    last = nms_seg_result(filter_by_score(raw, floor), iou_thresh=iou_thresh, max_n=max_n)
    used = floor
    for thr in thresholds:
        kept = nms_seg_result(
            filter_by_score(raw, thr), iou_thresh=iou_thresh, max_n=max_n)
        if len(kept) > 0:
            return kept, float(thr)
        last = kept
        used = float(thr)
    return last, used


def nms_seg_result(result, iou_thresh=0.5, max_n=3, cover_thresh=0.5):
    """按分数做 mask NMS，同类最多保留 ``max_n`` 个。

    IoU 或较小框被覆盖比例超阈值，都视为同一物体（大墙 mask 包住门时 IoU 往往 < 0.5）。

    Args:
        result (SegResult): 分割结果。
        iou_thresh (float): mask IoU 超过则视为同一物体。
        max_n (int): 最多保留个数。
        cover_thresh (float): 较小 mask 被覆盖超过则抑制。

    Returns:
        SegResult: 过滤后的副本。
    """
    n = len(result)
    if n == 0:
        return result
    order = np.argsort(-np.asarray(result.scores, dtype=np.float64))
    keep = []
    for i in order:
        i = int(i)
        dup = False
        for j in keep:
            if (_mask_iou(result.masks[i], result.masks[j]) >= iou_thresh
                    or _mask_cover(result.masks[i], result.masks[j]) >= cover_thresh):
                dup = True
                break
        if dup:
            continue
        keep.append(i)
        if len(keep) >= max_n:
            break
    keep = np.asarray(keep, dtype=int)
    labels = [result.labels[k] for k in keep] if result.labels else []
    return SegResult(
        boxes=result.boxes[keep],
        masks=result.masks[keep],
        scores=result.scores[keep],
        labels=labels,
        time_ms=result.time_ms,
    )


class SegBackend(ABC):
    """各后端共同实现的接口。"""

    name: str = "base"

    @abstractmethod
    def segment(self, image_rgb: np.ndarray, text: str, score_threshold=None) -> SegResult:
        """在 RGB uint8 图像中分割出与 ``text`` 匹配的实例。"""

    def close(self) -> None:  # pragma: no cover - optional hook
        """释放显存。无状态的后端可以不实现。"""
