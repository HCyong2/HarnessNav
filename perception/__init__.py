"""文本驱动的分割后端。

两个后端都实现 ``base.SegBackend`` 接口，下游的 2D->3D 反投影不关心 mask 由谁产生：

* ``glee`` —— 单模型开放词表，文本类别直接出 box + mask
* ``gdino_sam`` —— GroundingDINO 出 box，SAM 出 mask
"""

from .base import (
    DEPTH_WINDOW, SEG_SCORE_TRIES, SegResult, SegBackend, mask_center_pixel,
    mean_depth_window, empty_result, filter_by_score, nms_seg_result, segment_relax,
)

__all__ = [
    "DEPTH_WINDOW", "SEG_SCORE_TRIES", "SegResult", "SegBackend", "mask_center_pixel",
    "mean_depth_window", "empty_result", "filter_by_score", "nms_seg_result",
    "segment_relax",
]
