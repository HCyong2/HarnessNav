"""GLEE 后端：单模型开放词表，文本直接出 box + mask。

GLEE 按类别**列表**训练，给定一个类别名就能返回对应实例，所以 ``"chair"`` 这种单词提示
等价于单元素列表；传逗号分隔的提示（``"chair, table"``）可一次查询多个类别。

输入归一化与参考实现保持一致，注意用的是 **RGB** 通道序（detectron2 默认是 BGR）。
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from .base import SegBackend, SegResult, empty_result

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLEE_ROOT = os.path.join(REPO_ROOT, "thirdparty", "GLEE")

DEFAULT_CONFIG = os.path.join(GLEE_ROOT, "configs", "SwinL.yaml")
DEFAULT_CKPT = os.path.join(REPO_ROOT, "model", "GLEE_SwinL_Scaleup10m.pth")
DEFAULT_CLIP = os.path.join(REPO_ROOT, "model", "clip-vit-base-patch32")

# GLEE 不在 site-packages 里，得先把它的根目录加进 sys.path 才能 import glee.*。
if GLEE_ROOT not in sys.path:
    sys.path.insert(0, GLEE_ROOT)

from detectron2.config import get_cfg  # noqa: E402
from glee.config import add_glee_config  # noqa: E402
from glee.config_deeplab import add_deeplab_config  # noqa: E402
from glee.models.glee_model import GLEE_Model  # noqa: E402


def parse_categories(text: str):
    """把逗号分隔的提示拆成 GLEE 需要的类别列表。

    Args:
        text (str): 如 ``"chair"`` 或 ``"chair, table"``。

    Returns:
        list: 类别名列表，至少含一项。
    """
    categories = [c.strip() for c in text.split(",") if c.strip()]
    return categories or [text.strip()]


class GLEEBackend(SegBackend):
    name = "glee"

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG,
        checkpoint_path: str = DEFAULT_CKPT,
        clip_path: str = DEFAULT_CLIP,
        device: str = "cuda:0",
        num_inst_select: int = 15,
        score_threshold: float = 0.2,
    ):
        """加载 GLEE 模型。

        Args:
            config_path (str): detectron2 配置，默认 SwinL.yaml。
            checkpoint_path (str): GLEE 权重路径。
            clip_path (str): CLIP 文本编码器目录，会写入环境变量 ``GLEE_CLIP_PATH``。
            device (str): 推理设备。
            num_inst_select (int): 每帧最多取多少个候选实例。
            score_threshold (float): 分数阈值，低于它的实例被丢弃。
        """
        # GLEE_Model 构造时按 GLEE_CLIP_PATH 定位 CLIP 文本编码器，必须在建模之前设置。
        os.environ.setdefault("GLEE_CLIP_PATH", clip_path)

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_glee_config(cfg)
        cfg.merge_from_file(config_path)

        self.device = device
        self.num_inst_select = num_inst_select
        self.score_threshold = score_threshold

        self.model = GLEE_Model(cfg, None, device, None, True).to(device)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self._pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(3, 1, 1)
        self._pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(3, 1, 1)

    def segment(self, image_rgb: np.ndarray, text: str, score_threshold=None) -> SegResult:
        """在 RGB uint8 图像中分割出与 ``text`` 匹配的实例。

        Args:
            image_rgb (np.ndarray): ``(H, W, 3)`` RGB uint8。
            text (str): 类别提示，逗号分隔可一次查询多个类别。
            score_threshold (float, optional): 覆盖构造时的分数阈值。

        Returns:
            SegResult: 按分数降序排列的实例。
        """
        started = time.perf_counter()
        thr = self.score_threshold if score_threshold is None else float(score_threshold)
        categories = parse_categories(text)
        height, width = image_rgb.shape[:2]

        # 在 GPU 上做归一化，与参考实现一致。
        image = torch.as_tensor(np.ascontiguousarray(image_rgb.transpose(2, 0, 1)))
        image = ((image.to(self.device) - self._pixel_mean) / self._pixel_std)[None]

        resized = torchvision.transforms.Resize(800)(image)
        image_size = [resized.shape[-2], resized.shape[-1]]
        stride = 32
        padded = ((torch.tensor(image_size) + (stride - 1)) // stride * stride).tolist()

        batched = torch.zeros(1, 3, padded[0], padded[1], device=resized.device)
        batched[0, :, : image_size[0], : image_size[1]] = resized

        with torch.no_grad():
            outputs, _ = self.model(
                batched, [], task="coco",
                batch_name_list=categories, is_train=False,
            )

        boxes = outputs["pred_boxes"][0]
        boxes[:, 0], boxes[:, 2] = (
            boxes[:, 0] * width - boxes[:, 2] * width * 0.5,
            boxes[:, 0] * width + boxes[:, 2] * width * 0.5,
        )
        boxes[:, 1], boxes[:, 3] = (
            boxes[:, 1] * height - boxes[:, 3] * height * 0.5,
            boxes[:, 1] * height + boxes[:, 3] * height * 0.5,
        )

        mask_logits = outputs["pred_logits"][0]
        mask_pred = outputs["pred_masks"][0]
        scores = mask_logits.sigmoid().max(-1)[0]
        scores, indices = scores.topk(max(self.num_inst_select, 1), sorted=True)

        keep = scores > thr
        indices, scores = indices[keep], scores[keep]
        if len(indices) == 0:
            return empty_result((time.perf_counter() - started) * 1000.0, height, width)

        labels = [categories[c] for c in mask_logits[indices].max(-1)[1].tolist()]
        boxes = boxes[indices].cpu().numpy().astype(np.float32)

        masks = F.interpolate(
            mask_pred[indices][None], size=tuple(padded), mode="bilinear", align_corners=False
        )
        masks = masks[:, :, : image_size[0], : image_size[1]]
        masks = F.interpolate(masks, size=(height, width), mode="bilinear", align_corners=False)
        masks = (masks > 0).cpu().numpy()[0]

        return SegResult(
            boxes=boxes,
            masks=masks,
            scores=scores.cpu().numpy().astype(np.float32),
            labels=labels,
            time_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        """释放模型与显存。"""
        del self.model
        torch.cuda.empty_cache()
