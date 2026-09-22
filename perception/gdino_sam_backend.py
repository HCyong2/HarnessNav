"""GroundingDINO + SAM 后端：文本先出 box，再出 mask。

GroundingDINO 负责语义（哪块区域对应提示词），SAM 负责 box 内像素级精确的 mask。
"""

import os
import time
from typing import List

import numpy as np
import torch
from groundingdino.datasets import transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from torchvision.ops import box_convert

from .base import SegBackend, SegResult, empty_result

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DINO_CONFIG = os.path.join(
    REPO_ROOT, "thirdparty", "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"
)
DEFAULT_DINO_CKPT = os.path.join(REPO_ROOT, "model", "groundingdino_swint_ogc.pth")
DEFAULT_SAM_CKPT = os.path.join(REPO_ROOT, "model", "sam_vit_h_4b8939.pth")
# GroundingDINO 的文本编码器是 bert-base-uncased。配置里写的是 hub id，这里保留一份
# 本地副本，使该后端无需联网也能跑。
DEFAULT_BERT_DIR = os.path.join(REPO_ROOT, "model", "bert-base-uncased")


def preprocess_caption(caption: str) -> str:
    """把提示词转成 GroundingDINO 需要的格式：转小写、去首尾空白、补句点。

    Args:
        caption (str): 原始提示词。

    Returns:
        str: 处理后的提示词。
    """
    caption = caption.lower().strip()
    if caption.endswith("."):
        return caption
    return caption + "."


class GDinoSAMBackend(SegBackend):
    name = "gdino_sam"

    def __init__(
        self,
        dino_config: str = DEFAULT_DINO_CONFIG,
        dino_checkpoint: str = DEFAULT_DINO_CKPT,
        sam_checkpoint: str = DEFAULT_SAM_CKPT,
        sam_type: str = "vit_h",
        device: str = "cuda:0",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        max_instances: int = 15,
    ):
        """加载 GroundingDINO 与 SAM 两个模型。

        Args:
            dino_config (str): GroundingDINO 的模型配置文件。
            dino_checkpoint (str): GroundingDINO 权重路径。
            sam_checkpoint (str): SAM 权重路径。
            sam_type (str): SAM 的模型规格，如 ``vit_h``。
            device (str): 推理设备。
            box_threshold (float): GroundingDINO 的 box 分数阈值。
            text_threshold (float): GroundingDINO 的文本匹配阈值。
            max_instances (int): 每帧最多保留多少个 box。
        """
        self.device = torch.device(device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_instances = max_instances

        # --- GroundingDINO：文本 -> box ---
        args = SLConfig.fromfile(dino_config)
        args.device = device
        if os.path.isdir(DEFAULT_BERT_DIR):
            args.text_encoder_type = DEFAULT_BERT_DIR
        dino = build_model(args)
        checkpoint = torch.load(dino_checkpoint, map_location="cpu")
        dino.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.dino = dino.eval().to(self.device)

        self._transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # --- SAM：box -> mask ---
        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
        self.sam = SamPredictor(sam.to(self.device))

    def _detect(self, image_rgb: np.ndarray, caption: str, box_threshold=None):
        """GroundingDINO 推理，返回 ``(boxes xyxy 像素, scores, phrases)``。

        Args:
            image_rgb (np.ndarray): ``(H, W, 3)`` RGB uint8。
            caption (str): 原始提示词。
            box_threshold (float, optional): 覆盖构造时的 box 分数阈值。

        Returns:
            tuple: box 数组 ``(N, 4)``、分数数组 ``(N,)``、匹配到的短语列表。
        """
        height, width = image_rgb.shape[:2]
        transformed, _ = self._transform(Image.fromarray(image_rgb), None)
        transformed = transformed.to(self.device)

        caption = preprocess_caption(caption)
        with torch.no_grad():
            outputs = self.dino(transformed[None], captions=[caption])

        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        box_thr = self.box_threshold if box_threshold is None else float(box_threshold)
        keep = logits.max(dim=1)[0] > box_thr
        logits, boxes = logits[keep], boxes[keep]
        if boxes.numel() == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []

        # cxcywh（归一化）-> xyxy（像素）
        boxes = boxes * torch.tensor([width, height, width, height])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").clamp(
            min=0
        )
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(max=width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(max=height)

        tokenized = self.dino.tokenizer(caption)
        phrases = [
            get_phrases_from_posmap(row > self.text_threshold, tokenized, self.dino.tokenizer)
            .replace(".", "")
            for row in logits
        ]

        scores, order = logits.max(dim=1)[0].sort(descending=True)
        order = order[: self.max_instances]
        return (
            boxes[order].numpy().astype(np.float32),
            scores[: self.max_instances].numpy().astype(np.float32),
            [phrases[i] for i in order.tolist()],
        )

    def segment(self, image_rgb: np.ndarray, text: str, score_threshold=None) -> SegResult:
        """在 RGB uint8 图像中分割出与 ``text`` 匹配的实例。

        Args:
            image_rgb (np.ndarray): ``(H, W, 3)`` RGB uint8。
            text (str): 提示词。
            score_threshold (float, optional): 覆盖 box 分数阈值。

        Returns:
            SegResult: 按分数降序排列的实例。
        """
        started = time.perf_counter()
        height, width = image_rgb.shape[:2]
        box_thr = self.box_threshold if score_threshold is None else float(score_threshold)

        boxes, scores, phrases = self._detect(image_rgb, text, box_threshold=box_thr)
        if len(boxes) == 0:
            return empty_result((time.perf_counter() - started) * 1000.0, height, width)

        # SAM 的图像嵌入只需算一次，所有 box 共用。
        self.sam.set_image(image_rgb)

        masks, kept_boxes, kept_scores, kept_labels = [], [], [], []
        for box, score, phrase in zip(boxes, scores, phrases):
            mask_stack, quality, _ = self.sam.predict(
                box=box[None], multimask_output=True
            )
            masks.append(mask_stack[int(np.argmax(quality))])
            kept_boxes.append(box)
            kept_scores.append(score)
            kept_labels.append(phrase)

        return SegResult(
            boxes=np.stack(kept_boxes).astype(np.float32),
            masks=np.stack(masks).astype(bool),
            scores=np.asarray(kept_scores, dtype=np.float32),
            labels=kept_labels,
            time_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        """释放两个模型与显存。"""
        del self.dino, self.sam
        torch.cuda.empty_cache()
