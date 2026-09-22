"""分割结果的可视化绘制工具。

所有函数收发的都是 **RGB** uint8 图像，与 ``perception.base`` 一致；唯一的 RGB->BGR
转换发生在 ``save_image`` 里，其余地方不必关心通道序。
"""

from typing import Optional

import cv2
import numpy as np

# 一组区分度高、对色盲友好的 RGB 颜色，多个实例循环取用。
PALETTE = [
    (230, 75, 60), (60, 130, 230), (60, 175, 90), (240, 175, 40),
    (150, 90, 220), (40, 195, 195), (235, 110, 175), (140, 140, 60),
    (90, 90, 235), (200, 130, 60), (60, 200, 150), (220, 90, 130),
    (120, 200, 60), (80, 160, 220), (210, 160, 210), (170, 170, 170),
]

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label_text(label: str, score: Optional[float]) -> str:
    """把标签和分数拼成显示文字；分数为 ``None`` 时只返回标签。"""
    if score is None:
        return label
    return f"{label} {score:.2f}"


def overlay_instances(
    image_rgb: np.ndarray,
    result,
    alpha: float = 0.45,
    show_boxes: bool = True,
    show_centers: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """给每个实例的 mask 上色、画外框并标出中心像素。

    中心标记画的就是 ``perception.base.mask_center_pixel`` 会交给 2D->3D 反投影的那个
    像素，因此也可用来检查准备抬升到 3D 的点是否确实落在物体上。

    Args:
        image_rgb (np.ndarray): ``(H, W, 3)`` RGB uint8。
        result (SegResult): 某个后端返回的实例集合。
        alpha (float): mask 上色的不透明度。
        show_boxes (bool): 是否画外框。
        show_centers (bool): 是否标出中心像素。
        thickness (int): 外框线宽。

    Returns:
        np.ndarray: 叠加后的新图像，不改动输入。
    """
    out = image_rgb.astype(np.float32).copy()

    for i in range(len(result)):
        mask = result.masks[i].astype(bool)
        if not mask.any():
            continue
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        out[mask] = out[mask] * (1.0 - alpha) + color * alpha

    out = out.astype(np.uint8)

    for i in range(len(result)):
        color = PALETTE[i % len(PALETTE)]
        label = _label_text(
            result.labels[i] if i < len(result.labels) else f"obj{i}",
            float(result.scores[i]) if i < len(result.scores) else None,
        )

        if show_boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in result.boxes[i])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            _draw_tag(out, label, (x1, max(y1 - 6, 12)), color)

        if show_centers:
            center = _center_of(result.masks[i])
            if center is not None:
                _draw_crosshair(out, center, color)

    return out


def _center_of(mask: np.ndarray):
    """mask 的质心像素；mask 为空时返回 ``None``。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def _draw_crosshair(image: np.ndarray, center, color, size: int = 7) -> None:
    """在 ``center`` 处画一个带黑边的十字与圆点。"""
    x, y = center
    arm = size + 3
    cv2.line(image, (x - arm, y), (x + arm, y), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(image, (x, y - arm), (x, y + arm), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(image, (x - arm, y), (x + arm, y), color, 2, cv2.LINE_AA)
    cv2.line(image, (x, y - arm), (x, y + arm), color, 2, cv2.LINE_AA)
    cv2.circle(image, (x, y), size, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), size, color, 2, cv2.LINE_AA)


def _draw_tag(image: np.ndarray, text: str, origin, color) -> None:
    """在填充背景上写字，使其在任意图像上都清晰可读。

    标签会被钳制在画面内，因此贴着右边或下边的 box 其标签仍然完整可见。
    """
    height, width = image.shape[:2]
    x, y = origin
    scale, thick = 0.5, 1
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, thick)

    y = min(max(y, th + 4), height - base - 2)
    x = min(max(x, 0), max(width - tw - 8, 0))

    cv2.rectangle(image, (x, y - th - 4), (x + tw + 6, y + base), color, -1)
    cv2.putText(image, text, (x + 3, y), _FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _fit_text(text: str, max_width: int, scale: float, thick: int):
    """先逐步缩小 ``scale``（不低于下限），再截断文字，直到能放进 ``max_width``。"""
    min_scale = 0.34
    while scale > min_scale:
        if cv2.getTextSize(text, _FONT, scale, thick)[0][0] <= max_width:
            return text, scale
        scale -= 0.03

    while len(text) > 4 and cv2.getTextSize(text + "..", _FONT, scale, thick)[0][0] > max_width:
        text = text[:-1]
    return text, scale


def draw_info_bar(image_rgb: np.ndarray, lines) -> np.ndarray:
    """在图像顶部拼接一条深色信息条，用于显示提示词、后端、耗时等。

    文字会先缩小再截断到图像宽度，竖版图不会把标题尾部切掉。

    Args:
        image_rgb (np.ndarray): ``(H, W, 3)`` RGB uint8。
        lines: 一行文字，或文字行的列表。

    Returns:
        np.ndarray: 顶部带信息条的新图像。
    """
    if isinstance(lines, str):
        lines = [lines]

    out = image_rgb.copy()
    thick, pad = 1, 6
    max_width = max(out.shape[1] - 2 * pad, 16)

    fitted = [_fit_text(str(line), max_width, 0.55, thick) for line in lines]
    line_h = int(cv2.getTextSize("Ag", _FONT, fitted[0][1], thick)[0][1]) + pad
    bar_h = line_h * len(fitted) + pad

    strip = np.full((bar_h, out.shape[1], 3), 24, dtype=np.uint8)
    for i, (line, scale) in enumerate(fitted):
        y = pad + (i + 1) * line_h - 3
        cv2.putText(strip, line, (pad, y), _FONT, scale, (235, 235, 235), thick, cv2.LINE_AA)

    return np.vstack([strip, out])


def side_by_side(panels, title: str = "", gap: int = 12):
    """把多张 RGB 图横向拼接，较矮或较窄的补灰边。

    Args:
        panels: 图像序列，``None`` 会被跳过。
        title (str): 非空时在结果顶部加一条信息条。
        gap (int): 图与图之间的间隔像素数。

    Returns:
        np.ndarray: 拼接结果；``panels`` 全为空时返回 ``None``。
    """
    panels = [p for p in panels if p is not None]
    if not panels:
        return None

    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] == h and p.shape[1] == w:
            padded.append(p)
            continue
        canvas = np.full((h, w, 3), 60, dtype=np.uint8)
        canvas[: p.shape[0], : p.shape[1]] = p
        padded.append(canvas)

    sep = np.full((h, gap, 3), 60, dtype=np.uint8)
    out = padded[0]
    for p in padded[1:]:
        out = np.hstack([out, sep, p])

    if title:
        out = draw_info_bar(out, [title])
    return out


def save_image(path: str, image_rgb: np.ndarray) -> None:
    """把 RGB 图写到磁盘（cv2 要 BGR，在这里转换）。

    Raises:
        IOError: 写入失败时抛出。
    """
    if not cv2.imwrite(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write image: {path}")
