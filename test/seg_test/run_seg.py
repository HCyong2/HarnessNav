#!/usr/bin/env python
"""在样例图片上跑文本驱动的分割后端，并渲染结果。

用法
----
    python test/seg_test/run_seg.py --text "chair"
    python test/seg_test/run_seg.py --text "chair, table" --backend all
    python test/seg_test/run_seg.py --image img/room1.jpg --text "sofa" --backend gdino_sam

默认读取 ``test/seg_test/img/`` 下的所有图片（或用 ``--image`` 指定单张），把标注后的
``<name>_<backend>.png`` 写到 ``test/seg_test/out/``。``--backend all`` 会跑两个后端，
并额外输出一张 ``<name>_compare.png`` 用于并排对比。

每个实例还会打印 mask 的中心像素——那是后续 2D->3D 反投影会抬升到场景中的点。
"""

import argparse
import gc
import os
import sys
from glob import glob

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from perception import mask_center_pixel  # noqa: E402
from perception.render import draw_info_bar, overlay_instances, save_image, side_by_side  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMG_DIR = os.path.join(HERE, "img")
DEFAULT_OUT_DIR = os.path.join(HERE, "out")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

BACKENDS = ("glee", "gdino_sam")


def build_backend(name: str, args):
    """按名字构造一个分割后端。

    后端模块在这里按需导入，只跑其中一个后端时不必付出另一个后端的导入开销，因此
    这几处 import 有意不提到文件顶部。

    Args:
        name (str): ``glee`` 或 ``gdino_sam``。
        args: argparse 的解析结果，提供各后端所需的阈值与设备参数。

    Returns:
        SegBackend: 对应后端的实例。

    Raises:
        ValueError: ``name`` 不是已知的后端名。
    """
    if name == "glee":
        from perception.glee_backend import GLEEBackend

        return GLEEBackend(
            device=args.device,
            score_threshold=args.glee_threshold,
            num_inst_select=args.max_instances,
        )

    if name == "gdino_sam":
        from perception.gdino_sam_backend import GDinoSAMBackend

        return GDinoSAMBackend(
            device=args.device,
            sam_type=args.sam_type,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_instances=args.max_instances,
        )

    raise ValueError(f"unknown backend: {name}")


def collect_images(target: str):
    """收集要处理的图片路径。

    Args:
        target (str): 图片文件或目录。

    Returns:
        list: 图片路径列表；目录按文件名排序。
    """
    if os.path.isdir(target):
        paths = sorted(
            p for p in glob(os.path.join(target, "*")) if p.lower().endswith(IMAGE_EXTS)
        )
    else:
        paths = [target]
    return paths


def render(image_rgb: np.ndarray, result, backend_name: str, text: str, image_name: str):
    """把分割结果画到原图上，并在顶部加上信息条。

    Args:
        image_rgb (np.ndarray): ``(H, W, 3)`` RGB uint8。
        result (SegResult): 后端返回的实例集合。
        backend_name (str): 后端名，写进信息条。
        text (str): 本次使用的提示词。
        image_name (str): 图片名，写进信息条。

    Returns:
        np.ndarray: 叠加后的图像。
    """
    canvas = overlay_instances(image_rgb, result)
    return draw_info_bar(canvas, [
        f"{image_name}  |  backend: {backend_name}  |  prompt: \"{text}\"",
        f"instances: {len(result)}   inference: {result.time_ms:.0f} ms",
    ])


def report(result, image_name: str, backend_name: str) -> None:
    """把每个实例的标签、分数、box 与中心像素打印到终端。

    Args:
        result (SegResult): 后端返回的实例集合。
        image_name (str): 图片名。
        backend_name (str): 后端名。
    """
    if len(result) == 0:
        print(f"  [{backend_name}] {image_name}: no instances above threshold")
        return
    print(f"  [{backend_name}] {image_name}: {len(result)} instance(s)")
    for i in range(len(result)):
        center = mask_center_pixel(result.masks[i])
        x1, y1, x2, y2 = result.boxes[i]
        print(
            f"    #{i} {result.labels[i]:<20} score={result.scores[i]:.3f} "
            f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
            f"center={center} area={int(result.masks[i].sum())}px"
        )


def main():
    """命令行入口：逐个后端跑完所有图片，再输出对比图。

    Returns:
        int: 进程退出码，0 表示成功。
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=DEFAULT_IMG_DIR,
                        help="image file or directory (default: test/seg_test/img)")
    parser.add_argument("--text", required=True, help='target prompt, e.g. "chair" or "chair, table"')
    parser.add_argument("--backend", default="all", choices=list(BACKENDS) + ["all"])
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sam-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--box-threshold", type=float, default=0.3, help="GroundingDINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold")
    parser.add_argument("--glee-threshold", type=float, default=0.4, help="GLEE score threshold")
    parser.add_argument("--max-instances", type=int, default=15)
    args = parser.parse_args()

    images = collect_images(args.image)
    if not images:
        print(f"No images found in {args.image}. Put some images there and re-run.")
        return 1

    os.makedirs(args.out, exist_ok=True)
    names = BACKENDS if args.backend == "all" else (args.backend,)

    print(f"prompt  : {args.text}")
    print(f"images  : {len(images)} from {args.image}")
    print(f"backends: {', '.join(names)}")
    print(f"output  : {args.out}\n")

    # 一次只加载一个后端，用完立即释放，这样 'all' 不需要让 GLEE + DINO + SAM 同时驻留显存。
    for backend_name in names:
        print(f"loading {backend_name} ...")
        backend = build_backend(backend_name, args)
        try:
            for path in images:
                stem, _ = os.path.splitext(os.path.basename(path))
                image_bgr = cv2.imread(path)
                if image_bgr is None:
                    print(f"  !! could not read {path}, skipping")
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                result = backend.segment(image_rgb, args.text)
                report(result, stem, backend_name)

                out_path = os.path.join(args.out, f"{stem}_{backend_name}.png")
                save_image(out_path, render(image_rgb, result, backend_name, args.text, stem))
                print(f"    -> {out_path}")
        finally:
            backend.close()
            del backend
            gc.collect()
            torch.cuda.empty_cache()
        print()

    if len(names) == 2:
        for path in images:
            stem, _ = os.path.splitext(os.path.basename(path))
            panels = []
            for backend_name in names:
                p = os.path.join(args.out, f"{stem}_{backend_name}.png")
                if os.path.exists(p):
                    panels.append(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
            if len(panels) == 2:
                compare_path = os.path.join(args.out, f"{stem}_compare.png")
                save_image(compare_path, side_by_side(panels))
                print(f"compare -> {compare_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
