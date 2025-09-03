from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

from PIL import Image

from .bbox import crop_by_bbox
from .io import iter_coco_objects, save_classification_crop
from .reporting import class_distribution_report
from .transforms import dynamic_crop, resize_image


def _convert_det2cls(args: argparse.Namespace) -> None:
    ann = args.ann
    images_dir = args.images
    out_dir = args.out
    resize = args.resize
    size = args.size
    dynamic = args.dynamic_crop

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i, (img, ann) in enumerate(iter_coco_objects(ann, images_dir)):
        x, y, w, h = ann["bbox"]
        crop = dynamic_crop(img, (x, y, w, h), margin=0.1, square=True) if dynamic else crop_by_bbox(img, (x, y, w, h))
        if resize != "none":
            crop = resize_image(crop, size)
        class_name = str(ann.get("category_id", "unknown"))
        base_name = Path(ann.get("image_id", i)).name if isinstance(ann.get("image_id"), str) else str(ann.get("image_id", i))
        save_classification_crop(crop, class_name=class_name, out_dir=out_dir, base_name=base_name, index=i)


def _report(args: argparse.Namespace) -> None:
    df = class_distribution_report(args.src, out_csv=args.out)
    print(df)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dbutils", description="databaseMLUtils CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_c = sub.add_parser("convert", help="Convert datasets")
    p_c.add_argument("--task", choices=["det2cls"], required=True)
    p_c.add_argument("--ann", type=str, help="Path to COCO instances.json")
    p_c.add_argument("--images", type=str, help="Path to images directory")
    p_c.add_argument("--out", type=str, required=True, help="Output directory")
    p_c.add_argument("--resize", choices=["none", "fixed"], default="none")
    p_c.add_argument("--size", type=int, default=128)
    p_c.add_argument("--dynamic-crop", choices=["on", "off"], default="on")
    p_c.set_defaults(func=_convert_det2cls)

    p_r = sub.add_parser("report", help="Report dataset stats")
    p_r.add_argument("--task", choices=["cls"], required=True)
    p_r.add_argument("--src", type=str, required=True)
    p_r.add_argument("--out", type=str, required=False, default=None)
    p_r.set_defaults(func=_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "dynamic_crop"):
        args.dynamic_crop = args.dynamic_crop == "on"
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

