from __future__ import annotations

"""
Generate classification views from a Pascal VOC (XML) dataset by applying
transform functions defined in `databaseMLUtils.transforms`.

Usage example:

    python -m databaseMLUtils.examples.xml_to_classification_views \
        --dataset ./databases/tomate_rugoso_merge-1 \
        --out ./databases/rugoso_classification \
        --class-name Rugoso \
        --transforms MGRVI,VEG,GRRI,VARI,NGRDI,VDVI

This will search for typical Roboflow/VOC folders like `*/labels/*.xml` and the
corresponding `*/images/*.jpg` directories, then run each selected transform
over all images and save results into:

    out/class-name/<transform-id>/*.jpg

New transform functions added to `databaseMLUtils.transforms` can be used by
passing their identifier (function name) via `--transforms`.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Set

from tqdm import tqdm

from databaseMLUtils.transforms import get_transforms_registry
from databaseMLUtils.io import save_classification_view, read_voc_xml
from databaseMLUtils.bbox import crop_by_bbox
from databaseMLUtils.image_ops import dynamic_crop, resize_image


def _ensure_trailing_sep(p: str | os.PathLike) -> str:
    s = str(p)
    return s if s.endswith(os.sep) else s + os.sep


def discover_image_dirs_from_voc(dataset_root: str | os.PathLike) -> List[str]:
    """Discover image directories in a VOC/Roboflow export tree.

    Heuristics: find any `labels` directories with `*.xml` files, then use the
    sibling `images` directory. Falls back to scanning for directories that
    contain `.jpg` files directly under them.
    """
    root = Path(dataset_root)
    image_dirs: Set[str] = set()
    # Heuristic 1: */labels/*.xml with sibling */images
    for labels_dir in root.rglob("labels"):
        if not labels_dir.is_dir():
            continue
        if not any(labels_dir.glob("*.xml")):
            continue
        sib_images = labels_dir.parent / "images"
        if sib_images.exists() and any(sib_images.glob("*.jpg")):
            image_dirs.add(str(sib_images.resolve()))
    # Heuristic 2: any directory with immediate .jpg files
    if not image_dirs:
        for d in root.rglob("*"):
            if d.is_dir() and any(d.glob("*.jpg")):
                image_dirs.add(str(d.resolve()))
    return sorted(image_dirs)


def _iter_images_from_dirs(image_dirs: Iterable[str]):
    for d in image_dirs:
        for p in Path(d).glob("*.jpg"):
            yield p


def _iter_voc_objects(dataset_root: str | os.PathLike):
    root = Path(dataset_root)
    for labels_dir in root.rglob("labels"):
        if not labels_dir.is_dir():
            continue
        images_dir = labels_dir.parent / "images"
        if not images_dir.exists():
            continue
        for xml_path in labels_dir.glob("*.xml"):
            objs = read_voc_xml(xml_path)
            stem = xml_path.stem
            # Assume .jpg; if missing, try common alternatives
            img_path = images_dir / f"{stem}.jpg"
            if not img_path.exists():
                for ext in [".jpeg", ".png", ".bmp"]:
                    if (images_dir / f"{stem}{ext}").exists():
                        img_path = images_dir / f"{stem}{ext}"
                        break
            if not img_path.exists():
                continue
            yield img_path, objs


def run_transforms(
    dataset_root: str | os.PathLike,
    out_root: str | os.PathLike,
    class_name: str,
    transform_ids: List[str],
    crop: str = "none",
    margin: float = 0.1,
    square: bool = True,
    resize: int | None = None,
) -> None:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    registry = get_transforms_registry()
    tfs = []
    for t_id in transform_ids:
        t = registry.get(t_id)
        if not t:
            print(f"Warning: transform '{t_id}' not found; skipping.")
            continue
        tfs.append(t)

    if not tfs:
        print("No valid transforms selected.")
        return

    log = {
        "dataset_root": str(dataset_root),
        "out_root": str(out_root),
        "class_name": class_name,
        "transforms": [t.id for t in tfs],
        "crop": crop,
        "margin": margin,
        "square": square,
        "resize": resize,
    }

    if crop in {"bbox", "dynamic"}:
        # Iterate over VOC XML objs
        idx = 0
        from PIL import Image as _PILImage
        for img_path, objs in _iter_voc_objects(dataset_root):
            for j, obj in enumerate(objs):
                try:
                    im = _PILImage.open(img_path).convert("RGB")
                except Exception:
                    continue
                x, y, w, h = obj.get("bbox", (0, 0, im.size[0], im.size[1]))
                if crop == "bbox":
                    crop_im = crop_by_bbox(im, (x, y, w, h))
                else:
                    crop_im = dynamic_crop(im, (x, y, w, h), margin=margin, square=square)
                if resize:
                    crop_im = resize_image(crop_im, resize)
                for t in tfs:
                    view = t(crop_im)
                    save_classification_view(view, class_name=class_name, transform_id=t.id, out_dir=out_root, base_name=img_path.stem, index=idx)
                idx += 1
    else:
        # No cropping: transform whole images
        idx = 0
        from PIL import Image as _PILImage
        image_dirs = discover_image_dirs_from_voc(dataset_root)
        for p in tqdm(list(_iter_images_from_dirs(image_dirs)), desc="Images"):
            try:
                im = _PILImage.open(p).convert("RGB")
            except Exception:
                continue
            if resize:
                im = resize_image(im, resize)
            for t in tfs:
                view = t(im)
                save_classification_view(view, class_name=class_name, transform_id=t.id, out_dir=out_root, base_name=p.stem, index=idx)
            idx += 1

    # Write a small log for reproducibility
    class_dir = Path(out_root) / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    (class_dir / "_views_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply transforms to VOC images and build classification views")
    p.add_argument("--dataset", type=str, required=True, help="Root folder of VOC/Roboflow dataset")
    p.add_argument("--out", type=str, required=True, help="Output root directory for classification dataset")
    p.add_argument("--class-name", type=str, required=True, help="Target class folder name (e.g., from combobox)")
    p.add_argument(
        "--transforms",
        type=str,
        default="MGRVI,VEG,GRRI,VARI,NGRDI,VDVI",
        help="Comma-separated list of transform identifiers from databaseMLUtils.transforms",
    )
    p.add_argument("--crop", choices=["none", "bbox", "dynamic"], default="none", help="Cropping mode using VOC bboxes")
    p.add_argument("--margin", type=float, default=0.1, help="Margin for dynamic crop")
    p.add_argument("--square", choices=["on", "off"], default="on", help="Square dynamic crop")
    p.add_argument("--resize", type=int, default=None, help="Optional fixed output size (e.g., 128)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    transform_ids = [s.strip() for s in args.transforms.split(",") if s.strip()]
    run_transforms(
        dataset_root=args.dataset,
        out_root=args.out,
        class_name=args.class_name,
        transform_ids=transform_ids,
        crop=args.crop,
        margin=args.margin,
        square=(args.square == "on"),
        resize=args.resize,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
