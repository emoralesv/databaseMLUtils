from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from databaseMLUtils.cli import build_parser


def test_coco_iter_and_convert(tmp_path: Path) -> None:
    # Create synthetic image
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    # Create minimal COCO
    coco = {
        "images": [{"id": 1, "file_name": str(img_path.name), "width": 64, "height": 64}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [16, 16, 16, 16]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 8, 8]},
        ],
    }
    ann_path = tmp_path / "instances.json"
    ann_path.write_text(json.dumps(coco))

    out_dir = tmp_path / "out"

    # Run via CLI helper
    parser = build_parser()
    args = parser.parse_args([
        "convert",
        "--task",
        "det2cls",
        "--ann",
        str(ann_path),
        "--images",
        str(tmp_path),
        "--out",
        str(out_dir),
        "--resize",
        "fixed",
        "--size",
        "32",
        "--dynamic-crop",
        "on",
    ])
    # Patch args structure like main()
    args.dynamic_crop = args.dynamic_crop == "on"
    args.func(args)

    # Check files created
    cls0 = out_dir / "0"
    cls1 = out_dir / "1"
    assert cls0.exists() and cls1.exists()
    assert any(cls0.glob("*.jpg")) and any(cls1.glob("*.jpg"))

