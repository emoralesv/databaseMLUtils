from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Tuple

from PIL import Image


@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


def read_coco(annotations_path: str | os.PathLike) -> Tuple[Dict[int, CocoImage], List[dict]]:
    """Read a COCO annotations file.

    Args:
        annotations_path: Path to instances JSON.

    Returns:
        images_by_id: Mapping from image id to CocoImage.
        annotations: List of annotation dicts.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images_by_id: Dict[int, CocoImage] = {}
    for im in data.get("images", []):
        images_by_id[int(im["id"]) ] = CocoImage(
            id=int(im["id"]),
            file_name=str(im["file_name"]),
            width=int(im.get("width", 0)),
            height=int(im.get("height", 0)),
        )
    return images_by_id, list(data.get("annotations", []))


def iter_coco_objects(
    annotations_path: str | os.PathLike,
    images_dir: str | os.PathLike,
) -> Generator[Tuple[Image.Image, dict], None, None]:
    """Yield image and annotation pairs from COCO annotations.

    Args:
        annotations_path: Path to instances JSON.
        images_dir: Directory where images live.

    Yields:
        (PIL.Image, annotation dict)
    """
    images_by_id, annotations = read_coco(annotations_path)
    images_dir = Path(images_dir)
    for ann in annotations:
        img_info = images_by_id.get(int(ann["image_id"]))
        if img_info is None:
            continue
        img_path = images_dir / img_info.file_name
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        yield img, ann


def save_classification_crop(
    img: Image.Image,
    class_name: str,
    out_dir: str | os.PathLike,
    base_name: str,
    index: int,
) -> str:
    """Save a crop image under class folder.

    Args:
        img: PIL image to save.
        class_name: Class label name.
        out_dir: Root output directory.
        base_name: Base file stem.
        index: Index to ensure uniqueness.

    Returns:
        Path to saved image as str.
    """
    class_dir = Path(out_dir) / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    out_path = class_dir / f"{Path(base_name).stem}_{index}.jpg"
    img.save(out_path, format="JPEG", quality=95)
    return str(out_path)


# Minimal PASCAL VOC reader (single file)
def read_voc_xml(xml_path: str | os.PathLike) -> List[dict]:
    """Read PASCAL VOC XML and return list of objects with bbox and name.

    Args:
        xml_path: Path to VOC XML file.

    Returns:
        List of dicts with keys: name, bbox(xmin, ymin, xmax, ymax)
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs: List[dict] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bnd = obj.find("bndbox")
        if bnd is None or name_el is None:
            continue
        xmin = int(float(bnd.findtext("xmin", default="0")))
        ymin = int(float(bnd.findtext("ymin", default="0")))
        xmax = int(float(bnd.findtext("xmax", default="0")))
        ymax = int(float(bnd.findtext("ymax", default="0")))
        objs.append({
            "name": name_el.text or "unknown",
            "bbox": (xmin, ymin, xmax - xmin, ymax - ymin),
        })
    return objs

