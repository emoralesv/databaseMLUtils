from __future__ import annotations

from typing import Tuple

from PIL import Image


def normalize_bbox(bbox_xywh: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
    """Normalize bbox to [0,1] relative coords.

    Args:
        bbox_xywh: Bounding box (x, y, w, h) in pixels.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Normalized bbox (x, y, w, h) with values in [0,1].
    """
    x, y, w, h = bbox_xywh
    if width <= 0 or height <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return x / width, y / height, w / width, h / height


def denormalize_bbox(bbox_xywh: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Denormalize bbox from [0,1] to pixel coords.

    Args:
        bbox_xywh: Normalized bounding box (x, y, w, h).
        width: Image width.
        height: Image height.

    Returns:
        Pixel bbox (x, y, w, h) as integers.
    """
    x, y, w, h = bbox_xywh
    return int(round(x * width)), int(round(y * height)), int(round(w * width)), int(round(h * height))


def crop_by_bbox(
    img: Image.Image,
    bbox_xywh: Tuple[float, float, float, float],
    normalized: bool = False,
    clip: bool = True,
) -> Image.Image:
    """Crop an image using (x, y, w, h).

    Args:
        img: Input PIL image.
        bbox_xywh: Bounding box (x, y, w, h).
        normalized: If True, treat bbox as normalized [0,1].
        clip: If True, clip to image bounds.

    Returns:
        Cropped PIL Image.
    """
    W, H = img.size
    if normalized:
        x, y, w, h = denormalize_bbox(bbox_xywh, W, H)
    else:
        x, y, w, h = bbox_xywh
    x0, y0, x1, y1 = x, y, x + w, y + h
    if clip:
        x0 = max(0, min(W - 1, x0))
        y0 = max(0, min(H - 1, y0))
        x1 = max(1, min(W, x1))
        y1 = max(1, min(H, y1))
    return img.crop((int(x0), int(y0), int(x1), int(y1)))

