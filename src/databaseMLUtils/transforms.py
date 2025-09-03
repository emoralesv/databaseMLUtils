from __future__ import annotations

from typing import Tuple

from PIL import Image


def dynamic_crop(
    img: Image.Image,
    bbox_xywh: Tuple[float, float, float, float],
    margin: float = 0.1,
    square: bool = False,
) -> Image.Image:
    """Dynamically crop around bbox with optional margin and squaring.

    Args:
        img: PIL image.
        bbox_xywh: (x, y, w, h) in pixels.
        margin: Extra margin as a fraction of max(w, h).
        square: If True, expand to square crop.

    Returns:
        Cropped PIL image.
    """
    W, H = img.size
    x, y, w, h = bbox_xywh
    side = max(w, h)
    pad = side * margin
    cx, cy = x + w / 2.0, y + h / 2.0
    if square:
        new_w = new_h = side + 2 * pad
    else:
        new_w = w + 2 * pad
        new_h = h + 2 * pad
    x0 = int(round(cx - new_w / 2))
    y0 = int(round(cy - new_h / 2))
    x1 = int(round(cx + new_w / 2))
    y1 = int(round(cy + new_h / 2))
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(1, min(W, x1))
    y1 = max(1, min(H, y1))
    return img.crop((x0, y0, x1, y1))


def resize_image(img: Image.Image, size: int | Tuple[int, int]) -> Image.Image:
    """Resize image to fixed size.

    Args:
        img: PIL image.
        size: New size (int or (w, h)).

    Returns:
        Resized image.
    """
    if isinstance(size, int):
        size = (size, size)
    return img.resize(size, Image.BILINEAR)

