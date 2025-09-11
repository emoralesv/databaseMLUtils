from __future__ import annotations

from typing import Tuple

from PIL import Image


def resize_image(img: Image.Image, size: int | Tuple[int, int]) -> Image.Image:
    """Resize image to fixed size.

    If ``size`` is an int, produce a square (size x size). If it's a tuple,
    it must be (width, height).
    """
    if isinstance(size, int):
        dst = (int(size), int(size))
    else:
        dst = (int(size[0]), int(size[1]))
    return img.resize(dst, Image.BILINEAR)


def dynamic_crop(
    img: Image.Image,
    bbox_xywh: Tuple[float, float, float, float],
    margin: float = 0.1,
    square: bool = True,
) -> Image.Image:
    """Crop around bbox with margin and optional square aspect.

    Args:
        img: PIL image.
        bbox_xywh: (x, y, w, h) in pixels.
        margin: Fractional margin relative to max(w, h).
        square: If True, expand to square keeping center.
    """
    W, H = img.size
    x, y, w, h = bbox_xywh
    # Base side with margin
    side = max(w, h)
    side = side * (1.0 + float(margin) * 2.0)
    if square:
        w_exp = h_exp = side
    else:
        w_exp = w * (1.0 + float(margin) * 2.0)
        h_exp = h * (1.0 + float(margin) * 2.0)
    # Center of bbox
    cx = x + w * 0.5
    cy = y + h * 0.5
    x0 = int(round(cx - w_exp * 0.5))
    y0 = int(round(cy - h_exp * 0.5))
    x1 = int(round(cx + w_exp * 0.5))
    y1 = int(round(cy + h_exp * 0.5))
    # Clip to bounds and ensure at least 1px area
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(x0 + 1, min(W, x1))
    y1 = max(y0 + 1, min(H, y1))
    return img.crop((x0, y0, x1, y1))

