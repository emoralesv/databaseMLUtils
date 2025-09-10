from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


class Transform:
    """Prototype for per-image transforms that return a normalized image.

    Each transform has an identifier and description. Call with a PIL image to
    get a normalized single-channel PIL image suitable for saving.
    """

    id: str = ""
    description: str = ""

    def __call__(self, img: Image.Image) -> Image.Image:
        view = self.apply(img)
        norm = self.normalize_to_u8(view)
        return Image.fromarray(norm)

    def apply(self, img: Image.Image) -> np.ndarray:  # to be implemented by subclasses
        raise NotImplementedError

    @staticmethod
    def pil_to_rgb_np(img: Image.Image) -> np.ndarray:
        arr = np.asarray(img.convert("RGB"))
        return arr.astype(np.float32)

    @staticmethod
    def to_grayscale_u8(img: Image.Image) -> np.ndarray:
        return np.asarray(img.convert("L"))

    @staticmethod
    def normalize_to_u8(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        m, M = float(np.min(x)), float(np.max(x))
        if not np.isfinite(m) or not np.isfinite(M):
            return np.zeros_like(x, dtype=np.uint8)
        if abs(M - m) < 1e-12:
            return np.zeros_like(x, dtype=np.uint8)
        y = (x - m) / (M - m)
        y = np.clip(y * 255.0, 0, 255)
        return y.astype(np.uint8)

