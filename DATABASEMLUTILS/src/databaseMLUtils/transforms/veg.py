from __future__ import annotations

import numpy as np
from databaseMLUtils.transform_base import Transform


class VEGTransform(Transform):
    id = "VEG"
    description = "G / (R^a * B^(1-a)), a=0.667"

    def apply(self, img):
        rgb = self.pil_to_rgb_np(img)
        R = rgb[..., 0]
        G = rgb[..., 1]
        B = rgb[..., 2]
        a = 0.667
        eps = 1e-10
        return G / ((R ** a) * (B ** (1 - a)) + eps)

