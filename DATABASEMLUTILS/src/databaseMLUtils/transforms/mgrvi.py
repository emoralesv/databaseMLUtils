from __future__ import annotations

import numpy as np
from databaseMLUtils.transform_base import Transform


class MGRVITransform(Transform):
    id = "MGRVI"
    description = "Modified Green-Red Vegetation Index: (G^2 − R^2)/(G^2 + R^2)"

    def apply(self, img):
        rgb = self.pil_to_rgb_np(img)
        R = rgb[..., 0]
        G = rgb[..., 1]
        eps = 1e-10
        return (G ** 2 - R ** 2) / ((G ** 2 + R ** 2) + eps)

