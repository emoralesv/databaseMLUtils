from __future__ import annotations

from databaseMLUtils.transform_base import Transform


class GRRITransform(Transform):
    id = "GRRI"
    description = "Green-Red Ratio Index: G / R"

    def apply(self, img):
        rgb = self.pil_to_rgb_np(img)
        R = rgb[..., 0]
        G = rgb[..., 1]
        eps = 1e-10
        return G / (R + eps)

