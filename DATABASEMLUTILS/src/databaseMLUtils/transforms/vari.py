from __future__ import annotations

from databaseMLUtils.transform_base import Transform


class VARITransform(Transform):
    id = "VARI"
    description = "Visible Atmospherically Resistant Index: (G − R)/(G + R − B)"

    def apply(self, img):
        rgb = self.pil_to_rgb_np(img)
        R = rgb[..., 0]
        G = rgb[..., 1]
        B = rgb[..., 2]
        eps = 1e-10
        return (G - R) / ((G + R - B) + eps)

