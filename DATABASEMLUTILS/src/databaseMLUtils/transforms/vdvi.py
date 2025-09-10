from __future__ import annotations

from databaseMLUtils.transform_base import Transform


class VDVITransform(Transform):
    id = "VDVI"
    description = "Visible-band Difference Vegetation Index: (2G − R − B)/(2G + R + B)"

    def apply(self, img):
        rgb = self.pil_to_rgb_np(img)
        R = rgb[..., 0]
        G = rgb[..., 1]
        B = rgb[..., 2]
        eps = 1e-10
        return (2 * G - R - B) / ((2 * G + R + B) + eps)

