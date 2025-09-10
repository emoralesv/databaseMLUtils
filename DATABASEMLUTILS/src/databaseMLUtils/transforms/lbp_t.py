from __future__ import annotations

import numpy as np
from skimage.feature import local_binary_pattern

from databaseMLUtils.transform_base import Transform


class LBPTransform(Transform):
    id = "LBP"
    description = "Local Binary Pattern (var) with radius=3"

    def apply(self, img):
        gray = self.to_grayscale_u8(img)
        radius = 3
        n_points = 8 * radius
        return local_binary_pattern(gray, n_points, radius, method="var").astype(np.float32)

