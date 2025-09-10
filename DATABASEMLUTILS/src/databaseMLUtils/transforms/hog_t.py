from __future__ import annotations

import numpy as np
from skimage.feature import hog

from databaseMLUtils.transform_base import Transform


class HOGTransform(Transform):
    id = "HOG"
    description = "Histogram of Oriented Gradients visualization"

    def apply(self, img):
        gray = self.to_grayscale_u8(img)
        _, view = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
        )
        return view.astype(np.float32)

