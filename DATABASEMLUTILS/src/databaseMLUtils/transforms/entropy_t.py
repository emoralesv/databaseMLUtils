from __future__ import annotations

import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk

from databaseMLUtils.transform_base import Transform


class EntropyTransform(Transform):
    id = "ENTROPY"
    description = "Local entropy with disk radius=5"

    def apply(self, img):
        gray = self.to_grayscale_u8(img)
        return entropy(gray, disk(5)).astype(np.float32)

