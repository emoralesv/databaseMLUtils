from .transform_base import Transform
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import hog

class RGBTransform(Transform):
    id = "RGB"
    description = "No transform"
    def __call__(self, img):
        return img
    def _apply(self):
        pass
    def _normalize(self):
        pass

class LBPTransform(Transform):
    id = "LBP"
    description = "Local Binary Pattern (var) with radius=3"

    def _apply(self, img):
        gray = self._to_grayscale_u8(img)
        radius = 3
        self.n_points = 3 * radius
        return local_binary_pattern(gray, self.n_points, radius).astype(np.float32)
    def _normalize(self, x):
        return  x / float(2**self.n_points - 1)  # values in [0,1]
    
class EntropyTransform(Transform):
    id = "ENTROPY"
    description = "Local entropy with disk radius=5"

    def _apply(self, img):
        gray = self._to_grayscale_u8(img)
        return entropy(gray, disk(5)).astype(np.float32)
    def _normalize(self, x):
        return super()._normalize(x)
    


class HOGTransform(Transform):
    id = "HOG"
    description = "Histogram of Oriented Gradients visualization"

    def _apply(self, img):
        gray = self._to_grayscale_u8(img)
        _, view = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
        )
        return view.astype(np.float32)
    def _normalize(self, x):
        return super()._normalize(x)