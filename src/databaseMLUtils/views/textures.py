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
    def __init__(self, radius: int = 10, base: int = 2):
        super().__init__()
        self.radius = radius
        self.base = base
        self._fp = disk(self.radius)
        self._max_theoretical = np.log(self._fp.sum()) / np.log(self.base)

    def _apply(self, img):
        gray = self._to_grayscale_u8(img)
        gray = gray.copy()  # asegura writable
        return entropy(gray, self._fp).astype(np.float32)

    def _normalize(self, x):
        # Normaliza a [0,1] con el máximo teórico
        denom = self._max_theoretical if self._max_theoretical > 0 else max(float(x.max()), 1.0)
        x_norm = (x / denom).astype(np.float32, copy=True)
        # opcional: forzar valores >1 a 1
        # x_norm = np.minimum(x_norm, 1.0)
        return x_norm
    


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
