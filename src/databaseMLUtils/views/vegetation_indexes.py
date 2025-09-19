from __future__ import annotations

import numpy as np
from .transform_base import Transform
def show_histogram(img_array, title="Histogram", bins=100, log_scale=False):
    import matplotlib.pyplot as plt
    """
    Plot histogram of an image (NumPy array).
    
    Parameters
    ----------
    img_array : np.ndarray
        Input 2D array (image or feature map).
    title : str
        Plot title.
    bins : int
        Number of bins in histogram.
    log_scale : bool
        If True, use log scale on y-axis.
    """
    plt.figure(figsize=(6,4))
    plt.hist(img_array.ravel(), bins=bins, color="gray", alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    if log_scale:
        plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

class VEGTransform(Transform):
    id = "VEG"
    description = "G / (R^a * B^(1-a)), a=0.667"

    def _apply(self, img):
        rgb = self._pil_to_rgb_np(img)
        R = rgb[..., 0]
        G = rgb[..., 1]
        B = rgb[..., 2]
        a = 0.667
        eps = 1e-10
        return G / ((R ** a) * (B ** (1 - a)) + eps)
    def _normalize(self, x):
        x_clipped = np.clip(x, 0, 3)
        x_norm = (x_clipped / 3).astype(np.float32)
        return x_norm

    
class VDVITransform(Transform):
    id = "VDVI"
    description = "Visible-band Difference Vegetation Index: (2G − R − B)/(2G + R + B)"

    def _apply(self, img):
        rgb = self._pil_to_rgb_np(img)
        R = rgb[..., 0]
        G = rgb[..., 1]
        B = rgb[..., 2]
        eps = 1e-10
        return (2 * G - R - B) / ((2 * G + R + B) + eps)
    def _normalize(self, x):
        return ((x + 1) / 2).astype(np.float32)
    
class VARITransform(Transform):
    id = "VARI"
    description = "Visible Atmospherically Resistant Index: (G − R)/(G + R − B)"

    def _apply(self, img):
        rgb = self._pil_to_rgb_np(img)
        R = rgb[..., 0]
        G = rgb[..., 1]
        B = rgb[..., 2]
        eps = 1e-10
        return (G - R) / ((G + R - B) + eps)
    def _normalize(self, x):
        return ((x + 1) / 2).astype(np.float32)
    
class NGRDITransform(Transform):
    id = "NGRDI"
    description = "Normalized Green-Red Difference Index: (G − R)/(G + R)"

    def _apply(self, img):
        rgb = self._pil_to_rgb_np(img)
        R = rgb[..., 0]
        G = rgb[..., 1]
        eps = 1e-10
        return (G - R) / ((G + R) + eps)
    def _normalize(self, x):
        return ((x + 1) / 2).astype(np.float32)
    
class MGRVITransform(Transform):
    id = "MGRVI"
    description = "Modified Green-Red Vegetation Index: (G^2 − R^2)/(G^2 + R^2)"

    def _apply(self, img):
        rgb = self._pil_to_rgb_np(img)
        R = rgb[..., 0]
        G = rgb[..., 1]
        eps = 1e-10
        return (G ** 2 - R ** 2) / ((G ** 2 + R ** 2) + eps)
    def _normalize(self, x):
        return ((x + 1) / 2).astype(np.float32)
    
class GRRITransform(Transform):
    id = "GRRI"
    description = "Green-Red Ratio Index: G / R"

    def _apply(self, img):
        
        rgb = self._pil_to_rgb_np(img)
        R = rgb[..., 0]
        G = rgb[..., 1]
        eps = 1e-10
        return G / (R + eps)
    def _normalize(self, x):
        x_clipped = np.clip(x, 0, 3)
        x_norm = (x_clipped / 3).astype(np.float32)
        return x_norm





