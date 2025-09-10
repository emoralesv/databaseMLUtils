from __future__ import annotations

import numpy as np
from .transform_base import Transform


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
        return super()._normalize(x)

    
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
        return super()._normalize(x)
    
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
        return super()._normalize(x)
    
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
        return super()._normalize(x)
    
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
        return super()._normalize(x)
    
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
        return super()._normalize(x)




