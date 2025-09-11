from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from PIL import Image


class Transform(ABC):
    """
    Prototype for per-image transforms that return a normalized, single-channel image.

    Usage (pipeline):
      PIL.Image -> _pil_to_rgb_np(PIL.Image) -> _apply(np.ndarray) -> _normalize(np.ndarray in [0,1], HxW)
      -> _normalize_to_u8(np.ndarray) -> PIL.Image (mode="L")

    Subclasses must implement `_apply`. They may override `_normalize` when the output
    of `_apply` is not already single-channel float in [0,1].
    """

    id: str = ""
    description: str = ""

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Run the transform pipeline on a PIL image and return a single-channel (L) PIL image.
        """
        view = self._apply(img)                  # subclass-defined intermediate view
        view = self._normalize(view)             # expected float32 in [0,1], shape (H, W)
        norm = self._normalize_to_u8(view)       # uint8 in [0,255], shape (H, W)
        return Image.fromarray(norm, mode="L")  if norm.ndim < 3 else Image.fromarray(norm, mode="RGB")

    # ---------- Hooks to be implemented/overridden by subclasses ----------

    @abstractmethod
    def _apply(self, x: np.ndarray) -> np.ndarray:
        """
        Produce an intermediate representation from the input RGB array.

        Parameters
        ----------
        x : np.ndarray
            RGB image as float32 in [0,1], shape (H, W, 3).

        Returns
        -------
        np.ndarray
            An intermediate representation (any dtype/shape), to be standardized by `_normalize`.
        """
        raise NotImplementedError

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize the intermediate representation to single-channel float32 in [0,1].

        Default behavior is a no-op. Override in subclasses when `_apply` does not already
        return a single-channel (H, W) float array in [0,1].

        Parameters
        ----------
        x : np.ndarray
            Intermediate array from `_apply`.

        Returns
        -------
        np.ndarray
            Single-channel float32 array in [0,1], shape (H, W).
        """
        return x

    # ------------------------------- Utilities -------------------------------

    @staticmethod
    def _pil_to_rgb_np(img: Image.Image) -> np.ndarray:
        """
        Convert a PIL image to an RGB numpy array in float32 normalized to [0,1].

        Returns
        -------
        np.ndarray
            Array of shape (H, W, 3), dtype float32, values in [0,1].
        """
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        return arr / 255.0

    @staticmethod
    def _to_grayscale_u8(img: Image.Image) -> np.ndarray:
        """
        Convert a PIL image to an 8-bit grayscale numpy array.

        Returns
        -------
        np.ndarray
            Array of shape (H, W), dtype uint8, values in [0,255].
        """
        return np.asarray(img.convert("L"))

    @staticmethod
    def _normalize_to_u8(x: np.ndarray) -> np.ndarray:
        """
        Assume input is a float array in [0,1] and scale to uint8 [0,255].

        Parameters
        ----------
        x : np.ndarray
            Float array in [0,1], shape (H, W).

        Returns
        -------
        np.ndarray
            uint8 array in [0,255], shape (H, W).
        """
        y = np.clip(x, 0.0, 1.0) * 255.0
        return y.astype(np.uint8)
