from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Optional
import importlib
import sys
import PIL.Image as Image
from typing import List
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image
import numpy as np

from .views.transform_base import Transform  



def _load_module_from_path(mod_name: str, path: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class Transformer():
    
    def __init__(self):
        _TRANSFORMS_REGISTRY: Dict[str, Transform] = self._discover_transforms()
        self.transforms = _TRANSFORMS_REGISTRY
        self.transform_ids = sorted(self.transforms.keys())
        self.transforms_descriptions = [self.transforms[t_id].description for t_id in self.transform_ids]
    def print(self):
        print("Available transforms:")
        for t_id in self.transform_ids:
            t = self.transforms[t_id]
            print(f"  {t.id}: {t.description}")
        
    def apply_transforms(self,ids, pil_img):
        """Apply multiple transforms to a single PIL image and return [(id, out_img), ...]."""
        results = []
        registry = self.transforms
        for t_id in ids:
            if t_id not in registry:
                print(f"[WARN] Transform '{t_id}' not found, skipping.")
                continue
            transform = registry[t_id]
            out_img = transform(pil_img)   
            results.append((t_id, out_img))
        return results

    def get_transform_description(self, t_id: str) -> Optional[str]:
        t = self.transforms.get(t_id)
        return t.description if t else None
    def get_transforms_registry(self) -> Dict[str, Transform]:
        """Return mapping from transform id -> Transform instance."""
        return dict(self.transforms)
    @staticmethod
    def _discover_transforms() -> Dict[str, Transform]:
      here = Path(__file__).resolve().parent
      folder = here / "views"
      registry: Dict[str, Transform] = {}
      if not folder.exists():
          return registry
      for py in sorted(folder.glob("*.py")):
          if py.name.startswith("_") or py.stem in {"__init__", "transformDefinition"}:
              continue
          mod_name = f"databaseMLUtils.views."
          try:
              mod = _load_module_from_path(mod_name, str(py))
          except Exception as e:
              print(f"Error {e} loading {py}")
              continue
          for _, obj in inspect.getmembers(mod, inspect.isclass):
              if not issubclass(obj, Transform):
                  continue
              if obj is Transform:
                  continue
              try:
                  inst = obj()
              except Exception as e:
                  print(f"Error {e} instantiating {obj}")
                  continue
              if not getattr(inst, "id", None):
                  continue
              registry[str(inst.id)] = inst
      return registry
    @staticmethod
    def showImages(images: List[Tuple[str, Image.Image]], ncols: int = 3) -> None:
        """
        Display a list of (transform_id, PIL.Image) in a grid.

        Parameters
        ----------
        images : list of (str, Image.Image)
            Pairs of transform id and image to display.
        ncols : int, default=3
            Number of columns in the grid.
        """
        n = len(images)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)  # flatten, even if nrows=1 or ncols=1

        for ax, (t_id, img) in zip(axes, images):
            if img.mode == "L":  # grayscale
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(t_id)
            ax.axis("off")

        # hide any unused subplots
        for ax in axes[len(images):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
