from __future__ import annotations

from typing import Tuple


#Dependencies
import os
from skimage import io
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import exposure
import glob
import numpy as np
import cv2 as cv
from PIL import Image
from typing import Tuple, Dict, Callable, Optional
from pathlib import Path
import importlib.util
import sys
import inspect




#Functions

#function to generate Modified green blue vegetation index
def MGRVI (Dest,Origin):
  path = glob.glob(Origin+'*.jpg')
  y = len([name for name in os.listdir(Origin) 
	  if os.path.isfile(os.path.join(Origin, name))])
  z = 0
  print ('loading',z,'%')

  for archivo in path:
    file_name = archivo.split('/')[-1]
    image = cv.imread(archivo)
    B = image[:,:,0]
    G = image[:,:,1]
    R = image[:,:,2]

    # MGRVI = (G^2 − R^2)/(G^2 + R^2) transform
    MGRVI_view = (G**2 - R**2) / ((G**2 + R**2)+1e-10)

    norm = cv.normalize(
    MGRVI_view, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

	#saving
    cv.imwrite(Dest+file_name, norm) #change the values according to the lenght of the name characters of the image
  
    z = z+1

    x = (100 * (z/y))

    print ('loading ' + '%.2f' % x,' %')
  
  return

#function to generate Vegetativen
def VEG (Dest,Origin):
  path = glob.glob(Origin+'*.jpg')
  y = len([name for name in os.listdir(Origin) 
	  if os.path.isfile(os.path.join(Origin, name))])
  z = 0
  print ('loading',z,'%')

  a = 0.667
  for archivo in path:
    file_name = archivo.split('/')[-1]
    image = cv.imread(archivo)
    B = image[:,:,0]
    G = image[:,:,1]
    R = image[:,:,2]

    # VEG =  G/(R^a * B^(1 − a)) a = 0.667 transform
    VEG_view = G/((R**a * B**(1-a))+1e-10)

    norm = cv.normalize(
    VEG_view, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

	#saving
    cv.imwrite(Dest+file_name, norm) #change the values according to the lenght of the name characters of the image
  
    z = z+1

    x = (100 * (z/y))

    print ('loading ' + '%.2f' % x,' %')
  
  return

#function to generate Green–red ratio index
def GRRI (Dest,Origin):
  path = glob.glob(Origin+'*.jpg')
  y = len([name for name in os.listdir(Origin) 
	  if os.path.isfile(os.path.join(Origin, name))])
  z = 0
  print ('loading',z,'%')

  for archivo in path:
    file_name = archivo.split('/')[-1]
    image = cv.imread(archivo)
    B = image[:,:,0]
    G = image[:,:,1]
    R = image[:,:,2]

    # GRRI = G/R transform
    GRRI_view = G / (R+1e-10)

    norm = cv.normalize(
    GRRI_view, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

	#saving
    cv.imwrite(Dest+file_name, norm) #change the values according to the lenght of the name characters of the image
  
    z = z+1

    x = (100 * (z/y))

    print ('loading ' + '%.2f' % x,' %')
  
  return

#function to generate Visible atmospherically resistant index
def VARI (Dest,Origin):
  path = glob.glob(Origin+'*.jpg')
  y = len([name for name in os.listdir(Origin) 
	  if os.path.isfile(os.path.join(Origin, name))])
  z = 0
  print ('loading',z,'%')

  for archivo in path:
    file_name = archivo.split('/')[-1]
    image = cv.imread(archivo)
    B = image[:,:,0]
    G = image[:,:,1]
    R = image[:,:,2]

    # VARI = (G − R)/(G + R – B) transform
    VARI_view = (G-R)/((G+R-B)+1e-10)

    norm = cv.normalize(
    VARI_view, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

	#saving
    cv.imwrite(Dest+file_name, norm) #change the values according to the lenght of the name characters of the image
  
    z = z+1

    x = (100 * (z/y))

    print ('loading ' + '%.2f' % x,' %')
  
  return

#function to generate Normalized green–red difference index
def NGRDI (Dest,Origin):
  path = glob.glob(Origin+'*.jpg')
  y = len([name for name in os.listdir(Origin) 
	  if os.path.isfile(os.path.join(Origin, name))])
  z = 0
  print ('loading',z,'%')

  for archivo in path:
    file_name = archivo.split('/')[-1]
    image = cv.imread(archivo)
    B = image[:,:,0]
    G = image[:,:,1]
    R = image[:,:,2]

    # NGRDI = (G − R)/(G + R) transform
    NGRDI_view = (G - R) / ((G + R)+1e-10)

    norm = cv.normalize(
    NGRDI_view, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

	#saving
    cv.imwrite(Dest+file_name, norm) #change the values according to the lenght of the name characters of the image
  
    z = z+1

    x = (100 * (z/y))

    print ('loading ' + '%.2f' % x,' %')
  
  return

#function to generate Visible-band difference vegetation index
def VDVI (Dest,Origin):
  path = glob.glob(Origin+'*.jpg')
  y = len([name for name in os.listdir(Origin) 
	  if os.path.isfile(os.path.join(Origin, name))])
  z = 0
  print ('loading',z,'%')

  for archivo in path:
    file_name = archivo.split('/')[-1]
    image = cv.imread(archivo)
    B = image[:,:,0]
    G = image[:,:,1]
    R = image[:,:,2]

    # VDVI = (2 * G − R − B)/(2 * G + R + B) transform
    VDVI_view = (2 * G - R - B) / ((2*G + R + B)+1e-10)

    norm = cv.normalize(
    VDVI_view, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    #VDVI_view = Image.fromarray(VDVI_view)

	#saving
    cv.imwrite(Dest+file_name, norm) #change the values according to the lenght of the name characters of the image
  
    z = z+1

    x = (100 * (z/y))

    print ('loading ' + '%.2f' % x,' %')
  
  return



#function to generate HOG descriptor
def hog_des (Dest,Origin):
  path = glob.glob(Origin+'*.jpg')
  y = len([name for name in os.listdir(Origin) 
	  if os.path.isfile(os.path.join(Origin, name))])
  z = 0
  print ('loading',z,'%')

  for archivo in path:
    file_name = archivo.split('/')[-1]
    image = cv.imread(archivo,0)
    # HOG tranform
    fd, HOG_view = hog(
      image,
      orientations=9,
      pixels_per_cell=(8, 8),
      cells_per_block=(2, 2),
      visualize=True,
    )

    norm = cv.normalize(
    HOG_view, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

	#saving
    cv.imwrite(Dest+file_name, norm) #change the values according to the lenght of the name characters of the image
  
    z = z+1

    x = (100 * (z/y))

    print ('loading ' + '%.2f' % x,' %')
  
  return

#function to generate LBP descriptor
def lbp_des (Dest,Origin):
  path = glob.glob(Origin+'*.jpg')
  y = len([name for name in os.listdir(Origin) 
	  if os.path.isfile(os.path.join(Origin, name))])
  z = 0
  print ('loading',z,'%')
  # settings for LBP
  radius = 3
  n_points = 8 * radius
  for archivo in path:
    file_name = archivo.split('/')[-1]
    image = cv.imread(archivo,0)

    LBP_view = local_binary_pattern(image, n_points, radius,'var')

    norm = cv.normalize(
    LBP_view, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

	  #saving
    cv.imwrite(Dest + file_name, norm)#change the values according to the lenght of the name characters of the image
  
    z = z+1

    x = (100 * (z/y))

    print ('loading ' + '%.2f' % x,' %')
  
  return

#function to generate entropy descriptor
def ent_des (Dest,Origin):
  path = glob.glob(Origin+'*.jpg')
  y = len([name for name in os.listdir(Origin) 
	  if os.path.isfile(os.path.join(Origin, name))])
  z = 0
  print ('loading',z,'%')
  
  for archivo in path:
    file_name = archivo.split('/')[-1]
    image = cv.imread(archivo,0)

    Entropy_view = entropy(image, disk(5))

    norm = cv.normalize(
    Entropy_view, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

	  #saving
    cv.imwrite(Dest + file_name, norm)#change the values according to the lenght of the name characters of the image
  
    z = z+1

    x = (100 * (z/y))

    print ('loading ' + '%.2f' % x,' %')
  
  return

# ---------------------------------------------------------------------------
# Re-exports: cropping/resize are implemented in image_ops.py (separation of concerns)
from .image_ops import resize_image, dynamic_crop  # noqa: F401


# ---------------------------------------------------------------------------
# Object-oriented transform API: descriptions, registry, and normalization
# ---------------------------------------------------------------------------

def _pil_to_rgb_np(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    return arr.astype(np.float32)


def _to_grayscale_u8(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"))


def _normalize_to_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m, M = float(np.min(x)), float(np.max(x))
    if not np.isfinite(m) or not np.isfinite(M):
        return np.zeros_like(x, dtype=np.uint8)
    if abs(M - m) < 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - m) / (M - m)
    y = np.clip(y * 255.0, 0, 255)
    return y.astype(np.uint8)


from .transform_base import Transform  # canonical base class


def _load_module_from_path(mod_name: str, path: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _discover_transforms() -> Dict[str, Transform]:
    """Dynamically load Transform subclasses from ./transforms/*.py files.

    Each file should define at least one subclass of databaseMLUtils.transform_base.Transform
    with a unique non-empty `id`.
    """
    here = Path(__file__).resolve().parent
    folder = here / "transforms"
    registry: Dict[str, Transform] = {}
    if not folder.exists():
        return registry
    for py in sorted(folder.glob("*.py")):
        if py.name.startswith("_"):
            continue
        if py.stem in {"__init__", "transformDefinition"}:
            continue
        mod_name = f"databaseMLUtils._dyn_transforms.{py.stem}"
        try:
            mod = _load_module_from_path(mod_name, str(py))
        except Exception:
            continue
        if mod is None:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if not issubclass(obj, Transform):
                continue
            if obj is Transform:
                continue
            try:
                inst = obj()
            except Exception:
                continue
            if not getattr(inst, "id", None):
                continue
            registry[str(inst.id)] = inst
    return registry


_TRANSFORMS_REGISTRY: Dict[str, Transform] = _discover_transforms()


def get_transforms_registry() -> Dict[str, Transform]:
    """Return mapping from transform id -> Transform instance."""
    return dict(_TRANSFORMS_REGISTRY)


def get_transform_description(t_id: str) -> Optional[str]:
    t = _TRANSFORMS_REGISTRY.get(t_id)
    return t.description if t else None
