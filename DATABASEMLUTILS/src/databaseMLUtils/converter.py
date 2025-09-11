import os
import glob
import shutil
from pathlib import Path
from typing import List, Tuple

import xmltodict
from PIL import Image
from databaseMLUtils.transforms import Transformer


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _rect_crop(img: Image.Image, xmin: int, ymin: int, xmax: int, ymax: int) -> Image.Image:
    """Recorte rectangular con límites a la imagen (sin hacerlo cuadrado)."""
    W, H = img.size

    x0 = max(0, min(int(xmin), int(xmax)))
    x1 = min(W, max(int(xmin), int(xmax)))
    y0 = max(0, min(int(ymin), int(ymax)))
    y1 = min(H, max(int(ymin), int(ymax)))

    if x1 <= x0 or y1 <= y0:
        # bbox inválido: devuelve imagen completa como fallback
        return img.copy()

    return img.crop((x0, y0, x1, y1))


def convert_xml_to_Classification(
    inputUri: str,
    outputDir: str,
    List_transforms_ids: List[str] = ('RGB',),
    image_exts: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
    test: bool = False
) -> bool:
    """
    Estructura de salida:
      outputDir/
        <view_id>/
          <class_name>/
            <archivo>.jpg

    - Lee *.xml estilo Pascal VOC desde inputUri
    - Para cada objeto en el XML, recorta el bbox (rectangular)
    - Aplica transforms al crop
    - Guarda en views/<clase>
    - Si test=True, procesa solo las primeras 10 imágenes
    """
    in_dir = Path(inputUri)
    out_dir = Path(outputDir)

    # Preparar salida
    if out_dir.exists():
        shutil.rmtree(out_dir)
    _ensure_dir(out_dir)

    # Transformer
    transformer = Transformer()
    transformer.print()

    xml_files = sorted(in_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files in '{in_dir}'")

    # Limitar a 10 si test=True
    if test:
        xml_files = xml_files[:10]
        print("[INFO] Running in TEST mode: only first 10 XML files will be processed.")

    for xml_path in xml_files:
        try:
            with open(xml_path, "r", encoding="utf-8") as f:
                ann = xmltodict.parse(f.read()).get("annotation", {})
        except Exception as e:
            print(f"[WARN] Cannot parse {xml_path.name}: {e}")
            continue

        filename = ann.get("filename")
        if not filename:
            print(f"[WARN] No 'filename' in {xml_path.name}, skipping.")
            continue

        # Resolver ruta a la imagen
        img_path = in_dir / filename
        if not img_path.exists():
            stem = Path(filename).stem
            candidates = [in_dir / f"{stem}{ext}" for ext in image_exts]
            found = [c for c in candidates if c.exists()]
            if found:
                img_path = found[0]
            else:
                print(f"[WARN] Image '{filename}' not found for {xml_path.name}, skipping.")
                continue

        # Abrir imagen
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open image {img_path.name}: {e}")
            continue

        # Normalizar objects a lista
        objects = ann.get("object", [])
        if isinstance(objects, dict):
            objects = [objects]
        if not objects:
            print(f"[INFO] No objects in {xml_path.name}, skipping.")
            continue

        print(f"Processing '{img_path.name}': {len(objects)} objects")

        # Iterar objetos
        for i, obj in enumerate(objects, start=1):
            label = str(obj.get("name", "unknown")).strip() or "unknown"
            bbox = obj.get("bndbox", {})
            try:
                xmin = int(float(bbox.get("xmin", 0)))
                ymin = int(float(bbox.get("ymin", 0)))
                xmax = int(float(bbox.get("xmax", 0)))
                ymax = int(float(bbox.get("ymax", 0)))
            except Exception:
                print(f"[WARN] Invalid bbox in {xml_path.name}, object {i}, skipping.")
                continue

            # Recorte rectangular (sin cuadrado)
            crop = _rect_crop(img, xmin, ymin, xmax, ymax)

            # Aplicar transforms AL CROP
            transformed = transformer.apply_transforms(List_transforms_ids, crop)

            # Guardar por vista/clase
            base_name = f"{img_path.stem}_{label}_{i:03d}"
            for view_id, crop_img in transformed:
                class_dir = out_dir / str(view_id) / label
                _ensure_dir(class_dir)
                out_path = class_dir / f"{base_name}.jpg"
                try:
                    crop_img.save(out_path, format="JPEG", quality=100)
                except Exception as e:
                    print(f"[ERROR] Saving {out_path.name}: {e}")

    print("Done.")
    return True
