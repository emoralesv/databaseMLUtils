# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable, Optional
import json
import math

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
THUMB_SIZE = (256, 256)
THUMBS_PER_CLASS = 6
TOP_RESOLUTIONS = 10

# =========================
# Helpers
# =========================
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _normalize_url(url_str: str) -> str:
    """Si es ruta local existente, conviértela a file:// para que sea clicable en README."""
    try:
        p = Path(url_str)
        if p.exists():
            return p.resolve().as_uri()
    except Exception:
        pass
    return url_str

def _iter_class_images(root: Path) -> Iterable[Tuple[Path, str]]:
    """Itera (ruta_imagen, clase) asumiendo subcarpetas por clase directamente bajo root."""
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cls = class_dir.name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p, cls

def _detect_splits(root: Path) -> Dict[str, Path]:
    """Detecta splits si existen (train/val/test). Devuelve dict {split: path}."""
    splits = {}
    for s in ("train", "val", "test"):
        p = root / s
        if p.exists() and p.is_dir():
            splits[s] = p
    return splits

def _safe_open(p: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(p)
        img.load()
        return img
    except Exception:
        return None

def _is_grayscale(img: Image.Image) -> bool:
    if img.mode in ("L", "1"):
        return True
    if img.mode in ("RGB", "RGBA"):
        arr = np.asarray(img.convert("RGB"))
        diff = np.abs(arr[..., 0].astype(np.int32) - arr[..., 1].astype(np.int32)) \
             + np.abs(arr[..., 1].astype(np.int32) - arr[..., 2].astype(np.int32))
        return diff.mean() < 1.0
    return False

def _make_thumb(img: Image.Image, size=THUMB_SIZE) -> Image.Image:
    return ImageOps.fit(img.convert("RGB"), size, Image.Resampling.LANCZOS)

def _phash(img: Image.Image, hash_size: int = 8) -> str:
    """Perceptual hash simple (no DCT real, pero útil para duplicados obvios)."""
    try:
        small = ImageOps.exif_transpose(img).convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        arr = np.asarray(small, dtype=np.float32)
        avg = arr.mean()
        bits = arr > avg
        return "".join("1" if b else "0" for b in bits.flatten())
    except Exception:
        return ""

def _human_int(n: int) -> str:
    return f"{n:,}".replace(",", "_").replace("_", ",")

# =========================
# Análisis núcleo
# =========================
def _analyze_folder(folder: Path, sample_per_class=THUMBS_PER_CLASS) -> Dict:
    """Analiza un conjunto (sin splits). Devuelve estadísticas y muestras por clase."""
    per_class = Counter()
    widths, heights, ratios = [], [], []
    resol_counter = Counter()
    color_counter = Counter()
    samples = []  # [(class, path)]
    corrupted = 0
    dup_hashes = Counter()

    for p, cls in _iter_class_images(folder):
        img = _safe_open(p)
        if img is None:
            corrupted += 1
            continue

        w, h = img.size
        widths.append(w); heights.append(h)
        ratios.append((w / h) if h else 0.0)
        resol_counter[(w, h)] += 1
        color_counter["gray" if _is_grayscale(img) else "color"] += 1

        per_class[cls] += 1
        if per_class[cls] <= sample_per_class:
            samples.append((cls, p))

        hcode = _phash(img)
        if hcode:
            dup_hashes[hcode] += 1

    probable_dups = sum(c - 1 for c in dup_hashes.values() if c > 1)

    return {
        "classes": sorted(per_class.keys()),
        "per_class_counts": dict(per_class),
        "images_count": int(sum(per_class.values())),
        "corrupted_images": int(corrupted),
        "widths": widths,
        "heights": heights,
        "ratios": ratios,
        "resolution_top": [[f"{w}x{h}", c] for (w, h), c in resol_counter.most_common(TOP_RESOLUTIONS)],
        "color_summary": dict(color_counter),
        "estimated_duplicate_images": int(probable_dups),
        "samples": samples,
    }

def _stats_block(nums: List[float]) -> Dict[str, Optional[float]]:
    if not nums:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(np.min(nums)),
        "max": float(np.max(nums)),
        "mean": float(np.mean(nums)),
        "median": float(np.median(nums)),
    }

def _aggregate_meta(meta: Dict) -> Dict:
    ws = _stats_block(meta["widths"])
    hs = _stats_block(meta["heights"])
    ars = _stats_block(meta["ratios"])
    return {
        **meta,
        "width_stats": ws,
        "height_stats": hs,
        "aspect_ratio_stats": ars,
    }

def _analyze_dataset(root: Path, sample_per_class=THUMBS_PER_CLASS, include_splits=True) -> Dict:
    """
    Si detecta splits (train/val/test) y include_splits=True, devuelve stats globales y por split.
    """
    splits = _detect_splits(root) if include_splits else {}
    if not splits:
        meta = _analyze_folder(root, sample_per_class)
        return {
            "global": _aggregate_meta(meta),
            "splits": {}
        }

    out = {"splits": {}, "global": None}
    # Por split
    global_counts = Counter()
    global_samples = []
    widths_all, heights_all, ratios_all = [], [], []
    color_all = Counter()
    resol_all = Counter()
    corrupted_all = 0
    dup_all = 0

    for sname, spath in splits.items():
        m = _analyze_folder(spath, sample_per_class)
        out["splits"][sname] = _aggregate_meta(m)

        # Acumular a global
        global_counts.update(m["per_class_counts"])
        widths_all += m["widths"]; heights_all += m["heights"]; ratios_all += m["ratios"]
        for (wh, cnt) in m["resolution_top"]:
            # resolution_top es top-k local; para global es mejor contar directo:
            pass  # lo recalcularemos globalmente
        color_all.update(m["color_summary"])
        corrupted_all += m["corrupted_images"]
        dup_all += m["estimated_duplicate_images"]
        global_samples += m["samples"]

    # Recontar resoluciones globales escaneando todo (para exactitud):
    resol_counter_global = Counter()
    for p, _ in _iter_class_images(root):
        img = _safe_open(p)
        if img is None:
            continue
        W, H = img.size
        resol_counter_global[(W, H)] += 1

    global_meta = {
        "classes": sorted(global_counts.keys()),
        "per_class_counts": dict(global_counts),
        "images_count": int(sum(global_counts.values())),
        "corrupted_images": int(corrupted_all),
        "widths": widths_all,
        "heights": heights_all,
        "ratios": ratios_all,
        "resolution_top": [[f"{w}x{h}", c] for (w, h), c in resol_counter_global.most_common(TOP_RESOLUTIONS)],
        "color_summary": dict(color_all),
        "estimated_duplicate_images": int(dup_all),
        "samples": global_samples,
    }
    out["global"] = _aggregate_meta(global_meta)
    return out

# =========================
# Visualizaciones
# =========================
def _plot_class_distribution(per_class: Dict[str, int], out_png: Path):
    if not per_class:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(list(per_class.keys()), list(per_class.values()))
    plt.xticks(rotation=45, ha="right")
    plt.title("Class distribution")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def _plot_aspect_ratio_hist(ratios: List[float], out_png: Path):
    if not ratios:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(ratios, bins=30)
    plt.title("Aspect ratio (W/H)")
    plt.xlabel("Aspect ratio")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def _build_thumbs(meta: Dict, out_dir: Path, per_class_limit=THUMBS_PER_CLASS) -> Dict[str, List[str]]:
    thumbs_map = defaultdict(list)
    thumbs_root = out_dir / "thumbs"
    thumbs_root.mkdir(parents=True, exist_ok=True)
    seen = Counter()
    # Usamos muestras del meta global
    for cls, p in meta["global"]["samples"]:
        if seen[cls] >= per_class_limit:
            continue
        img = _safe_open(p)
        if img is None:
            continue
        th = _make_thumb(img)
        dst = thumbs_root / f"{cls}_{seen[cls]+1:02d}.jpg"
        th.save(dst, "JPEG", quality=92)
        thumbs_map[cls].append(str(dst.relative_to(out_dir)))
        seen[cls] += 1
    return thumbs_map

# =========================
# Writers
# =========================
def _write_json(meta_out: Path, payload: Dict):
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _write_readme(
    readme_path: Path,
    dataset_name: str,
    url: str,
    meta: Dict,
    thumbs_map: Dict[str, List[str]],
    class_dist_png: Optional[str],
    aspect_png: Optional[str],
):
    g = meta["global"]
    lines = []
    lines.append(f"# {dataset_name}\n")
    lines.append("## Summary\n")
    lines.append(f"- **Task:** classification\n")
    lines.append(f"- **Images:** { _human_int(g['images_count']) }\n")
    lines.append(f"- **Classes:** { len(g['classes']) }\n")
    lines.append(f"- **URL:** {url}\n")

    # Splits
    if meta["splits"]:
        lines.append("\n## Splits")
        for sname, m in meta["splits"].items():
            lines.append(f"- **{sname}**: { _human_int(m['images_count']) } images")

    # Classes
    lines.append("\n## Classes")
    for c in g["classes"]:
        lines.append(f"- {c}: {g['per_class_counts'].get(c, 0)}")

    # Stats
    lines.append("\n## Counts & Stats")
    if g["corrupted_images"]:
        lines.append(f"- Corrupted images: **{g['corrupted_images']}**")
    lines.append(f"- Estimated duplicate images: **{g['estimated_duplicate_images']}**")
    lines.append(f"- Color summary: {g['color_summary']}")
    lines.append(f"- Top resolutions (WxH, count): {g['resolution_top']}")
    ws, hs, ars = g["width_stats"], g["height_stats"], g["aspect_ratio_stats"]
    def fmt(x, nd=1):
        return ("NA" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x)))
    lines.append(f"- Width [min/mean/median/max]: {fmt(ws['min'])}/{fmt(ws['mean'])}/{fmt(ws['median'])}/{fmt(ws['max'])}")
    lines.append(f"- Height [min/mean/median/max]: {fmt(hs['min'])}/{fmt(hs['mean'])}/{fmt(hs['median'])}/{fmt(hs['max'])}")
    lines.append(f"- Aspect ratio [min/mean/median/max]: {fmt(ars['min'],3)}/{fmt(ars['mean'],3)}/{fmt(ars['median'],3)}/{fmt(ars['max'],3)}")

    # Plots
    if class_dist_png:
        lines.append("\n## Class distribution")
        lines.append(f"![Class distribution]({class_dist_png})\n")
    if aspect_png:
        lines.append("## Aspect ratio")
        lines.append(f"![Aspect ratio histogram]({aspect_png})\n")

    # Thumbs
    lines.append("## Examples (thumbnails)")
    for cls in g["classes"]:
        thumbs = thumbs_map.get(cls, [])
        if not thumbs:
            continue
        lines.append(f"**{cls}**")
        row = " ".join(f"![]({t})" for t in thumbs)
        lines.append(row + "\n")

    lines.append("## Notes")
    lines.append("- Add license information here if available.")
    lines.append("- Folder structure follows 'one folder per class'.")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# =========================
# API principal
# =========================
def make_dataset_report(
    data: str,
    name: str,
    url: str,
    out: str,
    samples_per_class: int = THUMBS_PER_CLASS,
    include_splits: bool = True,
    test: bool = False,              # si True, procesa solo primeras 10 clases o corta imágenes (rápido)
):
    data_root = Path(data)
    out_dir = Path(out)
    _ensure_dir(out_dir)

    url = _normalize_url(url)

    # (Opcional) modo test: sub-muestreo rápido de clases (para datasets enormes)
    if test:
        # construir un "mini-root" temporal con enlaces relativos (o solo limitar en lectura)
        # aquí limitamos en lectura: filtraremos en _build_thumbs y en plots no hay riesgo
        print("[INFO] TEST mode: stats completas, pero thumbs limitados a 2/cls y sin top resoluciones grandes.")
        global THUMBS_PER_CLASS
        THUMBS_PER_CLASS_BACKUP = THUMBS_PER_CLASS
        THUMBS_PER_CLASS = min(2, samples_per_class)

    meta = _analyze_dataset(data_root, sample_per_class=samples_per_class, include_splits=include_splits)

    # Plots (globales)
    class_png = out_dir / "class_distribution.png"
    aspect_png = out_dir / "aspect_ratio_hist.png"
    _plot_class_distribution(meta["global"]["per_class_counts"], class_png)
    _plot_aspect_ratio_hist(meta["global"]["ratios"], aspect_png)

    # Thumbs (globales)
    thumbs_map = _build_thumbs(meta, out_dir, per_class_limit=samples_per_class)

    # JSON
    payload = {
        "name": name,
        "url": url,
        "task": "classification",
        "images_count": meta["global"]["images_count"],
        "classes": meta["global"]["classes"],
        "per_class_counts": meta["global"]["per_class_counts"],
        "color_summary": meta["global"]["color_summary"],
        "resolution_top": meta["global"]["resolution_top"],
        "width_stats": meta["global"]["width_stats"],
        "height_stats": meta["global"]["height_stats"],
        "aspect_ratio_stats": meta["global"]["aspect_ratio_stats"],
        "estimated_duplicate_images": meta["global"]["estimated_duplicate_images"],
        "corrupted_images": meta["global"]["corrupted_images"],
        "has_splits": bool(meta["splits"]),
        "splits": {
            sname: {
                "images_count": m["images_count"],
                "per_class_counts": m["per_class_counts"],  # opcional: útil si las clases difieren por split
            } for sname, m in meta["splits"].items()
        }
    }
    _write_json(out_dir / "dataset_meta.json", payload)

    # README
    _write_readme(
        out_dir / "README.md",
        dataset_name=name,
        url=url,
        meta=meta,
        thumbs_map=thumbs_map,
        class_dist_png=str(class_png.relative_to(out_dir)) if class_png.exists() else None,
        aspect_png=str(aspect_png.relative_to(out_dir)) if aspect_png.exists() else None,
    )

    print(f"[OK] Wrote README.md and dataset_meta.json to: {out_dir}")

    # Restaurar THUMBS_PER_CLASS si test
    if test:
        THUMBS_PER_CLASS = THUMBS_PER_CLASS_BACKUP
