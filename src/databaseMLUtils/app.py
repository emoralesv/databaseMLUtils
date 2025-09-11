# app.py
# Streamlit DatasetNinja-style report builder for classification datasets

from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import io
import json

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------
# Config
# -------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
THUMB_SIZE = (256, 256)
TOP_RESOLUTIONS = 10

# -------------------------
# Low-level helpers
# -------------------------
def _iter_class_images(root: Path):
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cls = class_dir.name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p, cls

def _detect_splits(root: Path) -> Dict[str, Path]:
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
        diff = np.abs(arr[...,0].astype(np.int32) - arr[...,1].astype(np.int32)) \
             + np.abs(arr[...,1].astype(np.int32) - arr[...,2].astype(np.int32))
        return diff.mean() < 1.0
    return False

def _make_thumb(img: Image.Image, size=THUMB_SIZE) -> Image.Image:
    return ImageOps.fit(img.convert("RGB"), size, Image.Resampling.LANCZOS)

def _phash(img: Image.Image, hash_size: int = 8) -> str:
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

def _stats_block(nums: List[float]) -> Dict[str, Optional[float]]:
    if not nums:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(np.min(nums)),
        "max": float(np.max(nums)),
        "mean": float(np.mean(nums)),
        "median": float(np.median(nums)),
    }

# -------------------------
# Analysis
# -------------------------
def _analyze_folder(folder: Path, thumbs_per_class: int, test_limit_imgs: Optional[int] = None) -> Dict:
    per_class = Counter()
    widths, heights, ratios = [], [], []
    resol_counter = Counter()
    color_counter = Counter()
    samples = []  # [(class, path)]
    corrupted = 0
    dup_hashes = Counter()

    processed = 0
    for p, cls in _iter_class_images(folder):
        if test_limit_imgs is not None and processed >= test_limit_imgs:
            break

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
        # keep up to thumbs_per_class samples per class
        if sum(1 for c, _ in samples if c == cls) < thumbs_per_class:
            samples.append((cls, p))

        hcode = _phash(img)
        if hcode:
            dup_hashes[hcode] += 1

        processed += 1

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

@st.cache_data(show_spinner=False)
def analyze_dataset(root_str: str, thumbs_per_class: int = 6, include_splits: bool = True,
                    test_mode: bool = False, test_limit_imgs: int = 300) -> Dict:
    root = Path(root_str)
    splits = _detect_splits(root) if include_splits else {}
    if not splits:
        m = _analyze_folder(root, thumbs_per_class, test_limit_imgs if test_mode else None)
        return {"global": _aggregate_meta(m), "splits": {}}

    out = {"splits": {}, "global": None}
    # Per-split
    global_counts = Counter()
    widths_all, heights_all, ratios_all = [], [], []
    color_all = Counter()
    corrupted_all = 0
    dup_all = 0
    samples_all = []

    for sname, spath in splits.items():
        m = _analyze_folder(spath, thumbs_per_class, test_limit_imgs if test_mode else None)
        out["splits"][sname] = _aggregate_meta(m)

        global_counts.update(m["per_class_counts"])
        widths_all += m["widths"]; heights_all += m["heights"]; ratios_all += m["ratios"]
        color_all.update(m["color_summary"])
        corrupted_all += m["corrupted_images"]
        dup_all += m["estimated_duplicate_images"]
        samples_all += m["samples"]

    # recompute global resolutions precisely
    resol_counter_global = Counter()
    processed = 0
    for p, _ in _iter_class_images(root):
        if test_mode and processed >= test_limit_imgs:
            break
        img = _safe_open(p)
        if img is None:
            continue
        W, H = img.size
        resol_counter_global[(W, H)] += 1
        processed += 1

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
        "samples": samples_all,
    }
    out["global"] = _aggregate_meta(global_meta)
    return out

# -------------------------
# Plots (return PNG bytes for download or display)
# -------------------------
def plot_class_distribution(per_class: Dict[str, int]) -> Optional[bytes]:
    if not per_class:
        return None
    fig = plt.figure(figsize=(10, 5))
    plt.bar(list(per_class.keys()), list(per_class.values()))
    plt.xticks(rotation=45, ha="right")
    plt.title("Class distribution")
    plt.ylabel("Images")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def plot_aspect_ratio_hist(ratios: List[float]) -> Optional[bytes]:
    if not ratios:
        return None
    fig = plt.figure(figsize=(6, 4))
    plt.hist(ratios, bins=30)
    plt.title("Aspect ratio (W/H)")
    plt.xlabel("Aspect ratio")
    plt.ylabel("Count")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# -------------------------
# README & JSON builders
# -------------------------
def build_readme(dataset_name: str, url: str, meta: Dict, class_img_urls: Dict[str, List[str]],
                 objective: str, domain: str, source: str, usage_snippet: str,
                 class_descriptions: Dict[str, str]) -> str:
    g = meta["global"]
    def fmt(x, nd=1):
        return ("NA" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x)))

    lines = []
    lines.append(f"# {dataset_name}\n")

    # Overview / placeholders
    lines.append("## Overview\n")
    lines.append(f"> **Objective:** {objective or 'TODO: Describe the goal of this dataset.'}\n")
    lines.append(f"> **Domain:** {domain or 'TODO: Domain (e.g., Agriculture / Medical / Industrial).'}\n")
    lines.append(f"> **Source:** {source or 'TODO: Collection method, instruments, references.'}\n")

    # Summary
    lines.append("\n## Summary\n")
    lines.append(f"- **Task:** Classification\n")
    lines.append(f"- **Images:** { _human_int(g['images_count']) }\n")
    lines.append(f"- **Classes:** { len(g['classes']) }\n")
    lines.append(f"- **URL:** {url}\n")

    # Splits
    if meta["splits"]:
        lines.append("\n### Dataset Splits\n")
        for sname, m in meta["splits"].items():
            lines.append(f"- **{sname}**: { _human_int(m['images_count']) } images")

    # Structure
    lines.append("\n## Dataset Structure\n")
    lines.append("```\nroot/\n")
    if meta["splits"]:
        lines.append("  train/\n    class1/\n    class2/\n  val/\n    class1/\n    class2/\n  test/\n    class1/\n    class2/\n")
    else:
        lines.append("  class1/\n  class2/\n")
    lines.append("```\nEach class folder contains raw image files.\n")

    # Classes + descriptions
    lines.append("\n## Classes\n")
    for c in g["classes"]:
        desc = class_descriptions.get(c) or f"TODO: Describe {c}"
        lines.append(f"- **{c}** ({g['per_class_counts'].get(c, 0)} images) — {desc}")

    # Stats
    ws, hs, ars = g["width_stats"], g["height_stats"], g["aspect_ratio_stats"]
    lines.append("\n## Statistics\n")
    if g["corrupted_images"]:
        lines.append(f"- Corrupted images: **{g['corrupted_images']}**")
    lines.append(f"- Estimated duplicate images: **{g['estimated_duplicate_images']}**")
    lines.append(f"- Color summary: {g['color_summary']}")
    lines.append(f"- Top resolutions: {g['resolution_top']}")
    lines.append(f"- Width [min/mean/median/max]: {fmt(ws['min'])}/{fmt(ws['mean'])}/{fmt(ws['median'])}/{fmt(ws['max'])}")
    lines.append(f"- Height [min/mean/median/max]: {fmt(hs['min'])}/{fmt(hs['mean'])}/{fmt(hs['median'])}/{fmt(hs['max'])}")
    lines.append(f"- Aspect ratio [min/mean/median/max]: {fmt(ars['min'],3)}/{fmt(ars['mean'],3)}/{fmt(ars['median'],3)}/{fmt(ars['max'],3)}")

    # Visualizations (linked from app; images not embedded because we produce PNGs separately)
    lines.append("\n## Visualizations\n")
    lines.append("- Class distribution: `class_distribution.png`\n- Aspect ratio: `aspect_ratio_hist.png`\n")

    # Examples section (thumbs listed)
    lines.append("## Example Images\n")
    for cls in g["classes"]:
        thumbs = class_img_urls.get(cls, [])
        if not thumbs:
            continue
        lines.append(f"**{cls}**")
        row = " ".join(f"![]({t})" for t in thumbs)
        lines.append(row + "\n")

    # Usage / Citation / License
    lines.append("## Usage\n")
    lines.append("```python\n")
    lines.append((usage_snippet or "# TODO: Add code snippet to load/use this dataset (PyTorch, TF, etc.)\n").rstrip())
    lines.append("\n```\n")
    lines.append("## Citation\nIf you use this dataset, please cite:\n```\nTODO: BibTeX or reference\n```\n")
    lines.append("## License\nTODO: Add license information (MIT, CC-BY, proprietary, etc.)\n")

    return "\n".join(lines)

def build_json_payload(dataset_name: str, url: str, meta: Dict) -> Dict:
    return {
        "name": dataset_name,
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
                "per_class_counts": m["per_class_counts"],
            } for sname, m in meta["splits"].items()
        }
    }

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="DatasetNinja-Style Reporter", layout="wide")
st.title("🧰 DatasetNinja-Style Reporter (Classification)")

with st.sidebar:
    st.header("Settings")
    data_root = st.text_input("Dataset root (folder)", value="datos")
    dataset_name = st.text_input("Dataset name", value="My Dataset")
    dataset_url = st.text_input("Project URL (can be local path)", value=data_root)
    thumbs_per_class = st.number_input("Thumbnails per class", min_value=1, max_value=20, value=6, step=1)
    include_splits = st.checkbox("Detect splits (train/val/test)", value=True)
    test_mode = st.checkbox("Test mode (limit images for speed)", value=False)
    test_limit = st.number_input("Max images (per split) in test mode", min_value=50, max_value=10000, value=300, step=50)

    st.markdown("---")
    st.subheader("Placeholders")
    objective = st.text_area("Objective", value="TODO: Describe the goal of this dataset.")
    domain = st.text_input("Domain", value="TODO: Domain (e.g., Agriculture / Medical / Industrial).")
    source = st.text_area("Source", value="TODO: Collection method, instruments, references.")
    usage_snippet = st.text_area("Usage snippet (Python)", value="# TODO: Add code snippet to load/use this dataset.\n")

if not Path(data_root).exists():
    st.warning("Dataset root folder not found. Please check the path.")
    st.stop()

with st.spinner("Analyzing dataset..."):
    meta = analyze_dataset(
        data_root,
        thumbs_per_class=thumbs_per_class,
        include_splits=include_splits,
        test_mode=test_mode,
        test_limit_imgs=int(test_limit),
    )

g = meta["global"]
st.success(f"Analyzed { _human_int(g['images_count']) } images across {len(g['classes'])} classes.")

# Class descriptions editable table
st.subheader("Class descriptions")
default_desc = {c: "" for c in g["classes"]}
desc_rows = [{"class": c, "description": ""} for c in g["classes"]]
desc_df = st.data_editor(desc_rows, num_rows="dynamic", key="class_desc_editor")
class_descriptions = {row["class"]: row.get("description","") for row in desc_df if row.get("class") in g["classes"]}

# Stats columns
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Counts & Stats")
    st.write("- **Corrupted images:**", g["corrupted_images"])
    st.write("- **Estimated duplicates:**", g["estimated_duplicate_images"])
    st.write("- **Color summary:**", g["color_summary"])
    st.write("- **Top resolutions:**", g["resolution_top"])
    ws, hs, ars = g["width_stats"], g["height_stats"], g["aspect_ratio_stats"]
    st.markdown(
        f"""
        - **Width** [min/mean/median/max]: {ws['min']}/{ws['mean']}/{ws['median']}/{ws['max']}  
        - **Height** [min/mean/median/max]: {hs['min']}/{hs['mean']}/{hs['median']}/{hs['max']}  
        - **Aspect ratio** [min/mean/median/max]: {ars['min']}/{ars['mean']}/{ars['median']}/{ars['max']}
        """
    )
with col2:
    # Class distribution plot
    class_png = plot_class_distribution(g["per_class_counts"])
    if class_png:
        st.markdown("### Class distribution")
        st.image(class_png, use_column_width=True)
    # Aspect ratio plot
    aspect_png = plot_aspect_ratio_hist(g["ratios"])
    if aspect_png:
        st.markdown("### Aspect ratio")
        st.image(aspect_png, use_column_width=True)

# Thumbnails preview
st.markdown("### Example thumbnails")
thumbs_map = defaultdict(list)
# Build thumbs in-memory (and also list relative names for README)
seen = Counter()
for cls, p in g["samples"]:
    if seen[cls] >= thumbs_per_class:
        continue
    img = _safe_open(p)
    if img is None:
        continue
    th = _make_thumb(img)
    buf = io.BytesIO()
    th.save(buf, "JPEG", quality=90)
    buf.seek(0)
    thumbs_map[cls].append(buf.getvalue())
    seen[cls] += 1

# Show thumbs grid
for cls in g["classes"]:
    if not thumbs_map[cls]:
        continue
    st.markdown(f"**{cls}**")
    st.image(thumbs_map[cls], width=128)

st.markdown("---")

# Build README and JSON for download
# For README, we also need relative file names for thumbs; here we fake names as `thumbs/{cls}_{i}.jpg`
class_img_urls = {
    cls: [f"thumbs/{cls}_{i+1:02d}.jpg" for i in range(len(thumbs_map[cls]))]
    for cls in g["classes"]
}
readme_md = build_readme(
    dataset_name=dataset_name,
    url=(Path(dataset_url).resolve().as_uri() if Path(dataset_url).exists() else dataset_url),
    meta=meta,
    class_img_urls=class_img_urls,
    objective=objective,
    domain=domain,
    source=source,
    usage_snippet=usage_snippet,
    class_descriptions=class_descriptions
)
json_payload = build_json_payload(dataset_name, (Path(dataset_url).resolve().as_uri()
                          if Path(dataset_url).exists() else dataset_url), meta)

# Downloads
colL, colR = st.columns(2)
with colL:
    st.download_button(
        "⬇️ Download README.md",
        data=readme_md.encode("utf-8"),
        file_name="README.md",
        mime="text/markdown"
    )
with colR:
    st.download_button(
        "⬇️ Download dataset_meta.json",
        data=json.dumps(json_payload, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="dataset_meta.json",
        mime="application/json"
    )

st.caption("Tip: Save the PNGs shown above as `class_distribution.png` and `aspect_ratio_hist.png` next to README for a GitHub-style preview.")
