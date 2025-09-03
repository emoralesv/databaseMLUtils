from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


def class_distribution_report(dataset_dir: str | os.PathLike, out_csv: str | None = None) -> pd.DataFrame:
    """Create a class distribution report for a folder-structured classification dataset.

    Assumes structure: root/<class_name>/*.jpg

    Args:
        dataset_dir: Root directory of classification dataset.
        out_csv: Optional path to write CSV.

    Returns:
        DataFrame with columns [class, count, frac].
    """
    root = Path(dataset_dir)
    counts: Counter[str] = Counter()
    total = 0
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        n = sum(1 for _ in cls_dir.glob("**/*") if _.is_file())
        counts[cls_dir.name] = n
        total += n
    rows = []
    for k, v in counts.items():
        frac = (v / total) if total > 0 else 0.0
        rows.append({"class": k, "count": v, "frac": frac})
    df = pd.DataFrame(rows).sort_values("class").reset_index(drop=True)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def object_size_stats(objects_xywh: Iterable[Tuple[float, float, float, float]]) -> pd.DataFrame:
    """Compute simple size statistics for a list of (x,y,w,h) objects.

    Args:
        objects_xywh: Iterable of bbox (x, y, w, h) in pixels.

    Returns:
        DataFrame with width, height, area for each object.
    """
    rows = []
    for (x, y, w, h) in objects_xywh:
        rows.append({"width": float(w), "height": float(h), "area": float(w) * float(h)})
    return pd.DataFrame(rows)


def split_summary(splits: dict[str, int]) -> pd.DataFrame:
    """Summarize dataset split sizes.

    Args:
        splits: Mapping split -> count.

    Returns:
        DataFrame with split, count, frac.
    """
    total = sum(splits.values())
    rows = [
        {"split": k, "count": v, "frac": (v / total) if total else 0.0} for k, v in splits.items()
    ]
    return pd.DataFrame(rows)

