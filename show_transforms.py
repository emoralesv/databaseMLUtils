from __future__ import annotations

import os
from pathlib import Path
from typing import List

from PIL import Image
import matplotlib.pyplot as plt


def _add_src_to_path() -> None:
    import sys
    for p in [Path(__file__).resolve().parents[1] / "src", Path.cwd() / "DATABASEMLUTILS" / "src"]:
        if p.exists() and str(p) not in sys.path:
            sys.path.append(str(p))


def _find_any_image() -> Path | None:
    # Try common dataset layout
    for root in [Path.cwd(), Path.cwd() / "..", Path.cwd().parents[1]]:
        cand_dirs = [root / "databases", root]
        for base in cand_dirs:
            if not base.exists():
                continue
            for p in base.rglob("*.jpg"):
                return p
    return None


def main(img_path: str | None = None, limit: int | None = None) -> None:
    _add_src_to_path()
    from databaseMLUtils.transforms import get_transforms_registry

    if img_path is None:
        p = _find_any_image()
        if p is None:
            # create synthetic image
            import numpy as np
            arr = (np.random.rand(256, 256, 3) * 255).astype("uint8")
            img = Image.fromarray(arr)
            img_path_display = "<synthetic>"
        else:
            img = Image.open(p).convert("RGB")
            img_path_display = str(p)
    else:
        p = Path(img_path)
        img = Image.open(p).convert("RGB")
        img_path_display = str(p)

    registry = get_transforms_registry()
    ids: List[str] = sorted(registry.keys())
    if limit is not None:
        ids = ids[: int(limit)]

    n = len(ids) + 1
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if isinstance(axes, (list, tuple)) else (axes if hasattr(axes, 'flatten') else [axes])

    axes[0].imshow(img)
    axes[0].set_title(f"Original\n{img_path_display}")
    axes[0].axis("off")
    for i, t_id in enumerate(ids, start=1):
        t = registry[t_id]
        view = t(img)
        axes[i].imshow(view, cmap="gray")
        axes[i].set_title(f"{t_id}")
        axes[i].axis("off")

    # Hide any extra axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Show transform views for an image")
    ap.add_argument("--img", type=str, default=None, help="Path to an image (.jpg)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of transforms to display")
    args = ap.parse_args()
    main(args.img, args.limit)

