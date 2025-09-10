from __future__ import annotations

from pathlib import Path

from PIL import Image

from databaseMLUtils.reporting import class_distribution_report


def test_class_distribution_report(tmp_path: Path) -> None:
    for cls in ["A", "B"]:
        d = tmp_path / cls
        d.mkdir(parents=True)
        # two images in A, one in B
        n = 2 if cls == "A" else 1
        for i in range(n):
            Image.new("RGB", (8, 8), color=(i, i, i)).save(d / f"{i}.jpg")
    df = class_distribution_report(tmp_path)
    assert set(df["class"]) == {"A", "B"}
    assert df.set_index("class").loc["A", "count"] == 2
    assert df.set_index("class").loc["B", "count"] == 1

