from __future__ import annotations

from databaseMLUtils.bbox import denormalize_bbox, normalize_bbox


def test_normalize_denormalize_bbox() -> None:
    x, y, w, h = 10, 5, 20, 10
    nx, ny, nw, nh = normalize_bbox((x, y, w, h), 100, 50)
    dx, dy, dw, dh = denormalize_bbox((nx, ny, nw, nh), 100, 50)
    assert (dx, dy, dw, dh) == (x, y, w, h)

