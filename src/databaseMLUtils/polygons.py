from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def polygons_to_aabb(points: Iterable[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Compute axis-aligned bbox (AABB) from polygon points.

    Args:
        points: Iterable of (x, y).

    Returns:
        (x, y, w, h) AABB.
    """
    pts = np.asarray(list(points), dtype=float)
    if pts.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def polygons_to_obb(points: Iterable[Tuple[float, float]]):
    """Compute oriented bounding box (OBB) with minimal area.

    Uses shapely if available; falls back to PCA-based approximation otherwise.

    Args:
        points: Iterable of (x, y).

    Returns:
        center_x, center_y, width, height, angle_radians
    """
    pts = np.asarray(list(points), dtype=float)
    if pts.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        from shapely.geometry import Polygon  # type: ignore

        poly = Polygon(pts)
        if poly.is_empty:
            x, y, w, h = polygons_to_aabb(pts)
            return x + w / 2, y + h / 2, w, h, 0.0
        rect = poly.minimum_rotated_rectangle
        xs, ys = rect.exterior.coords.xy
        rect_pts = np.stack([xs, ys], axis=1)[:-1]
        # Compute center, width/height, and angle
        center = rect_pts.mean(axis=0)
        edge = rect_pts[1] - rect_pts[0]
        angle = float(np.arctan2(edge[1], edge[0]))
        # side lengths
        w = float(np.linalg.norm(rect_pts[1] - rect_pts[0]))
        h = float(np.linalg.norm(rect_pts[2] - rect_pts[1]))
        return float(center[0]), float(center[1]), w, h, angle
    except Exception:
        # PCA-based fallback
        mean = pts.mean(axis=0)
        X = pts - mean
        cov = X.T @ X / max(1, len(X) - 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vecs = vecs[:, order]
        R = vecs
        rot = X @ R
        x0, y0 = rot.min(axis=0)
        x1, y1 = rot.max(axis=0)
        w = float(x1 - x0)
        h = float(y1 - y0)
        angle = float(np.arctan2(R[1, 0], R[0, 0]))
        cx, cy = mean
        return float(cx), float(cy), w, h, angle

