"""Horizontal planes (floor levels and ceilings) from points with normals.

A surface counts as floor when it faces up and lies well below the camera that saw it, and
as ceiling when it faces down, lies above that camera, and is at least 1.9 m over the floor.
The height rule matters: without it, furniture tops were picked as the "ceiling" at about
1.2 m in our own captures (see explore/probe_basic.py vs probe_normals.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from cozmo.geometry.fusion import PointSet

HORIZONTAL_DOT = 0.97               # |normal . up| above this: within ~14 degrees of horizontal
FLOOR_BELOW_CAMERA_M = 0.7
CEILING_ABOVE_CAMERA_M = 0.1
MIN_CEILING_ABOVE_FLOOR_M = 1.9
MAX_CEILING_ABOVE_FLOOR_M = 4.0
BIN_M = 0.01
POINTS_PER_INDEPENDENT_SAMPLE = 100  # neighbouring LiDAR returns are strongly correlated


@dataclass(frozen=True)
class HorizontalPlane:
    height: float  # world y, metres
    spread: float  # robust per-point standard deviation, metres
    count: int     # inlier points

    @property
    def standard_error(self) -> float:
        return self.spread / np.sqrt(max(self.count / POINTS_PER_INDEPENDENT_SAMPLE, 1.0))


def robust_height(values: np.ndarray, center: float, window: float = 0.04, iterations: int = 3,
                  min_points: int = 50) -> HorizontalPlane | None:
    """Mean height of points near `center` after median/MAD outlier rejection."""
    sel = values[np.abs(values - center) < window]
    for _ in range(iterations):
        if len(sel) < min_points:
            return None
        med = np.median(sel)
        mad = 1.4826 * np.median(np.abs(sel - med)) + 1e-4
        sel = sel[np.abs(sel - med) < 2.5 * mad]
    if len(sel) < min_points:
        return None
    return HorizontalPlane(float(sel.mean()), float(sel.std()), int(len(sel)))


def _peak_heights(values: np.ndarray, min_share: float, min_points: int = 200) -> list[float]:
    if len(values) < min_points:
        return []
    # pad with empty bins so a peak at the extreme height still has lower neighbours on both sides
    edges = np.arange(values.min() - 5 * BIN_M, values.max() + 6 * BIN_M, BIN_M)
    hist, edges = np.histogram(values, bins=edges)
    smooth = uniform_filter1d(hist.astype(float), 3)
    peaks, _ = find_peaks(smooth, height=min_share * smooth.max(), distance=5)
    return [float((edges[k] + edges[k + 1]) / 2) for k in peaks]


def _planes(values: np.ndarray, min_share: float) -> list[HorizontalPlane]:
    planes = [robust_height(values, c) for c in _peak_heights(values, min_share)]
    return [p for p in planes if p is not None]


def find_floors(points: PointSet, min_share: float = 0.2) -> list[HorizontalPlane]:
    """Candidate floor levels, most supported first."""
    mask = (points.normal[:, 1] > HORIZONTAL_DOT) & (points.xyz[:, 1] < points.camera_y - FLOOR_BELOW_CAMERA_M)
    return sorted(_planes(points.xyz[mask, 1], min_share), key=lambda p: -p.count)


def ceiling_mask(points: PointSet, floor: HorizontalPlane) -> np.ndarray:
    above_floor = points.xyz[:, 1] - floor.height
    return ((points.normal[:, 1] < -HORIZONTAL_DOT)
            & (points.xyz[:, 1] > points.camera_y + CEILING_ABOVE_CAMERA_M)
            & (above_floor > MIN_CEILING_ABOVE_FLOOR_M) & (above_floor < MAX_CEILING_ABOVE_FLOOR_M))


def find_ceilings(points: PointSet, floor: HorizontalPlane, min_share: float = 0.2) -> list[HorizontalPlane]:
    """Candidate ceiling planes over `floor`, most supported first. Empty if the ceiling was never seen."""
    return sorted(_planes(points.xyz[ceiling_mask(points, floor), 1], min_share), key=lambda p: -p.count)
