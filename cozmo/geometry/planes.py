"""Horizontal planes (floor levels and ceilings) from points with normals.

A surface counts as floor when it faces up and lies well below the camera that saw it, and
as ceiling when it faces down, lies above that camera, and is at least 1.9 m over the floor.
The height rule matters: without it, furniture tops were picked as the "ceiling" at about
1.2 m in our own captures (see explore/probe_basic.py vs probe_normals.py).

Candidates are ranked by the plan-view area they cover, not by point count. A surface seen up
close returns many more points than a larger one seen from further away: on ARKitScenes walk
41142278 a 0.9 m2 platform out-pointed the 5.7 m2 floor and was reported as the floor, 21 cm high.
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
CELL_M = 0.10                        # plan-view grid used to measure covered area
AREA_BAND_M = 0.025                  # points this close to a plane count towards its area
POINTS_PER_INDEPENDENT_SAMPLE = 100  # neighbouring LiDAR returns are strongly correlated


@dataclass(frozen=True)
class HorizontalPlane:
    height: float         # world y, metres
    spread: float         # robust per-point standard deviation, metres
    count: int            # inlier points
    area_m2: float = 0.0  # plan-view area covered by points on the plane

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


def _plan_cells(xyz: np.ndarray) -> np.ndarray:
    return np.floor(xyz[:, [0, 2]] / CELL_M).astype(np.int64)


def _area_weighted_heights(xyz: np.ndarray) -> np.ndarray:
    """One height per occupied (plan cell, height bin), so point density does not decide the peaks."""
    keys = np.column_stack([_plan_cells(xyz), np.floor(xyz[:, 1] / BIN_M).astype(np.int64)])
    _, first = np.unique(keys, axis=0, return_index=True)
    return xyz[first, 1]


def _area_m2(xyz: np.ndarray, height: float) -> float:
    near = xyz[np.abs(xyz[:, 1] - height) < AREA_BAND_M]
    return float(len(np.unique(_plan_cells(near), axis=0)) * CELL_M ** 2) if len(near) else 0.0


def _peak_heights(values: np.ndarray, min_share: float, min_points: int = 100) -> list[float]:
    if len(values) < min_points:
        return []
    # pad with empty bins so a peak at the extreme height still has lower neighbours on both sides
    edges = np.arange(values.min() - 5 * BIN_M, values.max() + 6 * BIN_M, BIN_M)
    hist, edges = np.histogram(values, bins=edges)
    smooth = uniform_filter1d(hist.astype(float), 3)
    peaks, _ = find_peaks(smooth, height=min_share * smooth.max(), distance=5)
    return [float((edges[k] + edges[k + 1]) / 2) for k in peaks]


def _planes(xyz: np.ndarray, min_share: float) -> list[HorizontalPlane]:
    """Planes at area-weighted histogram peaks, largest covered area first."""
    planes = []
    for center in _peak_heights(_area_weighted_heights(xyz), min_share):
        p = robust_height(xyz[:, 1], center)
        if p is not None:
            planes.append(HorizontalPlane(p.height, p.spread, p.count, _area_m2(xyz, p.height)))
    return sorted(planes, key=lambda p: -p.area_m2)


def find_floors(points: PointSet, min_share: float = 0.2) -> list[HorizontalPlane]:
    """Candidate floor levels, largest area first."""
    mask = (points.normal[:, 1] > HORIZONTAL_DOT) & (points.xyz[:, 1] < points.camera_y - FLOOR_BELOW_CAMERA_M)
    return _planes(points.xyz[mask], min_share)


def ceiling_mask(points: PointSet, floor: HorizontalPlane) -> np.ndarray:
    above_floor = points.xyz[:, 1] - floor.height
    return ((points.normal[:, 1] < -HORIZONTAL_DOT)
            & (points.xyz[:, 1] > points.camera_y + CEILING_ABOVE_CAMERA_M)
            & (above_floor > MIN_CEILING_ABOVE_FLOOR_M) & (above_floor < MAX_CEILING_ABOVE_FLOOR_M))


def find_ceilings(points: PointSet, floor: HorizontalPlane, min_share: float = 0.2) -> list[HorizontalPlane]:
    """Candidate ceiling planes over `floor`, largest area first. Empty if the ceiling was never seen."""
    return _planes(points.xyz[ceiling_mask(points, floor)], min_share)
