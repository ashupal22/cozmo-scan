"""Align one set of wall points onto another in plan view (x, z).

Convention used across cozmo.slam: rot2(theta) acts on (x, z) exactly as a rotation about +y
acts on (x, y, z): (x, z) -> (x cos + z sin, -x sin + z cos). A match (yaw, shift) means
    target ~= rot2(yaw) @ source + shift

1. Coarse: for each candidate yaw, rasterise the rotated source and find the best shift by FFT
   cross-correlation with the blurred target raster (correlative scan matching).
2. Fine: point-to-line ICP against the target's wall normals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

INLIER_M = 0.03


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])


def yaw_matrix(theta: float) -> np.ndarray:
    """3x3 rotation about +y; its (x, z) block is rot2(theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def yaw_of(R: np.ndarray) -> float:
    """Yaw angle of a (nearly) vertical-axis rotation, in the yaw_matrix convention."""
    return float(np.arctan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2]))


def wrap(angle, period):
    """Wrap to [-period/2, period/2)."""
    return (np.asarray(angle) + period / 2) % period - period / 2


def downsample(xz: np.ndarray, normals: np.ndarray | None = None, cell: float = 0.02):
    """Keep one point per plan cell."""
    _, keep = np.unique(np.floor(xz / cell).astype(np.int64), axis=0, return_index=True)
    keep = np.sort(keep)
    return (xz[keep], normals[keep]) if normals is not None else xz[keep]


def two_wall_directions(normals: np.ndarray, min_share: float = 0.15) -> bool:
    """True if normals fall into two roughly perpendicular families, so a match constrains both axes."""
    angle = np.degrees(np.arctan2(normals[:, 1], normals[:, 0])) % 180
    hist, _ = np.histogram(angle, bins=36, range=(0, 180))
    k = int(np.argmax(hist))
    first = sum(hist[(k + d) % 36] for d in (-1, 0, 1))
    second = sum(hist[(k + 18 + d) % 36] for d in (-1, 0, 1))
    return min(first, second) >= min_share * len(normals)


@dataclass
class Match:
    yaw_deg: float
    shift: np.ndarray
    inlier_fraction: float
    median_residual_m: float
    score: float
    ambiguity: float  # best correlation >= 0.3 m away from the chosen shift, relative to the best

    def apply(self, xz: np.ndarray) -> np.ndarray:
        return xz @ rot2(np.radians(self.yaw_deg)).T + self.shift


def _raster(xz: np.ndarray, origin: np.ndarray, size: int, cell: float, blur: float) -> np.ndarray:
    img = np.zeros((size, size), np.float32)
    ij = np.floor((xz - origin) / cell).astype(int)
    ok = (ij >= 0).all(axis=1) & (ij < size).all(axis=1)
    np.add.at(img, (ij[ok, 1], ij[ok, 0]), 1.0)
    img = np.minimum(img, 3.0)
    return ndimage.gaussian_filter(img, blur) if blur > 0 else img


def icp_point_to_line(source: np.ndarray, target: np.ndarray, target_normals: np.ndarray,
                      theta: float, shift: np.ndarray, iterations: int = 30,
                      start_radius: float = 0.15, tree: cKDTree | None = None):
    """Refine target ~= rot2(theta) @ source + shift. Returns (theta, shift, distances)."""
    tree = tree or cKDTree(target)
    shift = np.asarray(shift, float).copy()
    radius = start_radius
    for _ in range(iterations):
        moved = source @ rot2(theta).T + shift
        d, j = tree.query(moved, distance_upper_bound=radius)
        ok = np.isfinite(d)
        if ok.sum() < 10:
            break
        s, q, n = moved[ok], target[j[ok]], target_normals[j[ok]]
        c = s.mean(axis=0)
        p = s - c
        r = ((s - q) * n).sum(axis=1)
        J = np.column_stack([p[:, 1] * n[:, 0] - p[:, 0] * n[:, 1], n[:, 0], n[:, 1]])
        w = np.where(np.abs(r) < 0.02, 1.0, 0.02 / np.maximum(np.abs(r), 1e-9))
        A = J.T @ (J * w[:, None]) + 1e-6 * np.eye(3)
        step = np.linalg.solve(A, -J.T @ (r * w))
        theta += step[0]
        shift = rot2(step[0]) @ (shift - c) + c + step[1:]
        radius = max(radius * 0.7, 0.05)
        if abs(step[0]) < 1e-6 and np.linalg.norm(step[1:]) < 1e-5:
            break
    d, _ = tree.query(source @ rot2(theta).T + shift)
    return theta, shift, d


def match_walls(target: np.ndarray, target_normals: np.ndarray, source: np.ndarray,
                yaw_range_deg=(-8.0, 8.0), yaw_step_deg: float = 1.0, max_shift_m: float | None = 1.0,
                cell_m: float = 0.08, center: bool = False) -> Match | None:
    """Best rigid 2D alignment of source onto target, or None if either set is too small.

    center=True first moves the source centroid onto the target centroid (for maps in unrelated
    frames); max_shift_m=None allows any shift inside the search grid."""
    if len(target) < 20 or len(source) < 20:
        return None
    t_mid, s_mid = target.mean(axis=0), source.mean(axis=0)
    radius = max(np.linalg.norm(target - t_mid, axis=1).max(),
                 np.linalg.norm(source - s_mid, axis=1).max() if center else
                 np.linalg.norm(source - t_mid, axis=1).max())
    pad = (max_shift_m if max_shift_m is not None else radius) + 4 * cell_m
    size = int(np.ceil(2 * (radius + pad) / cell_m))
    origin = t_mid - size * cell_m / 2
    T = _raster(target, origin, size, cell_m, 1.0)
    FT = np.fft.fft2(T)
    t_norm = np.linalg.norm(T)
    k = np.fft.fftfreq(size, 1.0 / size)  # integer offsets, wrapped
    ky, kx = np.meshgrid(k, k, indexing="ij")
    reach = np.hypot(kx, ky) * cell_m
    allowed = reach <= max_shift_m if max_shift_m is not None else np.ones_like(reach, bool)

    best = None
    for yaw in np.arange(yaw_range_deg[0], yaw_range_deg[1] + 1e-9, yaw_step_deg):
        rotated = source @ rot2(np.radians(yaw)).T
        offset = (t_mid - rotated.mean(axis=0)) if center else np.zeros(2)
        S = _raster(rotated + offset, origin, size, cell_m, 1.0)
        corr = np.fft.ifft2(FT * np.conj(np.fft.fft2(S))).real
        corr[~allowed] = -np.inf
        idx = np.unravel_index(int(np.argmax(corr)), corr.shape)
        score = float(corr[idx] / (t_norm * np.linalg.norm(S) + 1e-9))
        if best is None or score > best[0]:
            best = (score, yaw, offset + np.array([kx[idx], ky[idx]]) * cell_m, corr, idx)

    score, yaw, shift, corr, idx = best
    far = np.hypot(kx - kx[idx], ky - ky[idx]) * cell_m >= 0.3
    other = corr[far & allowed]
    ambiguity = float(other.max() / corr[idx]) if other.size and corr[idx] > 0 else 1.0

    theta, shift, d = icp_point_to_line(source, target, target_normals, np.radians(yaw), shift)
    return Match(float(np.degrees(theta)), shift, float(np.mean(d < INLIER_M)), float(np.median(d)),
                 score, ambiguity)
