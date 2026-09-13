"""Focal length from the straight lines of a room (Manhattan world), for videos without metadata.

Rooms are full of straight edges in three perpendicular directions: wall corners, door frames, the
floor-wall line, tiles, furniture. A line segment in the image, with the camera centre, spans a plane;
the segment's 3D direction lies in that plane. Only with the right focal length can one set of three
perpendicular directions explain most segments: with a wrong focal length the rays fan out wrongly and
parallel 3D lines stop meeting at perpendicular vanishing points.

For every candidate focal length, each key frame scores the best three perpendicular directions by the
share of its segment length that points along one of them (RANSAC over triples of segments, the same
triples for every candidate). Frames where the score barely changes with the focal length (e.g. facing
one wall squarely, where two vanishing points are at infinity) carry no information and are skipped.
Scores are normalised per frame and summed; the peak is the estimate. Resampling the frames
(bootstrap) gives its uncertainty.

On our walks: c00a170fe1 +0.0%, 1a8384c3f6 +1.6% against ARKit's calibration, where DA3's own estimate
is 10% too long on both (bench/README.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_FRAMES = 60
FOCAL_RANGE = (0.6, 2.0)        # focal / width: 94 to 28 degrees of horizontal view
GRID = 81
HYPOTHESES = 400
TAU_DEG = 1.0                   # a segment supports a direction within this angle
MIN_SEGMENT = 0.04              # of the long side
MIN_SEGMENTS = 20
MIN_SWING = 0.05                # a frame must change its score by this much across the candidates
MIN_FRAMES = 8
BOOTSTRAP = 200


@dataclass
class FocalEstimate:
    fx_over_width: float
    sigma: float                # relative, one sigma, from the bootstrap
    frames_used: int
    frames_tried: int


def line_segments(gray: np.ndarray) -> np.ndarray:
    """(N, 4) segments x1, y1, x2, y2 of at least MIN_SEGMENT of the image's long side."""
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    L = lines[:, 0, :].astype(float)
    length = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    return L[length >= MIN_SEGMENT * max(gray.shape)]


def manhattan_support(L: np.ndarray, w: int, h: int, focal_px: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For each candidate focal length (pixels), the largest share of segment length that three
    perpendicular 3D directions explain. The principal point is taken at the image centre."""
    cx, cy = (w - 1) / 2, (h - 1) / 2
    length = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    weight = length / length.sum()
    a, b, c = (rng.choice(len(L), HYPOTHESES, p=weight) for _ in range(3))
    distinct = (a != b) & (b != c) & (a != c)
    a, b, c = a[distinct], b[distinct], c[distinct]
    threshold = np.sin(np.radians(TAU_DEG))
    out = np.zeros(len(focal_px))
    ones = np.ones(len(L))
    for q, f in enumerate(focal_px):
        r1 = np.stack([(L[:, 0] - cx) / f, (L[:, 1] - cy) / f, ones], 1)
        r2 = np.stack([(L[:, 2] - cx) / f, (L[:, 3] - cy) / f, ones], 1)
        n = np.cross(r1, r2)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        d1 = np.cross(n[a], n[b])                 # where the planes of two segments meet: one direction
        norm = np.linalg.norm(d1, axis=1)
        ok = norm > 1e-6
        d1 = d1[ok] / norm[ok, None]
        d2 = np.cross(d1, n[c[ok]])               # in the third segment's plane and perpendicular to d1
        d2 /= np.linalg.norm(d2, axis=1, keepdims=True) + 1e-12
        D = np.stack([d1, d2, np.cross(d1, d2)], 1)
        aligned = np.abs(np.einsum("sk,hjk->hsj", n, D)).min(axis=2) < threshold
        out[q] = (aligned @ weight).max() if len(D) else 0.0
    return out


def _peak(grid_log: np.ndarray, total: np.ndarray) -> float:
    k = int(np.argmax(total))
    if 0 < k < len(total) - 1:
        y0, y1, y2 = total[k - 1], total[k], total[k + 1]
        denom = y0 - 2 * y1 + y2
        offset = 0.5 * (y0 - y2) / denom if denom < 0 else 0.0
        return float(np.exp(grid_log[k] + offset * (grid_log[1] - grid_log[0])))
    return float(np.exp(grid_log[k]))


def estimate_focal(images: list, seed: int = 0) -> FocalEstimate | None:
    """Focal length / image width from up to MAX_FRAMES evenly spaced images (paths or grey arrays),
    or None when too few frames show usable line structure."""
    step = max(1, len(images) // MAX_FRAMES)
    grid_log = np.linspace(np.log(FOCAL_RANGE[0]), np.log(FOCAL_RANGE[1]), GRID)
    rng = np.random.default_rng(seed)
    curves, tried = [], 0
    for item in images[::step]:
        gray = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE) if not isinstance(item, np.ndarray) else item
        tried += 1
        h, w = gray.shape
        L = line_segments(gray)
        if len(L) < MIN_SEGMENTS:
            continue
        score = manhattan_support(L, w, h, np.exp(grid_log) * w, rng)
        if score.max() - score.min() < MIN_SWING:
            continue
        curves.append(score / score.max())
    if len(curves) < MIN_FRAMES:
        return None
    curves = np.array(curves)
    estimate = _peak(grid_log, curves.sum(axis=0))
    boot = [_peak(grid_log, curves[rng.integers(0, len(curves), len(curves))].sum(axis=0)) for _ in range(BOOTSTRAP)]
    sigma = float(np.std(np.log(boot)))
    return FocalEstimate(estimate, sigma, len(curves), tried)
