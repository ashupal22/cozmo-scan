"""Focal length from room lines (cozmo/video/focal.py) on rendered line drawings of a tiled room."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cozmo.video.focal import estimate_focal

W, H = 360, 480
ROOM = (-2.0, 2.5, -1.4, 1.2, -1.8, 2.2)


def room_segments():
    x0, x1, y0, y1, z0, z1 = ROOM
    segs = [((x, y0, z0), (x, y0, z1)) for x in np.arange(x0, x1 + 1e-9, 0.5)]           # floor tiles
    segs += [((x0, y0, z), (x1, y0, z)) for z in np.arange(z0, z1 + 1e-9, 0.5)]
    segs += [((x, y0, z), (x, y1, z)) for x in (x0, x1) for z in np.arange(z0, z1 + 1e-9, 0.8)]  # panels
    segs += [((x, y0, z), (x, y1, z)) for z in (z0, z1) for x in np.arange(x0, x1 + 1e-9, 0.8)]
    segs += [((x0, y1, z0), (x1, y1, z0)), ((x0, y1, z1), (x1, y1, z1)),                    # ceiling edges
             ((x0, y1, z0), (x0, y1, z1)), ((x1, y1, z0), (x1, y1, z1))]
    segs += [((x0, y, z0), (x0, y, z1)) for y in (0.0, 0.6)] + [((x, y, z1), (x1, y, z1)) for x, y in ((x0, 0.3),)]
    return np.array(segs, float)


def render(R, C, K):
    img = np.full((H, W), 60, np.uint8)
    for P, Q in room_segments():
        p, q = (P - C) @ R, (Q - C) @ R                  # camera coordinates
        if p[2] < 0.2 and q[2] < 0.2:
            continue
        if p[2] < 0.2 or q[2] < 0.2:                     # clip at the near plane
            t = (0.2 - p[2]) / (q[2] - p[2])
            (p, q) = (p + t * (q - p), q) if p[2] < 0.2 else (p, p + t * (q - p))
        a = K @ (p / p[2])
        b = K @ (q / q[2])
        cv2.line(img, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), 230, 2, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0)


@pytest.mark.parametrize("fx_over_width", [0.85, 1.11, 1.4])
def test_focal_from_room_lines(fx_over_width):
    f = fx_over_width * W
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])
    base = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1.0]])
    images = []
    for k, yaw in enumerate(np.linspace(-60, 250, 12)):
        R = Rotation.from_euler("yxz", [yaw, -18 - 6 * np.sin(k), 4 * np.cos(k)], degrees=True).as_matrix() @ base
        C = np.array([0.2 * np.cos(k), 0.0, 0.3 * np.sin(k)])
        images.append(render(R, C, K))
    est = estimate_focal(images)
    assert est is not None and est.frames_used >= 8
    assert est.fx_over_width == pytest.approx(fx_over_width, rel=0.015)
    assert 0 < est.sigma < 0.05


def test_no_lines_no_estimate():
    rng = np.random.default_rng(0)
    blank = [cv2.GaussianBlur(rng.integers(0, 255, (H, W)).astype(np.uint8), (31, 31), 0) for _ in range(12)]
    assert estimate_focal(blank) is None
