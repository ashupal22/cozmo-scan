import numpy as np
import pytest

from cozmo.geometry.fusion import PointSet, fuse, select_keyframes
from cozmo.geometry.planes import find_ceilings, find_floors, robust_height
from cozmo.ingest.stray import StrayCapture

rng = np.random.default_rng(0)


def _patch(n, height, normal_y, camera_y=1.5, noise=0.01, frame=0, extent=4.0):
    half = extent / 2
    xyz = np.column_stack([rng.uniform(-half, half, n), height + rng.normal(0, noise, n), rng.uniform(-half, half, n)])
    normal = np.tile([0.0, normal_y, 0.0], (n, 1))
    return xyz, normal, np.full(n, frame), np.full(n, camera_y)


def _points(*patches):
    return PointSet(*(np.concatenate(parts).astype(np.float32) if k != 2 else np.concatenate(parts).astype(np.int32)
                      for k, parts in enumerate(zip(*patches))))


def test_robust_height_ignores_outliers():
    values = np.concatenate([rng.normal(2.35, 0.01, 5000), rng.uniform(2.2, 2.5, 300)])
    plane = robust_height(values, 2.35)
    assert plane.height == pytest.approx(2.35, abs=0.001)
    assert plane.spread == pytest.approx(0.01, abs=0.003)


def test_floor_is_the_largest_up_facing_level():
    pts = _points(_patch(20000, 0.0, 1.0), _patch(5000, 0.75, 1.0, extent=1.0))  # floor + table top
    floors = find_floors(pts)
    assert floors[0].height == pytest.approx(0.0, abs=0.002)
    assert floors[0].area_m2 == pytest.approx(16.0, rel=0.05)


def test_a_dense_close_platform_does_not_beat_a_sparse_large_floor():
    # the ARKitScenes 41142278 failure: many returns from a small raised surface near the camera
    pts = _points(_patch(8000, 0.0, 1.0), _patch(30000, 0.20, 1.0, extent=1.5))
    floors = find_floors(pts)
    assert floors[0].height == pytest.approx(0.0, abs=0.002)
    assert floors[1].height == pytest.approx(0.20, abs=0.002)
    assert floors[0].area_m2 > 10 > floors[1].area_m2


def test_ceilings_ignore_furniture_and_find_two_heights():
    pts = _points(
        _patch(20000, 0.0, 1.0),                    # floor
        _patch(9000, 2.40, -1.0),                   # main ceiling
        _patch(4000, 2.27, -1.0, extent=2.0),       # lower bathroom ceiling
        _patch(6000, 0.72, -1.0, extent=1.0),       # table underside: faces down but far too low
        _patch(3000, 2.10, 1.0, extent=0.6),        # top of a tall cabinet: faces up
    )
    floor = find_floors(pts)[0]
    ceilings = find_ceilings(pts, floor)
    assert [round(c.height, 2) for c in ceilings] == [2.40, 2.27]


def test_no_ceiling_points_means_no_ceiling():
    pts = _points(_patch(20000, 0.0, 1.0), _patch(6000, 1.20, -1.0, extent=1.0))
    assert find_ceilings(pts, find_floors(pts)[0]) == []


def test_fused_points_from_downward_camera(stray_dir):
    cap = StrayCapture(stray_dir)
    assert list(select_keyframes(cap)) == [0, 2]
    pts = fuse(cap, stride=1)
    assert len(pts) > 0
    assert np.allclose(pts.xyz[:, 1], 0.0, atol=1e-5)
    assert np.allclose(pts.normal, [0.0, 1.0, 0.0], atol=1e-3)  # floor normals face up
    assert find_floors(pts)[0].height == pytest.approx(0.0, abs=1e-4)
