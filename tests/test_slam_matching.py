import numpy as np
import pytest

from cozmo.slam.matching import match_walls, rot2, two_wall_directions, wrap, yaw_matrix, yaw_of

rng = np.random.default_rng(3)


def wall_map(noise=0.004):
    """Plan-view wall points of an L-shaped flat with a partition, plus normals."""
    segments = [((0, 0), (5, 0)), ((5, 0), (5, 3)), ((5, 3), (2.5, 3)), ((2.5, 3), (2.5, 6)),
                ((2.5, 6), (0, 6)), ((0, 6), (0, 0)), ((2.5, 0), (2.5, 1.2)), ((2.5, 2.1), (2.5, 3))]
    pts, nrm = [], []
    for (x0, z0), (x1, z1) in segments:
        n = int(np.hypot(x1 - x0, z1 - z0) / 0.01)
        t = rng.uniform(0, 1, n)
        p = np.column_stack([x0 + t * (x1 - x0), z0 + t * (z1 - z0)])
        direction = np.array([x1 - x0, z1 - z0]) / np.hypot(x1 - x0, z1 - z0)
        normal = np.array([-direction[1], direction[0]])
        pts.append(p + np.outer(rng.normal(0, noise, n), normal))
        nrm.append(np.tile(normal, (n, 1)))
    return np.vstack(pts), np.vstack(nrm)


def test_yaw_helpers_agree():
    R = yaw_matrix(np.radians(17.0))
    assert np.allclose(R[[0, 2]][:, [0, 2]], rot2(np.radians(17.0)))
    assert np.degrees(yaw_of(R)) == pytest.approx(17.0)
    assert wrap(50.0, 90.0) == pytest.approx(-40.0)


def test_small_drift_is_recovered_precisely():
    target, normals = wall_map()
    beta, move = np.radians(3.0), np.array([0.35, -0.2])
    source = (target @ rot2(beta).T + move)[::2]
    m = match_walls(target, normals, source)
    assert m.yaw_deg == pytest.approx(-3.0, abs=0.1)
    assert np.median(np.linalg.norm(m.apply(source) - target[::2], axis=1)) < 0.005
    assert m.inlier_fraction > 0.95 and m.ambiguity < 0.8


def test_global_search_between_unrelated_frames():
    target, normals = wall_map()
    source = target @ rot2(np.radians(31.0)).T + np.array([3.0, -2.0])
    m = match_walls(target, normals, source, yaw_range_deg=(-180, 179), yaw_step_deg=1.0,
                    max_shift_m=None, cell_m=0.05, center=True)
    assert m.yaw_deg == pytest.approx(-31.0, abs=0.1)
    assert m.median_residual_m < 0.005


def test_single_wall_is_flagged_as_underconstrained():
    target, normals = wall_map()
    straight = np.abs(normals[:, 1]) > 0.9  # walls along x only
    assert two_wall_directions(normals)
    assert not two_wall_directions(normals[straight])
