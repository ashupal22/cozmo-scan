"""Joining DA3 runs into one metric, level trajectory (cozmo/video/capture.py), on a synthetic box room.

Each run gets its own random rotation, shift and scale, as DA3's runs do; the chain must undo them
using only the shared frames, the floor and walls, and metric depth on a few frames."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cozmo.video import capture as vc
from cozmo.video.da3 import ViewSet

W, H = 48, 64                      # portrait, like an upright phone video
FX = 1.1                           # focal length / width
ROOM = (-2.0, 2.5, -1.4, 1.2, -1.8, 2.2)   # x0, x1, y0 (floor), y1 (ceiling), z0, z1


def box_depth(R, C, K):
    """z-depth image of the inside of ROOM for a camera at C with rotation R (camera to world)."""
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)  # z = 1
    dw = d @ R.T
    t = np.full(u.shape, np.inf)
    for axis, lo, hi in ((0, ROOM[0], ROOM[1]), (1, ROOM[2], ROOM[3]), (2, ROOM[4], ROOM[5])):
        with np.errstate(divide="ignore", invalid="ignore"):
            for plane in (lo, hi):
                tt = (plane - C[axis]) / dw[..., axis]
                t = np.where((tt > 0) & (tt < t), tt, t)
    return t.astype(np.float32)   # the ray has z = 1 in the camera, so t is the z-depth


def camera(yaw_deg, pitch_deg=-25.0, roll_deg=0.0):
    """Camera-to-world rotation, OpenCV axes (x right, y down, z forward), world y up."""
    base = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1.0]])  # camera looking along +z, level
    turn = Rotation.from_euler("yxz", [yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix()
    return turn @ base


def walk(n=40):
    t = np.linspace(0, 1, n)
    C = np.stack([-1.0 + 2.5 * t, np.full(n, 0.0) + 0.02 * np.sin(9 * t), -0.8 + 1.5 * t ** 2], 1)
    Rs = [camera(-40 + 200 * s, pitch_deg=-20 - 10 * np.sin(5 * s), roll_deg=3 * np.sin(7 * s)) for s in t]
    return C, np.array(Rs)


def views_for(C, Rs, runs, K, rng):
    """DA3-like outputs: every run in its own frame (random rotation, shift, scale)."""
    views = []
    for a, b in runs:
        G = Rotation.random(random_state=rng).as_matrix()
        g = rng.normal(size=3)
        s = float(np.exp(rng.normal(0, 0.5)))
        depth, w2c = [], []
        for k in range(a, b):
            R_loc = G.T @ Rs[k]
            C_loc = G.T @ (C[k] - g) / s
            w2c.append(np.hstack([R_loc.T, (-R_loc.T @ C_loc)[:, None]]))
            depth.append(box_depth(Rs[k], C[k], K) / s)
        views.append(ViewSet(np.array(depth), np.ones((b - a, H, W), np.float32), np.array(w2c),
                             np.repeat(K[None], b - a, 0), (W, H)))
    return views


def rigid_error(est, true):
    """Camera position error after the best rotation + shift (no scale)."""
    ms, mt = est.mean(0), true.mean(0)
    U, _, Vt = np.linalg.svd((true - mt).T @ (est - ms))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return np.linalg.norm((est - ms) @ R.T + mt - true, axis=1)


@pytest.fixture(scope="module")
def chained():
    rng = np.random.default_rng(3)
    C, Rs = walk()
    runs = vc.runs_for(len(C))
    K = vc.camera_matrix(FX, W, H)
    views = views_for(C, Rs, runs, K, rng)
    metric = {k: box_depth(Rs[k], C[k], K) for k in range(0, len(C), 4)}
    return vc.chain_runs(views, runs, K, metric), C, Rs


def test_runs_share_frames():
    runs = vc.runs_for(40)
    assert runs[0][0] == 0 and runs[-1][1] == 40
    for (a0, b0), (a1, b1) in zip(runs, runs[1:]):
        assert b0 - a1 == vc.OVERLAP


def test_chain_recovers_the_walk_in_metres(chained):
    chain, C, _ = chained
    err = rigid_error(chain.c2w[:, :3, 3], C)
    assert err.max() < 0.01   # 1 cm, after a rigid fit only: the scale must come out right by itself


def test_chain_is_level(chained):
    chain, _, Rs = chained
    # the world up seen from each camera must match the truth: tilt error, independent of heading
    for k in range(len(Rs)):
        up_est = chain.c2w[k, :3, :3].T @ np.array([0, 1.0, 0])
        up_true = Rs[k].T @ np.array([0, 1.0, 0])
        assert np.degrees(np.arccos(np.clip(up_est @ up_true, -1, 1))) < 1.0


def test_depths_are_metric(chained):
    chain, C, Rs = chained
    K = chain.K
    for k in (1, 13, 27, 38):
        truth = box_depth(Rs[k], C[k], K)
        assert np.median(chain.depth[k] / truth) == pytest.approx(1.0, abs=0.005)


def test_scale_solve_follows_seams_and_anchors():
    # three runs, truth x = [0.1, 0.3, 0.2]; seams exact, anchors noisy but unbiased, one bad anchor
    seams = [(1, 0.2, 0.01), (2, -0.1, 0.01)]
    anchors = [(0, 0.1 + e, 0.07) for e in (0.05, -0.04, 0.0)] + [(1, 0.3 + e, 0.07) for e in (0.03, -0.02)] + \
              [(2, 0.2, 0.07), (2, 1.5, 0.07)]
    x = vc.solve_log_scales(3, seams, anchors)
    assert x[1] - x[0] == pytest.approx(0.2, abs=0.005)
    assert x[2] - x[1] == pytest.approx(-0.1, abs=0.005)
    assert x[0] == pytest.approx(0.1, abs=0.06)


def test_level_finds_up_from_floor_and_walls():
    rng = np.random.default_rng(0)
    up = np.array([0.1, 0.99, -0.05])
    up /= np.linalg.norm(up)
    a = np.cross(up, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    normals = np.vstack([np.tile(up, (3000, 1)), np.tile(a, (4000, 1)), np.tile(-b, (4000, 1))])
    normals += rng.normal(0, 0.05, normals.shape)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    est, count = vc.level(normals, np.array([0.3, 0.95, 0.0]) / np.linalg.norm([0.3, 0.95, 0.0]))
    assert np.degrees(np.arccos(est @ up)) < 0.5
    assert count > 2500


def test_yaw_part_and_mean_rotation():
    R = Rotation.from_euler("y", 37, degrees=True).as_matrix()
    assert np.allclose(vc.yaw_part(R), R, atol=1e-9)
    tilted = Rotation.from_euler("x", 2, degrees=True).as_matrix() @ R
    assert vc.angle_deg(vc.yaw_part(tilted).T @ R) < 0.1
    Rs = [Rotation.from_rotvec(np.array([0, 0.3, 0]) + 0.01 * np.random.default_rng(i).normal(size=3)).as_matrix()
          for i in range(20)]
    assert vc.angle_deg(vc.mean_rotation(Rs).T @ Rotation.from_rotvec([0, 0.3, 0]).as_matrix()) < 0.3


def test_tracking_breaks_are_cut_at_the_least_confident_pair():
    conf = np.array([5, 5, 1.2, 5, 5, 1.0, 1.1, 1.0, 1.3, 1.5, 5, 5, 1.9])
    # doubtful pairs 1-2, 4-9 and 11: each stretch is cut once, at its least confident pair; no frame is dropped
    assert vc.tracking_breaks(conf) == [(1, 2), (4, 5), (11, 12)]
    assert vc.tracking_breaks(np.full(10, 4.0)) == []


def test_range_correction_stays_inside_its_calibrated_range():
    d = np.array([0.0, 0.2, 1.0, 3.6, 10.0])
    out = vc.correct_range(d)
    assert out[0] == 0.0
    factor = out[1:] / d[1:]
    lo, hi = vc.RANGE_CALIBRATED_M
    assert factor[0] == pytest.approx(np.exp(vc.RANGE_A) * lo ** vc.RANGE_B)
    assert factor[3] == pytest.approx(factor[2])          # beyond the calibrated range: the edge factor
    assert np.all(np.abs(factor - 1) < 0.05)



def test_unused_da3_imports_are_stubbed_once_and_again():
    import importlib
    import sys
    from cozmo.video.da3 import _stub_unused_imports
    _stub_unused_imports()
    _stub_unused_imports()               # the second model load must not fail on the stub it left
    importlib.import_module("pycolmap")
    assert hasattr(sys.modules["evo.core.trajectory"], "PosePath3D") or importlib.util.find_spec("evo") is not None
