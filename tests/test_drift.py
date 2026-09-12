import numpy as np
import pytest

from cozmo.geometry.fusion import PointSet
from cozmo.slam.drift import estimate_drift, find_jumps
from cozmo.slam.matching import yaw_matrix

# 6 x 4 m room with two short stub walls, so no place looks like another
WALLS = [((0, 0), (6, 0)), ((6, 0), (6, 4)), ((6, 4), (0, 4)), ((0, 4), (0, 0)),
         ((3, 4), (3, 3.4)), ((0, 2.5), (0.8, 2.5))]
LAP = np.array([[1, 1], [5, 1], [5, 3], [1, 3], [1, 1]], float)
FPS, SPEED = 60.0, 0.6


class FakeCapture:
    def __init__(self, timestamps, positions, rotations):
        self.timestamps, self.positions, self._rotations = timestamps, positions, rotations

    def __len__(self):
        return len(self.timestamps)

    def rotation(self, i):
        return self._rotations[i]


def true_walk(laps=2):
    corners = np.vstack([LAP] + [LAP[1:]] * (laps - 1))
    xz = [corners[0]]
    for a, b in zip(corners[:-1], corners[1:]):
        n = int(np.linalg.norm(b - a) / (SPEED / FPS))
        xz += [a + (b - a) * k / n for k in range(1, n + 1)]
    xz = np.array(xz)
    heading = np.arctan2(np.gradient(xz[:, 1]), np.gradient(xz[:, 0]))
    return np.arange(len(xz)) / FPS, np.column_stack([xz[:, 0], np.full(len(xz), 1.5), xz[:, 1]]), heading


def record(true_positions, heading, creep_deg=4.0, rise_m=0.02, snap_at=None):
    """Recorded ARKit-style path: heading creeps, height rises; at snap_at ARKit relocalises onto the truth."""
    n = len(true_positions)
    theta = -np.radians(creep_deg) * np.arange(n) / n
    rise = rise_m * np.arange(n) / n
    if snap_at is not None:
        theta[snap_at:] -= theta[snap_at]
        rise[snap_at:] -= rise[snap_at]
    recorded = np.zeros_like(true_positions)
    recorded[0] = true_positions[0]
    for i in range(n - 1):
        if snap_at is not None and i + 1 == snap_at:
            recorded[i + 1] = true_positions[i + 1]
            continue
        step = yaw_matrix(theta[i]).T @ (true_positions[i + 1] - true_positions[i])
        recorded[i + 1] = recorded[i] + step + [0.0, rise[i + 1] - rise[i], 0.0]
    rotations = np.array([yaw_matrix(theta[i]).T @ yaw_matrix(-heading[i]) for i in range(n)])
    return recorded, rotations, theta


def observe(true_positions, recorded, theta, rng, every=6, reach=3.5):
    """Wall and floor points seen from every 6th frame, expressed in the recorded (drifting) map.
    Returns the points and each point's true position."""
    xyz, normals, frames, camera_y, truth = [], [], [], [], []
    for i in range(0, len(true_positions), every):
        c = true_positions[i, [0, 2]]
        pts, nrm = [], []
        for (x0, z0), (x1, z1) in WALLS:
            a, b = np.array([x0, z0], float), np.array([x1, z1], float)
            p = a + np.outer(rng.uniform(0, 1, int(np.linalg.norm(b - a) * 25)), b - a)
            p = p[np.linalg.norm(p - c, axis=1) < reach]
            if not len(p):
                continue
            d = (b - a) / np.linalg.norm(b - a)
            normal = np.array([-d[1], d[0]])
            facing = np.sign((c - p) @ normal)[:, None] * normal
            pts.append(np.column_stack([p[:, 0], rng.uniform(0.5, 1.8, len(p)), p[:, 1]]))
            nrm.append(np.column_stack([facing[:, 0], np.zeros(len(p)), facing[:, 1]]))
        floor = np.clip(c + rng.uniform(-1.5, 1.5, (60, 2)), [0.1, 0.1], [5.9, 3.9])
        pts.append(np.column_stack([floor[:, 0], np.zeros(60), floor[:, 1]]))
        nrm.append(np.tile([0.0, 1.0, 0.0], (60, 1)))
        true_xyz = np.vstack(pts) + rng.normal(0, 0.003, (sum(len(p) for p in pts), 3))
        R = yaw_matrix(theta[i]).T
        xyz.append((true_xyz - true_positions[i]) @ R.T + recorded[i])
        normals.append(np.vstack(nrm) @ R.T)
        frames.append(np.full(len(true_xyz), i))
        camera_y.append(np.full(len(true_xyz), recorded[i, 1]))
        truth.append(true_xyz)
    points = PointSet(np.vstack(xyz).astype(np.float32), np.vstack(normals).astype(np.float32),
                      np.concatenate(frames).astype(np.int32), np.concatenate(camera_y).astype(np.float32))
    return points, np.vstack(truth)


def distortion(xyz: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Plan distance of each point from its true position after the best rigid fit of the whole map.
    Where the map as a whole sits is arbitrary (only shapes are measured), so that is removed first."""
    p, q = xyz[:, [0, 2]].astype(float), truth[:, [0, 2]]
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    U, _, Vt = np.linalg.svd((p - pc).T @ (q - qc))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    return np.linalg.norm((p - pc) @ R.T + qc - q, axis=1)


def floor_heights(points: PointSet) -> np.ndarray:
    return points.xyz[points.normal[:, 1] > 0.9, 1]


def test_consecutive_jumping_steps_are_one_jump():
    positions = np.zeros((10, 3))
    positions[5:, 0] = 0.3
    positions[6:, 0] = 0.6
    assert find_jumps(positions) == [(4, 6)]


def test_heading_creep_and_height_drift_are_removed():
    t, true_positions, heading = true_walk()
    recorded, rotations, theta = record(true_positions, heading)
    points, truth = observe(true_positions, recorded, theta, np.random.default_rng(0))
    correction, report = estimate_drift(FakeCapture(t, recorded, rotations), points)
    fixed = correction.apply(points)

    before, after = distortion(points.xyz, truth), distortion(fixed.xyz, truth)
    assert np.percentile(before, 90) > 0.08                      # the drift is real
    assert np.median(after) < 0.01 and np.percentile(after, 90) < 0.02
    assert np.abs(floor_heights(fixed)).max() < 0.02             # height rise removed
    assert report.heading_priors_used and report.loops_accepted >= 1
    assert report.wall_heading_p90_after_deg < report.wall_heading_p90_before_deg
    assert report.wall_map_area_after_m2 < report.wall_map_area_before_m2
    heading_error = np.degrees(correction.theta - theta)
    assert np.abs(heading_error - heading_error.mean()).max() < 0.6


def test_arkit_relocalization_jump_is_used_as_a_link():
    t, true_positions, heading = true_walk()
    snap = int(0.7 * len(t))
    recorded, rotations, theta = record(true_positions, heading, creep_deg=6.0, snap_at=snap)
    points, truth = observe(true_positions, recorded, theta, np.random.default_rng(1))
    correction, report = estimate_drift(FakeCapture(t, recorded, rotations), points)

    assert len(report.jumps) == 1 and report.relocalizations == 1
    assert report.jumps[0]["degrees"] == pytest.approx(np.degrees(theta[snap - 1]), abs=0.2)
    before, after = distortion(points.xyz, truth), distortion(correction.apply(points).xyz, truth)
    assert np.percentile(before, 90) > 0.08
    assert np.median(after) < 0.01 and np.percentile(after, 90) < 0.03
    assert report.to_schema()["correction"][:2] == ["jump_cut", "loop_closure"]
