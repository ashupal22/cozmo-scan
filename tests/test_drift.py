import numpy as np
import pytest

from cozmo.geometry.fusion import PointSet
from cozmo.slam.drift import Submap, _build_submaps, _group_by_time, _segments, estimate_drift, find_jumps, gate_floors
from cozmo.slam.matching import yaw_matrix
from cozmo.slam.posegraph import Node

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


def record(true_positions, heading, creep_deg=4.0, rise_m=0.02, snap_at=None, break_at=None, break_deg=0.0,
           break_shift_m=0.0):
    """Recorded ARKit-style path: heading creeps, height rises; at snap_at ARKit relocalises onto the truth.
    At break_at a video tracking break turns the rest of the walk by break_deg and shifts it by break_shift_m."""
    n = len(true_positions)
    theta = -np.radians(creep_deg) * np.arange(n) / n
    if break_at is not None:
        theta[break_at:] -= np.radians(break_deg)
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
        if break_at is not None and i + 1 == break_at:
            step = step + [break_shift_m, 0.0, 0.0]
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


def test_short_last_group_is_merged_without_losing_or_repeating_frames():
    times = np.arange(0, 13.0, 0.5)             # groups start every 4 s; the last one is 0.5 s long
    frames = np.arange(len(times))
    groups = _group_by_time(frames, times)
    flat = [f for g in groups for f in g]
    assert flat == list(frames)
    assert len(groups) == 3


def test_submaps_cover_every_keyframe_once():
    t, true_positions, heading = true_walk()
    recorded, rotations, theta = record(true_positions, heading)
    points, _ = observe(true_positions, recorded, theta, np.random.default_rng(2))
    capture = FakeCapture(t, recorded, rotations)
    frames = np.concatenate([s.frames for s in _build_submaps(capture, points, _segments(len(capture), []))])
    assert len(frames) == len(np.unique(frames)) == len(np.unique(points.frame))


def test_a_table_top_is_not_used_as_the_floor():
    def submap(anchor, floor):
        return Submap(0, np.array([anchor]), anchor, np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, int),
                      Node(np.zeros(3), None, floor))
    positions = np.zeros((5, 3))
    positions[:, 1] = 1.45
    submaps = [submap(0, 0.012), submap(1, 0.0), submap(2, 0.43), submap(3, -0.02), submap(4, 0.01)]
    assert gate_floors(submaps, positions) == 1
    assert [s.node.floor_y for s in submaps] == [0.012, 0.0, None, -0.02, 0.01]


def test_a_match_between_corrected_maps_converts_back_to_recorded_coordinates():
    from cozmo.slam.drift import to_recorded_link
    from cozmo.slam.matching import rot2
    rng = np.random.default_rng(4)
    anchors = rng.uniform(-5, 5, (2, 2))
    theta = np.radians([1.7, -2.4])
    delta = rng.uniform(-0.3, 0.3, (2, 2))

    def corrected(k, q):
        return (q - anchors[k]) @ rot2(theta[k]).T + anchors[k] + delta[k]

    true_yaw, true_shift = np.radians(3.1), np.array([0.21, -0.37])  # recorded: q_0 = rot2(yaw) q_1 + shift
    q1 = rng.uniform(-4, 4, (20, 2))
    q0 = q1 @ rot2(true_yaw).T + true_shift
    y0, y1 = corrected(0, q0), corrected(1, q1)                       # what the matcher sees
    c0, c1 = y0.mean(axis=0), y1.mean(axis=0)
    U, _, Vt = np.linalg.svd((y1 - c1).T @ (y0 - c0))
    R = (U @ Vt).T
    yaw_seen = np.arctan2(R[0, 1], R[0, 0])
    shift_seen = c0 - rot2(yaw_seen) @ c1
    link = to_recorded_link(0, 1, np.degrees(yaw_seen), shift_seen, anchors, theta, delta, (0.03, 0.5))
    assert link.yaw_deg == pytest.approx(np.degrees(true_yaw), abs=1e-9)
    assert np.allclose(link.shift, true_shift, atol=1e-9)


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


def test_arkit_snap_back_to_the_start_becomes_a_loop_closure():
    # As on walk 1a8384c3f6: ARKit snaps back onto its map just after the walker returns to the start
    t, true_positions, heading = true_walk()
    snap = len(t) // 2 + 30
    recorded, rotations, theta = record(true_positions, heading, creep_deg=6.0, snap_at=snap)
    points, truth = observe(true_positions, recorded, theta, np.random.default_rng(1))
    correction, report = estimate_drift(FakeCapture(t, recorded, rotations), points)

    assert len(report.jumps) == 1 and report.relocalizations == 1 and report.anchors == 1
    assert report.anchor_wall_agreement and report.anchor_wall_agreement[0] > 0.5
    assert report.jumps[0]["degrees"] == pytest.approx(np.degrees(theta[snap - 1]), abs=0.2)
    before, after = distortion(points.xyz, truth), distortion(correction.apply(points).xyz, truth)
    assert np.percentile(before, 90) > 0.05           # the snap halfway resets part of the drift
    assert np.median(after) < 0.01 and np.percentile(after, 90) < 0.03
    assert report.to_schema()["correction"][:2] == ["jump_cut", "loop_closure"]


def test_a_snap_far_from_the_start_is_not_tied_to_it():
    t, true_positions, heading = true_walk()
    snap = int(0.7 * len(t))  # on the far side of the room, 3+ m from where the walk began
    recorded, rotations, theta = record(true_positions, heading, creep_deg=6.0, snap_at=snap)
    points, _ = observe(true_positions, recorded, theta, np.random.default_rng(1))
    _, report = estimate_drift(FakeCapture(t, recorded, rotations), points)
    assert report.relocalizations == 1 and report.anchors == 0


def test_a_video_tracking_break_is_held_loosely_and_reattached():
    # As in a video walk: tracking was lost for a moment, and everything after it came out turned and shifted
    t, true_positions, heading = true_walk()
    brk = len(t) // 3
    recorded, rotations, theta = record(true_positions, heading, creep_deg=2.0, break_at=brk, break_deg=25.0,
                                        break_shift_m=0.25)
    points, truth = observe(true_positions, recorded, theta, np.random.default_rng(3))
    capture = FakeCapture(t, recorded, rotations)
    correction, report = estimate_drift(capture, points, jumps=[(brk - 1, brk)], relocalized=False)
    undeclared, _ = estimate_drift(capture, points, jumps=[])

    before = distortion(points.xyz, truth)
    after = distortion(correction.apply(points).xyz, truth)
    after_undeclared = distortion(undeclared.apply(points).xyz, truth)
    assert np.percentile(before, 90) > 0.3
    assert np.median(after) < 0.02 and np.percentile(after, 90) < 0.05
    assert np.percentile(after_undeclared, 90) > 2 * np.percentile(after, 90)
    assert report.jumps[0]["kind"] == "tracking break"
    assert report.relocalizations == 0 and report.anchors == 0
    assert "tracking break" in report.to_schema()["notes"]
