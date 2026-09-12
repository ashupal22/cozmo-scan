import numpy as np
import pytest

from cozmo.slam.matching import rot2, yaw_matrix
from cozmo.slam.posegraph import Link, Node, Odometry, odometry_sigmas, solve

BUILDING_DEG = 20.0


def square_walk(creep_deg_per_step=-0.15, rise_m_per_step=0.0008, heading=True, floors=True, seed=5):
    """True anchors walk a 6 x 4 m rectangle back to the start. Recorded anchors carry a heading creep
    and a slow vertical rise, as ARKit drift does. Returns nodes, odometry, true anchors, true yaw corrections."""
    rng = np.random.default_rng(seed)
    corners = np.array([[0, 0], [6, 0], [6, 4], [0, 4], [0, 0]], float)
    path = [corners[0]]
    for a, b in zip(corners[:-1], corners[1:]):
        n = int(np.linalg.norm(b - a) / 0.5)
        path += [a + (b - a) * k / n for k in range(1, n + 1)]
    true = np.column_stack([np.array(path)[:, 0], np.full(len(path), 1.5), np.array(path)[:, 1]])
    S = len(true)
    theta_true = np.radians(creep_deg_per_step * np.arange(S))
    recorded = [true[0].copy()]
    for s in range(S - 1):
        step = yaw_matrix(theta_true[s]).T @ (true[s + 1] - true[s])
        recorded.append(recorded[-1] + step + [0.0, rise_m_per_step, 0.0])
    recorded = np.array(recorded)
    nodes = []
    for s in range(S):
        wall = (BUILDING_DEG + np.degrees(theta_true[s]) + rng.normal(0, 0.2)) % 90 if heading else None
        floor = (0.0 - (true[s, 1] - recorded[s, 1])) if floors else None  # true floor at y = 0
        nodes.append(Node(recorded[s], wall, floor))
    odometry = [Odometry(s, s + 1, *odometry_sigmas(0.5)) for s in range(S - 1)]
    return nodes, odometry, true, theta_true


def loop_link(nodes, true, theta_true, target, source):
    alpha = theta_true[source] - theta_true[target]
    # the target node's recorded frame, corrected by its own true correction, is the truth
    p_t, p_s = nodes[source].anchor[[0, 2]], nodes[target].anchor[[0, 2]]
    true_t_in_target_recorded = rot2(theta_true[target]).T @ (true[source, [0, 2]] - true[target, [0, 2]]) + p_s
    shift = true_t_in_target_recorded - rot2(alpha) @ p_t
    return Link(target, source, float(np.degrees(alpha)), shift, 0.03, 0.5)


def test_heading_creep_vertical_drift_and_loop_are_corrected():
    nodes, odometry, true, theta_true = square_walk()
    last = len(nodes) - 1
    solution = solve(nodes, odometry, [loop_link(nodes, true, theta_true, 0, last)])
    corrected = np.array([solution.corrected_anchor(nodes, s) for s in range(len(nodes))])
    before = np.linalg.norm(np.array([n.anchor for n in nodes])[:, [0, 2]] - true[:, [0, 2]], axis=1).max()
    assert before > 0.3                                                   # the drift is real
    assert np.degrees(np.abs(solution.theta - theta_true)).max() < 0.3
    assert np.linalg.norm(corrected[:, [0, 2]] - true[:, [0, 2]], axis=1).max() < 0.05
    assert np.abs(corrected[:, 1] - true[:, 1]).max() < 0.01
    assert solution.building_yaw_deg == pytest.approx(BUILDING_DEG, abs=0.2)
    assert solution.heading_priors_used and len(solution.links_used) == 1


def test_nothing_to_correct_without_evidence():
    nodes, odometry, _, _ = square_walk(heading=False, floors=False)
    solution = solve(nodes, odometry, [])
    assert np.allclose(solution.theta, 0, atol=1e-6) and np.allclose(solution.delta, 0, atol=1e-6)


def test_a_wrong_loop_is_rejected():
    nodes, odometry, true, theta_true = square_walk()
    last = len(nodes) - 1
    good = loop_link(nodes, true, theta_true, 0, last)
    wrong = loop_link(nodes, true, theta_true, 6, 26)
    wrong.shift = wrong.shift + np.array([1.2, -0.8])
    solution = solve(nodes, odometry, [good, wrong])
    assert [l.source for l in solution.links_rejected] == [26]
    assert np.degrees(np.abs(solution.theta - theta_true)).max() < 0.3


def test_heading_priors_switch_off_in_a_non_rectangular_home():
    nodes, odometry, _, _ = square_walk(floors=False)
    rng = np.random.default_rng(11)
    for n in nodes:
        n.wall_yaw_deg = float(rng.uniform(0, 90))
    assert not solve(nodes, odometry, []).heading_priors_used
