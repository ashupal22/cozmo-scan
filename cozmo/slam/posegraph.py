"""Pose graph over submap anchors with 4 degrees of freedom per node.

ARKit keeps gravity well, so roll and pitch are trusted; drift lives in heading and position.
Node s carries a correction around its recorded anchor p_s:
    corrected(X) = yaw_matrix(theta_s) (X - p_s) + p_s + delta_s

Residuals, each divided by its standard deviation:
- odometry a -> b:  yaw_matrix(theta_a) (p_b - p_a) = (p_b + delta_b) - (p_a + delta_a),  theta_b = theta_a
- link s <- t (loop closure or ARKit relocalisation), measured in recorded coordinates as
  q_s = rot2(alpha) q_t + m for the same physical point: corrected positions must coincide
- wall heading (plane-anchored): recorded wall direction phi_s (mod 90) - theta_s = building direction
- floor: recorded floor height h_s + delta_s.y = building floor height
Node 0 is the reference and is never corrected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from cozmo.slam.matching import wrap

HEADING_SIGMA_DEG = 0.5
FLOOR_SIGMA_M = 0.01
PRUNE_SIGMAS = 4.0
NON_MANHATTAN_DEG = 3.0


def odometry_sigmas(distance_m: float) -> tuple[float, float]:
    """Standard deviations (metres, degrees) of ARKit's relative motion over one edge.

    The heading term is loose on purpose: ARKit heading drift is a steady creep (4.5 degrees over 54 m
    on walk 1a8384c3f6), and a tight random-walk model resists correcting it. Chosen on synthetic
    walks (tests/test_slam_posegraph.py, 5 seeds): heading 0.5 deg + this setting gave the lowest
    worst-case errors, 0.22 deg and 1 cm."""
    return 0.01 + 0.01 * distance_m, 0.2 + 0.5 * distance_m


@dataclass
class Node:
    anchor: np.ndarray                  # (3,) recorded position
    wall_yaw_deg: float | None = None   # dominant wall direction mod 90, when reliable
    floor_y: float | None = None        # floor height seen from this node, when reliable
    wall_yaw_sigma_deg: float | None = None  # uncertainty of wall_yaw_deg; HEADING_SIGMA_DEG if None


@dataclass
class Odometry:
    a: int
    b: int
    sigma_m: float
    sigma_deg: float


@dataclass
class Link:
    target: int
    source: int
    yaw_deg: float
    shift: np.ndarray  # (2,)
    sigma_m: float
    sigma_deg: float
    kind: str = "loop"  # "loop" or "relocalization"


@dataclass
class Solution:
    theta: np.ndarray                   # (S,) radians
    delta: np.ndarray                   # (S, 3) metres
    building_yaw_deg: float | None
    floor_y: float | None
    heading_priors_used: bool
    links_used: list[Link] = field(default_factory=list)
    links_rejected: list[Link] = field(default_factory=list)

    def corrected_anchor(self, nodes: list[Node], s: int) -> np.ndarray:
        return nodes[s].anchor + self.delta[s]


def _circular_mean_mod90(angles_deg) -> float:
    z = np.exp(1j * np.radians(np.asarray(angles_deg) * 4)).mean()
    return float(np.degrees(np.angle(z)) / 4 % 90)


class _Problem:
    def __init__(self, nodes, odometry, links, use_heading):
        self.S = len(nodes)
        self.P = np.array([n.anchor for n in nodes], float)
        self.odo = odometry
        self.links = links
        self.heading_idx = np.array([i for i, n in enumerate(nodes) if n.wall_yaw_deg is not None], int) \
            if use_heading else np.zeros(0, int)
        self.heading = np.array([nodes[i].wall_yaw_deg for i in self.heading_idx], float)
        self.heading_sigma = np.array([nodes[i].wall_yaw_sigma_deg or HEADING_SIGMA_DEG
                                       for i in self.heading_idx], float)
        self.floor_idx = np.array([i for i, n in enumerate(nodes) if n.floor_y is not None], int)
        self.floor = np.array([nodes[i].floor_y for i in self.floor_idx], float)
        self.has_phi, self.has_h = len(self.heading_idx) > 0, len(self.floor_idx) > 0

    def x0(self):
        parts = [np.zeros(4 * (self.S - 1))]
        if self.has_phi:
            parts.append([_circular_mean_mod90(self.heading[:5])])
        if self.has_h:
            parts.append([float(np.median(self.floor[:5]))])
        return np.concatenate(parts)

    def unpack(self, x):
        n = self.S - 1
        theta = np.concatenate([[0.0], x[:n]])
        delta = np.vstack([np.zeros((1, 3)), x[n:4 * n].reshape(n, 3)])
        k = 4 * n
        phi = x[k] if self.has_phi else None
        h = x[k + int(self.has_phi)] if self.has_h else None
        return theta, delta, phi, h

    def residuals(self, x):
        theta, delta, phi, h = self.unpack(x)
        P, out = self.P, []
        if self.odo:
            a = np.array([e.a for e in self.odo])
            b = np.array([e.b for e in self.odo])
            sm = np.array([e.sigma_m for e in self.odo])[:, None]
            sd = np.array([e.sigma_deg for e in self.odo])
            d = P[b] - P[a]
            c, s = np.cos(theta[a]), np.sin(theta[a])
            rotated = np.column_stack([c * d[:, 0] + s * d[:, 2], d[:, 1], -s * d[:, 0] + c * d[:, 2]])
            out.append(((rotated - (d + delta[b] - delta[a])) / sm).ravel())
            out.append(np.degrees(theta[b] - theta[a]) / sd)
        if self.links:
            s_idx = np.array([e.target for e in self.links])
            t_idx = np.array([e.source for e in self.links])
            alpha = np.radians([e.yaw_deg for e in self.links])
            m = np.array([e.shift for e in self.links], float)
            sm = np.array([e.sigma_m for e in self.links])[:, None]
            sd = np.array([e.sigma_deg for e in self.links])
            ps, pt = P[s_idx][:, [0, 2]], P[t_idx][:, [0, 2]]
            ca, sa = np.cos(alpha), np.sin(alpha)
            q = np.column_stack([ca * pt[:, 0] + sa * pt[:, 1], -sa * pt[:, 0] + ca * pt[:, 1]]) + m - ps
            cs, ss = np.cos(theta[s_idx]), np.sin(theta[s_idx])
            lhs = np.column_stack([cs * q[:, 0] + ss * q[:, 1], -ss * q[:, 0] + cs * q[:, 1]]) \
                + ps + delta[s_idx][:, [0, 2]]
            rhs = pt + delta[t_idx][:, [0, 2]]
            out.append(((lhs - rhs) / sm).ravel())
            out.append(np.degrees(wrap(theta[t_idx] - theta[s_idx] - alpha, 2 * np.pi)) / sd)
        if self.has_phi:
            out.append(wrap(self.heading - np.degrees(theta[self.heading_idx]) - phi, 90.0) / self.heading_sigma)
        if self.has_h:
            out.append((self.floor + delta[self.floor_idx, 1] - h) / FLOOR_SIGMA_M)
        return np.concatenate(out) if out else np.zeros(1)

    def link_errors(self, x) -> np.ndarray:
        """Normalised error of each link: max of translation (per axis) and rotation."""
        if not self.links:
            return np.zeros(0)
        start = 4 * len(self.odo) if self.odo else 0
        r = self.residuals(x)[start:start + 3 * len(self.links)]
        trans = np.abs(r[:2 * len(self.links)]).reshape(-1, 2).max(axis=1)
        rot = np.abs(r[2 * len(self.links):])
        return np.maximum(trans, rot)


def solve(nodes: list[Node], odometry: list[Odometry], links: list[Link]) -> Solution:
    links = list(links)
    rejected: list[Link] = []
    use_heading = any(n.wall_yaw_deg is not None for n in nodes)
    for _ in range(4):
        problem = _Problem(nodes, odometry, links, use_heading)
        if len(nodes) < 2:
            break
        fit = least_squares(problem.residuals, problem.x0(), loss="soft_l1", f_scale=3.0, x_scale="jac")
        errors = problem.link_errors(fit.x)
        bad = [i for i, e in enumerate(errors) if e > PRUNE_SIGMAS]
        if bad:
            worst = int(np.argmax(errors))  # drop one at a time: one wrong link can make good ones look bad
            rejected.append(links.pop(worst))
            continue
        theta, delta, phi, h = problem.unpack(fit.x)
        if use_heading and problem.has_phi:
            spread = np.median(np.abs(wrap(problem.heading - np.degrees(theta[problem.heading_idx]) - phi, 90.0)))
            if spread > NON_MANHATTAN_DEG:
                use_heading = False
                continue
        return Solution(theta, delta, phi, h, use_heading and problem.has_phi, links, rejected)
    S = len(nodes)
    return Solution(np.zeros(S), np.zeros((S, 3)), None, None, False, links, rejected)
