"""Pose graph over submap anchors with 4 degrees of freedom per node.

ARKit keeps gravity well, so roll and pitch are trusted; drift lives in heading and position.
Node s carries a correction around its recorded anchor p_s:
    corrected(X) = yaw_matrix(theta_s) (X - p_s) + p_s + delta_s

Residuals, each divided by its standard deviation:
- odometry a -> b:  yaw_matrix(theta_a) (p_b - p_a) = (p_b + delta_b) - (p_a + delta_a),  theta_b = theta_a
- heading creep: the rate at which the heading correction changes, (theta_c - theta_b)/dt - (theta_b - theta_a)/dt,
  should itself change slowly. Gyro heading drift is a steady creep (6 degrees over walk 1a8384c3f6), so
  a constant creep costs nothing here while a zig-zag that follows noisy wall headings costs a lot.
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
RATE_CHANGE_SIGMA_DEG_S = 0.01   # how fast the heading creep rate may change, deg/s per submap step
PRUNE_SIGMAS = 4.0
NON_MANHATTAN_DEG = 3.0


def odometry_sigmas(distance_m: float) -> tuple[float, float]:
    """Standard deviations (metres, degrees) of ARKit's relative motion over one edge.
    The heading term is loose: steady creep is handled by the creep-rate residual instead."""
    return 0.01 + 0.01 * distance_m, 0.2 + 0.5 * distance_m


@dataclass
class Node:
    anchor: np.ndarray                  # (3,) recorded position
    wall_yaw_deg: float | None = None   # dominant wall direction mod 90, when reliable
    floor_y: float | None = None        # floor height seen from this node, when reliable
    wall_yaw_sigma_deg: float | None = None  # uncertainty of wall_yaw_deg; HEADING_SIGMA_DEG if None
    time_s: float | None = None         # anchor time; enables the heading-creep residual


@dataclass
class Odometry:
    a: int
    b: int
    sigma_m: float
    sigma_deg: float
    smooth: bool = True  # False across an ARKit jump: no creep continuity there


@dataclass
class Link:
    target: int
    source: int
    yaw_deg: float
    shift: np.ndarray  # (2,)
    sigma_m: float
    sigma_deg: float
    kind: str = "loop"  # "loop", "relocalization" or "anchor"


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


def _creep_triples(nodes, odometry) -> np.ndarray:
    """(a, b, c) for consecutive smooth odometry edges a->b, b->c with known times."""
    by_start = {e.a: e for e in odometry if e.smooth}
    triples = []
    for e in odometry:
        f = by_start.get(e.b)
        if e.smooth and f is not None and f.b != e.a:
            a, b, c = e.a, e.b, f.b
            times = [nodes[k].time_s for k in (a, b, c)]
            if None not in times and times[0] < times[1] < times[2]:
                triples.append((a, b, c))
    return np.array(triples, int).reshape(-1, 3)


class _Problem:
    def __init__(self, nodes, odometry, links, use_heading):
        self.S = len(nodes)
        self.P = np.array([n.anchor for n in nodes], float)
        self.T = np.array([np.nan if n.time_s is None else n.time_s for n in nodes], float)
        self.odo = odometry
        self.links = links
        self.triples = _creep_triples(nodes, odometry)
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

    def link_block(self, theta, delta):
        """(translation residuals (L, 2), rotation residuals (L,)), normalised."""
        P = self.P
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
        return (lhs - rhs) / sm, np.degrees(wrap(theta[t_idx] - theta[s_idx] - alpha, 2 * np.pi)) / sd

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
        if len(self.triples):
            a, b, c = self.triples.T
            rate_1 = np.degrees(theta[b] - theta[a]) / (self.T[b] - self.T[a])
            rate_2 = np.degrees(theta[c] - theta[b]) / (self.T[c] - self.T[b])
            out.append((rate_2 - rate_1) / RATE_CHANGE_SIGMA_DEG_S)
        if self.links:
            trans, rot = self.link_block(theta, delta)
            out.append(trans.ravel())
            out.append(rot)
        if self.has_phi:
            out.append(wrap(self.heading - np.degrees(theta[self.heading_idx]) - phi, 90.0) / self.heading_sigma)
        if self.has_h:
            out.append((self.floor + delta[self.floor_idx, 1] - h) / FLOOR_SIGMA_M)
        return np.concatenate(out) if out else np.zeros(1)

    def link_errors(self, x) -> np.ndarray:
        """Normalised error of each link: max of translation (per axis) and rotation."""
        if not self.links:
            return np.zeros(0)
        theta, delta, _, _ = self.unpack(x)
        trans, rot = self.link_block(theta, delta)
        return np.maximum(np.abs(trans).max(axis=1), np.abs(rot))


def solve(nodes: list[Node], odometry: list[Odometry], links: list[Link]) -> Solution:
    links = list(links)
    rejected: list[Link] = []
    use_heading = any(n.wall_yaw_deg is not None for n in nodes)
    for _ in range(4 + len(links)):
        problem = _Problem(nodes, odometry, links, use_heading)
        if len(nodes) < 2:
            break
        fit = least_squares(problem.residuals, problem.x0(), loss="soft_l1", f_scale=3.0, x_scale="jac")
        errors = problem.link_errors(fit.x)
        prunable = [i for i, e in enumerate(errors) if e > PRUNE_SIGMAS and links[i].kind == "loop"]
        if prunable:
            worst = max(prunable, key=lambda i: errors[i])  # one at a time: a wrong link can make good ones look bad
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
