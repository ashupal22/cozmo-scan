"""Drift correction for a LiDAR walk (the brief's "drift accountability" gate).

1. Split the walk at pose jumps. A jump is ARKit relocalising: it snaps its map back onto what it saw
   earlier, and the frames after the snap are in the corrected map. On walk 1a8384c3f6 the snap was
   about 60 cm in the last second, landing 10 cm from where the walk began. So a jump becomes a
   relocalisation link (the map change between the frames just before and after it) plus, when the
   camera landed near the start, an anchor tying the post-jump frames to the start: a loop closure
   that ARKit found for us.
2. Cut each segment into submaps of about SUBMAP_S seconds. From its own points, each submap measures
   the dominant wall direction (mod 90, with its uncertainty) and the floor height. Floor heights far
   from the walk's typical floor are dropped: a table top seen up close is not the floor.
3. First pass: solve the pose graph (cozmo.slam.posegraph) from walls, floor and snaps alone.
4. Propose loop closures between submaps that saw the same walls far apart in time, match them on the
   maps straightened by the first pass (a raw 20-second map is itself bent by drift, which biased the
   matches), keep those the wall matcher verifies on the overlapping part, and solve again with them.
   The search is bounded by how far drift can have grown. Each frame's correction is interpolated
   between the anchors of neighbouring submaps in the same segment.
5. Apply the correction to fused points, normals and the camera path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from cozmo.geometry.fusion import PointSet
from cozmo.geometry.planes import find_floors
from cozmo.geometry.walls import dominant_yaw
from cozmo.slam.matching import match_walls, rot2, two_wall_directions, wrap, yaw_of
from cozmo.slam.posegraph import Link, Node, Odometry, odometry_sigmas, solve

JUMP_M = 0.10
SUBMAP_S = 4.0
MIN_TAIL_S = 1.5
WALL_BAND_M = (-1.3, 0.3)          # wall points between these heights relative to the camera
WALL_NORMAL_MAX_Y = 0.2
MIN_HEADING_POINTS = 300
MIN_HEADING_SHARE = 0.15           # LiDAR normals are noisy: on real walks 15-45% fall within 3 degrees
HEADING_SYSTEMATIC_DEG = 0.7       # scatter between submaps on c00a170fe1 beyond the statistical error
MIN_FLOOR_AREA_M2 = 1.0
FLOOR_GATE_M = 0.05                # vertical drift stays within a few cm; a table top does not
MIN_CAMERA_ABOVE_FLOOR_M = 1.0
LOCAL_MAP_REACH = 2                # submaps on each side that form a loop-closure local map
LOOP_MIN_GAP_S = 20.0
LOOP_RADIUS_M = 3.0
LOOP_CANDIDATES = 3
LOOP_MIN_POINTS = 400
LOOP_MIN_SHARED_BEFORE = 0.2       # share of source walls within 30 cm of target walls before matching
LOOP_MIN_OVERLAP = 0.3
LOOP_MAX_OVERLAP_RESIDUAL_M = 0.02
LOOP_MIN_INLIERS = 0.2
LOOP_MAX_AMBIGUITY = 0.85
LOOP_SIGMA = (0.03, 0.5)
RELOCALIZATION_SIGMA = (0.05, 1.0)
ACROSS_JUMP_ODOMETRY_SIGMA = (0.5, 10.0)
BREAK_ODOMETRY_SIGMA = (0.5, 20.0)  # video: tracking lost for a few frames; the step across is a rough guess
BREAK_LOOP_SHIFT_M = 0.5            # each tracking break between two places widens the loop search this much
ANCHOR_SIGMA = (0.05, 1.0)
EARLY_WALK_FRACTION = 0.2          # "early" = the first 20% of the distance walked: little drift yet
MIN_EARLY_WALK_M = 5.0
ANCHOR_RADIUS_M = 1.5
ANCHOR_MIN_AGREEMENT = 0.2         # share of post-jump wall points within 5 cm of early walls
ANCHOR_CHECK_MIN_POINTS = 200
MAP_CELL_M = 0.02
CHUNK = 2_000_000


def drift_bounds(path_between_m: float) -> tuple[float, float]:
    """How far (degrees, metres) drift can plausibly have grown over this much walking.
    Walk 1a8384c3f6: about 6 degrees and 60 cm over 54 m."""
    return min(8.0, 1.0 + 0.15 * path_between_m), min(1.2, 0.1 + 0.02 * path_between_m)


def find_jumps(positions: np.ndarray, threshold_m: float = JUMP_M) -> list[tuple[int, int]]:
    """(last frame before, first frame after) for each jump; consecutive jumping steps form one jump."""
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    spans: list[list[int]] = []
    for i in np.nonzero(steps > threshold_m)[0]:
        if spans and i == spans[-1][1]:
            spans[-1][1] = int(i + 1)
        else:
            spans.append([int(i), int(i + 1)])
    return [(a, b) for a, b in spans]


def relocalization(capture, before: int, after: int, history: int = 3):
    """Map change at a jump, as a link measurement (yaw_deg, shift) with old-map points
    q_old = rot2(yaw) q_new + shift, plus the jump size (metres, degrees) for the report."""
    p = capture.positions
    lo = max(before - history, 0)
    velocity = (p[before] - p[lo]) / max(before - lo, 1)
    expected = p[before] + velocity * (after - before)       # where the camera was, in the old map
    gamma = yaw_of(capture.rotation(after) @ capture.rotation(before).T)
    t_new = p[after, [0, 2]] - rot2(gamma) @ expected[[0, 2]]  # new = rot2(gamma) old + t_new
    alpha = -gamma
    shift = -rot2(alpha) @ t_new
    return float(np.degrees(alpha)), shift, float(np.linalg.norm(p[after] - expected)), float(np.degrees(gamma))


def heading_of(normals_xz: np.ndarray) -> tuple[float, float] | None:
    """Dominant wall direction (degrees mod 90) and its uncertainty, or None if unreliable."""
    if len(normals_xz) < MIN_HEADING_POINTS:
        return None
    yaw, share = dominant_yaw(normals_xz)
    if share < MIN_HEADING_SHARE:
        return None
    deviation = wrap(np.degrees(np.arctan2(normals_xz[:, 1], normals_xz[:, 0])) - yaw, 90.0)
    near = deviation[np.abs(deviation) < 10.0]
    spread = 1.4826 * float(np.median(np.abs(near - np.median(near))))
    statistical = spread / np.sqrt(max(len(near) / 10.0, 1.0))  # neighbouring normals are correlated
    return yaw, float(np.hypot(statistical, HEADING_SYSTEMATIC_DEG))


@dataclass
class Submap:
    segment: int
    frames: np.ndarray
    anchor: int
    wall_xz: np.ndarray
    wall_normals: np.ndarray
    wall_frames: np.ndarray
    node: Node


@dataclass
class FrameCorrection:
    theta: np.ndarray       # (N,) radians
    positions: np.ndarray   # (N, 3) corrected camera positions
    recorded: np.ndarray    # (N, 3) recorded camera positions
    valid: np.ndarray       # (N,) False for frames inside a jump

    def transform_xz(self, xz: np.ndarray, frames: np.ndarray) -> np.ndarray:
        th = self.theta[frames]
        rel = xz - self.recorded[frames][:, [0, 2]]
        c, s = np.cos(th), np.sin(th)
        return np.column_stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]]) \
            + self.positions[frames][:, [0, 2]]

    def apply(self, points: PointSet) -> PointSet:
        xyz = np.empty_like(points.xyz)
        normal = np.empty_like(points.normal)
        camera_y = np.empty_like(points.camera_y)
        for start in range(0, len(points), CHUNK):
            sl = slice(start, start + CHUNK)
            f = points.frame[sl]
            th = self.theta[f]
            c, s = np.cos(th), np.sin(th)
            rel = points.xyz[sl].astype(np.float64) - self.recorded[f]
            xyz[sl] = np.column_stack([c * rel[:, 0] + s * rel[:, 2], rel[:, 1],
                                       -s * rel[:, 0] + c * rel[:, 2]]) + self.positions[f]
            n = points.normal[sl]
            normal[sl] = np.column_stack([c * n[:, 0] + s * n[:, 2], n[:, 1], -s * n[:, 0] + c * n[:, 2]])
            camera_y[sl] = self.positions[f, 1]
        keep = self.valid[points.frame]
        return PointSet(xyz[keep], normal[keep], points.frame[keep], camera_y[keep])


@dataclass
class DriftReport:
    jumps: list[dict] = field(default_factory=list)
    submaps: int = 0
    heading_priors: int = 0
    floor_priors: int = 0
    floors_dropped: int = 0
    heading_priors_used: bool = False
    loops_proposed: int = 0
    loops_accepted: int = 0
    loops_rejected_by_graph: int = 0
    relocalizations: int = 0
    anchors: int = 0
    anchor_wall_agreement: list = field(default_factory=list)
    max_heading_correction_deg: float = 0.0
    max_position_correction_m: float = 0.0
    wall_heading_p90_before_deg: float | None = None
    wall_heading_p90_after_deg: float | None = None
    wall_map_area_before_m2: float = 0.0
    wall_map_area_after_m2: float = 0.0
    seconds: float = 0.0

    def to_schema(self) -> dict:
        correction = []
        if self.jumps:
            correction.append("jump_cut")
        if self.loops_accepted or self.relocalizations or self.anchors:
            correction.append("loop_closure")
        if self.heading_priors_used:
            correction.append("wall_plane_factors")
        if self.floor_priors:
            correction.append("floor_plane_factor")
        breaks = sum(j.get("kind") == "tracking break" for j in self.jumps)
        if breaks:
            moves = f"{breaks} video tracking break(s) held loosely; "
        else:
            moves = (f"{len(self.jumps)} ARKit relocalisation jump(s) turned into links, "
                     f"{self.anchors} tied back to the start of the walk; ")
        notes = (moves +
                 f"{self.loops_accepted} of {self.loops_proposed} loop candidates accepted; "
                 f"wall heading p90 {self._fmt(self.wall_heading_p90_before_deg)} -> "
                 f"{self._fmt(self.wall_heading_p90_after_deg)} deg; "
                 f"wall map area {self.wall_map_area_before_m2:.1f} -> {self.wall_map_area_after_m2:.1f} m2; "
                 f"max position correction {self.max_position_correction_m:.2f} m")
        return {"enabled": True, "correction": correction,
                "loop_closures": self.loops_accepted + self.relocalizations + self.anchors,
                "heading_change_deg": round(self.max_heading_correction_deg, 2), "notes": notes}

    @staticmethod
    def _fmt(v):
        return "n/a" if v is None else f"{v:.2f}"


def _segments(n_frames: int, jumps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    segments, start = [], 0
    for before, after in jumps:
        segments.append((start, before))
        start = after
    segments.append((start, n_frames - 1))
    return segments


def _group_by_time(frames: np.ndarray, times: np.ndarray) -> list[list[int]]:
    groups, current = [], [int(frames[0])]
    for f in frames[1:]:
        if times[f] - times[current[0]] >= SUBMAP_S:
            groups.append(current)
            current = [int(f)]
        else:
            current.append(int(f))
    groups.append(current)
    if len(groups) > 1 and times[groups[-1][-1]] - times[groups[-1][0]] < MIN_TAIL_S:
        tail = groups.pop()
        groups[-1] = groups[-1] + tail
    return groups


def gate_floors(submaps: list[Submap], positions: np.ndarray) -> int:
    """Drop floor heights that are not the walk's floor. Returns how many were dropped."""
    heights = [s.node.floor_y for s in submaps if s.node.floor_y is not None]
    if not heights:
        return 0
    typical = float(np.median(heights))
    dropped = 0
    for s in submaps:
        y = s.node.floor_y
        if y is None:
            continue
        if abs(y - typical) > FLOOR_GATE_M or positions[s.anchor, 1] - y < MIN_CAMERA_ABOVE_FLOOR_M:
            s.node.floor_y = None
            dropped += 1
    return dropped


def _build_submaps(capture, points: PointSet, segments) -> list[Submap]:
    t = capture.timestamps
    keyframes = np.unique(points.frame)
    starts = np.searchsorted(points.frame, keyframes, "left")
    ends = np.searchsorted(points.frame, keyframes, "right")
    span = {int(f): (int(a), int(b)) for f, a, b in zip(keyframes, starts, ends)}
    submaps = []
    for g, (first, last) in enumerate(segments):
        kfs = keyframes[(keyframes >= first) & (keyframes <= last)]
        if len(kfs) == 0:
            continue
        for group in _group_by_time(kfs, t):
            idx = np.concatenate([np.arange(*span[f]) for f in group])
            xyz, nrm = points.xyz[idx], points.normal[idx]
            rel = xyz[:, 1] - points.camera_y[idx]
            wall = (np.abs(nrm[:, 1]) < WALL_NORMAL_MAX_Y) & (rel > WALL_BAND_M[0]) & (rel < WALL_BAND_M[1])
            wxz = xyz[wall][:, [0, 2]].astype(np.float64)
            wn = nrm[wall][:, [0, 2]].astype(np.float64)
            wn /= np.linalg.norm(wn, axis=1, keepdims=True) + 1e-9
            _, keep = np.unique(np.floor(wxz / MAP_CELL_M).astype(np.int64), axis=0, return_index=True)
            keep.sort()
            wxz, wn, wf = wxz[keep], wn[keep], points.frame[idx][wall][keep]

            heading = heading_of(wn)
            floors = find_floors(points.subset(idx[::4]))
            floor = floors[0].height if floors and floors[0].area_m2 >= MIN_FLOOR_AREA_M2 else None
            middle = (t[group[0]] + t[group[-1]]) / 2
            anchor = min(group, key=lambda f: abs(t[f] - middle))
            node = Node(capture.positions[anchor].astype(float), heading[0] if heading else None, floor,
                        heading[1] if heading else None, float(t[anchor]))
            submaps.append(Submap(g, np.array(group), anchor, wxz, wn, wf, node))
    return submaps


def _local_map(submaps: list[Submap], k: int, maps=None):
    """Walls of submap k and its neighbours in the same segment. `maps` optionally replaces each
    submap's (wall_xz, wall_normals), e.g. with pre-corrected copies."""
    ids = [j for j in range(k - LOCAL_MAP_REACH, k + LOCAL_MAP_REACH + 1)
           if 0 <= j < len(submaps) and submaps[j].segment == submaps[k].segment]
    parts = [maps[j] if maps is not None else (submaps[j].wall_xz, submaps[j].wall_normals) for j in ids]
    xz = np.vstack([p[0] for p in parts])
    normals = np.vstack([p[1] for p in parts])
    _, keep = np.unique(np.floor(xz / MAP_CELL_M).astype(np.int64), axis=0, return_index=True)
    return xz[keep], normals[keep]


def to_recorded_link(target: int, source: int, yaw_deg: float, shift: np.ndarray, anchors_xz: np.ndarray,
                     theta: np.ndarray, delta_xz: np.ndarray, sigmas, kind: str = "loop") -> Link:
    """A match found between pre-corrected maps, C_a(q_a) = rot2(yaw) C_b(q_b) + shift, re-expressed in
    recorded coordinates, q_a = rot2(yaw') q_b + shift', using each node's first-pass correction
    C_k(q) = rot2(theta_k) (q - p_k) + p_k + delta_k."""
    a, b = target, source
    p_a, p_b = anchors_xz[a], anchors_xz[b]
    alpha = np.radians(yaw_deg)
    yaw_rec = alpha + theta[b] - theta[a]
    inner = rot2(alpha) @ (p_b + delta_xz[b] - rot2(theta[b]) @ p_b) + shift - p_a - delta_xz[a]
    shift_rec = rot2(-theta[a]) @ inner + p_a
    return Link(a, b, float(np.degrees(yaw_rec)), shift_rec, *sigmas, kind=kind)


def _anchor_links(positions: np.ndarray, submaps: list[Submap], n_segments: int):
    """After an ARKit jump the camera is back in ARKit's reference frame: ARKit recognised a place it
    mapped earlier and snapped onto it. When that place is near the start of the walk, where drift is
    still small, the post-jump submap and the nearest early submap share one frame: identity link.
    On walk 1a8384c3f6 the camera landed 10 cm from where the walk began."""
    walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])
    early_m = max(EARLY_WALK_FRACTION * walked[-1], MIN_EARLY_WALK_M)
    early = [k for k, s in enumerate(submaps) if s.segment == 0 and walked[s.anchor] <= early_m]
    links, agreement = [], []
    for g in range(1, n_segments):
        first = next((k for k, s in enumerate(submaps) if s.segment == g), None)
        if first is None or not early:
            continue
        here = submaps[first].node.anchor[[0, 2]]
        nearest = min(early, key=lambda k: np.linalg.norm(submaps[k].node.anchor[[0, 2]] - here))
        if np.linalg.norm(submaps[nearest].node.anchor[[0, 2]] - here) > ANCHOR_RADIUS_M:
            continue
        mine = submaps[first].wall_xz
        share = None
        if len(mine) >= ANCHOR_CHECK_MIN_POINTS:
            d, _ = cKDTree(_local_map(submaps, nearest)[0]).query(mine)
            share = float(np.mean(d < 0.05))
            agreement.append(round(share, 3))
            if share < ANCHOR_MIN_AGREEMENT:
                continue  # the walls do not agree: do not trust the snap as a return to the start
        links.append(Link(nearest, first, 0.0, np.zeros(2), *ANCHOR_SIGMA, kind="anchor"))
    return links, agreement


def _loop_links(capture, submaps: list[Submap], correction: "FrameCorrection", first_pass, breaks: bool = False):
    """Loop closures matched on maps already straightened by the first pass (walls, floor, snaps), so a
    20-second local map is not itself bent by drift; each match is then re-expressed in recorded
    coordinates for the final graph. With `breaks` (video), segments are joined only loosely: the first
    pass fixes their headings from the walls, but their placement is off by up to about half a metre per
    break, so the search between places in different segments widens accordingly."""
    t = capture.timestamps
    positions = np.asarray(capture.positions, float)
    walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])
    corrected = []
    for sm in submaps:
        th = correction.theta[sm.wall_frames]
        c, si = np.cos(th), np.sin(th)
        n = sm.wall_normals
        corrected.append((correction.transform_xz(sm.wall_xz, sm.wall_frames),
                          np.column_stack([c * n[:, 0] + si * n[:, 1], -si * n[:, 0] + c * n[:, 1]])))
    maps = [_local_map(submaps, k, corrected) for k in range(len(submaps))]
    usable = [len(xz) >= LOOP_MIN_POINTS and two_wall_directions(n) for xz, n in maps]
    here = np.array([correction.positions[sm.anchor][[0, 2]] for sm in submaps])
    anchors_xz = np.array([sm.node.anchor[[0, 2]] for sm in submaps])
    delta_xz = first_pass.delta[:, [0, 2]]
    trees: dict[int, cKDTree] = {}
    links, proposed = [], 0
    for b in range(len(submaps)):
        if not usable[b]:
            continue
        near = [(float(np.linalg.norm(here[b] - here[a])), a) for a in range(b)
                if usable[a] and t[submaps[b].anchor] - t[submaps[a].anchor] >= LOOP_MIN_GAP_S]
        for distance, a in sorted(near)[:LOOP_CANDIDATES]:
            if distance > LOOP_RADIUS_M:
                break
            yaw_bound, shift_bound = drift_bounds(abs(walked[submaps[b].anchor] - walked[submaps[a].anchor]))
            if breaks:
                shift_bound += BREAK_LOOP_SHIFT_M * abs(submaps[b].segment - submaps[a].segment)
            tree = trees.setdefault(a, cKDTree(maps[a][0]))
            d0, _ = tree.query(maps[b][0], distance_upper_bound=min(max(0.3, shift_bound), 0.6))
            if np.mean(np.isfinite(d0)) < LOOP_MIN_SHARED_BEFORE:
                continue  # the two places do not show the same walls
            proposed += 1
            m = match_walls(maps[a][0], maps[a][1], maps[b][0], yaw_range_deg=(-yaw_bound, yaw_bound),
                            max_shift_m=shift_bound)
            if m is None:
                continue
            centre = maps[b][0].mean(axis=0)
            moved = float(np.linalg.norm(m.apply(centre[None])[0] - centre))
            if (m.overlap_fraction >= LOOP_MIN_OVERLAP and m.overlap_median_m <= LOOP_MAX_OVERLAP_RESIDUAL_M
                    and m.inlier_fraction >= LOOP_MIN_INLIERS and m.ambiguity <= LOOP_MAX_AMBIGUITY
                    and abs(m.yaw_deg) <= yaw_bound and moved <= shift_bound + 0.05):
                links.append(to_recorded_link(a, b, m.yaw_deg, m.shift, anchors_xz, first_pass.theta, delta_xz,
                                              LOOP_SIGMA))
    return links, proposed


def _frame_correction(capture, submaps, segments, jumps, solution) -> FrameCorrection:
    n = len(capture)
    t, recorded = capture.timestamps, np.asarray(capture.positions, float)
    theta, positions = np.zeros(n), recorded.copy()
    valid = np.ones(n, bool)
    for before, after in jumps:
        valid[before + 1:after] = False
    anchors_all = np.array([s.node.anchor for s in submaps])
    for g, (first, last) in enumerate(segments):
        ids = np.array([k for k, s in enumerate(submaps) if s.segment == g], int)
        if len(ids) == 0:
            continue
        frames = np.arange(first, last + 1)
        at = t[[submaps[k].anchor for k in ids]]
        k0 = np.clip(np.searchsorted(at, t[frames], "right") - 1, 0, len(ids) - 1)
        k1 = np.minimum(k0 + 1, len(ids) - 1)
        gap = at[k1] - at[k0]
        w = np.where(gap > 0, np.clip((t[frames] - at[k0]) / np.where(gap > 0, gap, 1.0), 0.0, 1.0), 0.0)

        def via(k):
            node = ids[k]
            th = solution.theta[node]
            anchor = anchors_all[node]
            rel = recorded[frames] - anchor
            c, s = np.cos(th), np.sin(th)
            return np.column_stack([c * rel[:, 0] + s * rel[:, 2], rel[:, 1], -s * rel[:, 0] + c * rel[:, 2]]) \
                + anchor + solution.delta[node]

        theta[frames] = (1 - w) * solution.theta[ids[k0]] + w * solution.theta[ids[k1]]
        positions[frames] = (1 - w)[:, None] * via(k0) + w[:, None] * via(k1)
    return FrameCorrection(theta, positions, recorded, valid)


def _map_area(xz: np.ndarray) -> float:
    return float(len(np.unique(np.floor(xz / MAP_CELL_M).astype(np.int64), axis=0)) * MAP_CELL_M ** 2)


def estimate_drift(capture, points: PointSet, jumps: list[tuple[int, int]] | None = None,
                   relocalized: bool = True) -> tuple[FrameCorrection, DriftReport]:
    """Per-frame corrections for a walk. `capture` needs timestamps, positions, rotation(i) and len();
    `points` must come from that capture (frame indices are sorted here if needed).

    `jumps` are (frame before, frame after); frames strictly between are dropped. None finds them in the
    positions, which suits ARKit at 60 fps. With `relocalized` (ARKit), the pose after a jump is ARKit's
    relocalised pose: the jump becomes a link, and a return near the start an anchor. Without it (video),
    a jump is a tracking break: the step across it is only a rough guess, so it is held loosely and wall
    directions, floors and loop closures place what follows."""
    t0 = time.time()
    if np.any(np.diff(points.frame) < 0):
        order = np.argsort(points.frame, kind="stable")
        points = PointSet(points.xyz[order], points.normal[order], points.frame[order], points.camera_y[order])
    report = DriftReport()
    positions = np.asarray(capture.positions, float)
    jumps = find_jumps(positions) if jumps is None else list(jumps)
    segments = _segments(len(capture), jumps)
    submaps = _build_submaps(capture, points, segments)
    report.submaps = len(submaps)
    report.floors_dropped = gate_floors(submaps, positions)
    report.heading_priors = sum(s.node.wall_yaw_deg is not None for s in submaps)
    report.floor_priors = sum(s.node.floor_y is not None for s in submaps)

    odometry, links = [], []
    for k in range(len(submaps) - 1):
        a, b = submaps[k], submaps[k + 1]
        distance = float(np.linalg.norm(b.node.anchor - a.node.anchor))
        if a.segment == b.segment:
            odometry.append(Odometry(k, k + 1, *odometry_sigmas(distance)))
        elif not relocalized:
            before, after = jumps[a.segment]
            report.jumps.append({"frame": after, "kind": "tracking break", "frames_dropped": after - before - 1})
            odometry.append(Odometry(k, k + 1, *BREAK_ODOMETRY_SIGMA, smooth=False))
        else:
            before, after = jumps[a.segment]
            yaw, shift, metres, degrees = relocalization(capture, before, after)
            report.jumps.append({"frame": after, "metres": round(metres, 3), "degrees": round(degrees, 2)})
            links.append(Link(k, k + 1, yaw, shift, *RELOCALIZATION_SIGMA, kind="relocalization"))
            odometry.append(Odometry(k, k + 1, *ACROSS_JUMP_ODOMETRY_SIGMA, smooth=False))
    if relocalized:
        anchors, report.anchor_wall_agreement = _anchor_links(positions, submaps, len(segments))
        links += anchors

    if len(submaps) < 2:
        n = len(capture)
        report.seconds = time.time() - t0
        return FrameCorrection(np.zeros(n), positions.copy(), positions, np.ones(n, bool)), report

    nodes = [s.node for s in submaps]
    first_pass = solve(nodes, odometry, links)
    straightened = _frame_correction(capture, submaps, segments, jumps, first_pass)
    loops, report.loops_proposed = _loop_links(capture, submaps, straightened, first_pass, breaks=not relocalized)
    solution = solve(nodes, odometry, links + loops)
    report.heading_priors_used = solution.heading_priors_used
    report.loops_accepted = sum(l.kind == "loop" for l in solution.links_used)
    report.loops_rejected_by_graph = sum(l.kind == "loop" for l in solution.links_rejected)
    report.relocalizations = sum(l.kind == "relocalization" for l in solution.links_used)
    report.anchors = sum(l.kind == "anchor" for l in solution.links_used)

    correction = _frame_correction(capture, submaps, segments, jumps, solution)
    report.max_heading_correction_deg = float(np.degrees(np.abs(correction.theta).max()))
    report.max_position_correction_m = float(np.linalg.norm(correction.positions - correction.recorded, axis=1).max())

    headed = [k for k, s in enumerate(submaps) if s.node.wall_yaw_deg is not None]
    if headed and solution.building_yaw_deg is not None:
        yaws = np.array([submaps[k].node.wall_yaw_deg for k in headed])
        before = np.abs(wrap(yaws - solution.building_yaw_deg, 90.0))
        after = np.abs(wrap(yaws - np.degrees(solution.theta[headed]) - solution.building_yaw_deg, 90.0))
        report.wall_heading_p90_before_deg = float(np.percentile(before, 90))
        report.wall_heading_p90_after_deg = float(np.percentile(after, 90))
    all_xz = np.vstack([s.wall_xz for s in submaps])
    all_frames = np.concatenate([s.wall_frames for s in submaps])
    report.wall_map_area_before_m2 = _map_area(all_xz)
    report.wall_map_area_after_m2 = _map_area(correction.transform_xz(all_xz, all_frames))
    report.seconds = time.time() - t0
    return correction, report
