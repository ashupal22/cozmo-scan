"""Room layout from walls (walls first), for homes whose walls meet at right angles.

1. Wall faces: vertical surfaces seen between the evidence height (1.1-1.5 m, adapted to how high this
   walk saw walls) and DOOR_HEAD_M over the floor form straight face lines along the two main wall
   directions. Sofas, beds, tables and counters are lower; the wall above a door is higher.
2. Each face line gets a profile along its length: wall, door or open. A gap in a wall is a passage only
   if the walked path crosses the line there, or the floor is seen right at the line (under a solid wall
   it cannot be). Free floor on both sides is not enough: every inner wall has that. A passage empty
   from knee to head height is a door if 0.55-1.4 m wide and open if wider; every other gap is wall,
   hidden behind something. Wall runs ending near the wall they meet are joined to it, and the two
   faces of one wall share what either saw, so rooms cannot leak around corners or along a wall.
3. The face lines cut the plan into rectangular cells. A minimum cut labels each cell inside or outside:
   seen floor and the walked path pull cells inside, unseen space pushes weakly outside, and cutting
   is cheap along walls and expensive through open space. So a room extends behind furniture up to its
   walls, instead of stopping where the floor was hidden (Mura et al. 2014, Ochmann et al. 2019).
4. Rooms: inside cells connected without crossing a wall or a door. A second, independent clue splits
   rooms further: where the seen free space narrows at a doorway (rooms.segment_rooms). Where the phone
   saw walls only low, walls-first merges rooms that the narrowing still separates.
5. Outlines: the union of each room's cells, short jogs removed, every edge re-fitted to the wall
   points facing into the room. Doors become openings on the walls of the rooms they join; where the
   walk went from one room into another with no door found between them, a door of nominal width is
   added where it crossed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, maximum_flow
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from cozmo.geometry.fusion import PointSet
from cozmo.geometry.planes import HorizontalPlane
from cozmo.geometry.rooms import CLOSE_RADIUS_M, PlanGrid, _disk, plan_maps, segment_rooms
from cozmo.geometry.walls import OpeningOnWall, RoomOutline, WallSegment, _to_aligned, dominant_yaw

WALL_EVIDENCE_RANGE_M = (1.1, 1.5)  # wall points at or above the evidence height count as wall, not
WALL_EVIDENCE_PERCENTILE = 70       # furniture. It adapts to how high this walk saw walls: two of our
                                    # three walks saw most walls only up to 1.5 m (phone pointed down)
DOOR_HEAD_M = 2.0              # ...and below this: the wall above a door (seen when the phone looks up)
                               # would otherwise hide the doorway
PASSAGE_BAND_M = (0.3, 1.9)    # a doorway is empty in this band
MAX_WALL_HEIGHT_M = 3.5
FACE_COS = np.cos(np.radians(12))
MANHATTAN_MIN_SHARE = 0.6      # share of vertical normals within 12 deg of the two main directions
ACROSS_BIN_M = 0.01
ALONG_BIN_M = 0.05
LINE_MIN_M = 0.3               # a face line needs this much seen wall
LINE_MERGE_M = 0.03
LINE_BAND_M = 0.03
TALL_MIN_POINTS = 2            # per along-bin
WALL_RUN_TALL_M = 0.10         # a continuous face is wall if at least this much of it reached wall height
FILL_GAP_M = 0.10
DOOR_MIN_M, DOOR_MAX_M = 0.55, 1.40
CORNER_SNAP_M = 0.30           # a wall run ending this close to the wall it meets is joined to it
CROSSING_REACH_M = 0.40        # a walk through a wall proves a passage this far either side of the crossing
JAMB_SEARCH_M = 0.15
SEEN_OPENING_SHARE = 0.5       # floor seen at the line over this share of a gap: a seen opening
NOMINAL_DOOR_M = 0.80          # width reported, with a wide range, when neither jamb was seen
WALL_THICKNESS_M = (0.05, 0.35)  # the two faces of one wall are this far apart
BOX_MARGIN_M = 0.5
W_FREE, W_UNSEEN, W_OPEN, W_WALL = 10.0, 3.0, 3.0, 0.05   # graph-cut costs per m2 / per m: unseen space
                                                          # up to ~1 m deep behind a boundary without wall
                                                          # evidence joins the room (furniture); more does not
FLOW_SCALE = 1000
MIN_ROOM_M2 = 1.0
SPLIT_MIN_M2 = 2.0             # split a room that holds two floor-first rooms with this much of each
MIN_ROOM_WIDTH_M = 0.4         # thinner pieces are the inside of a wall at a doorway, not rooms
JOG_M = 0.08
REFIT_BAND_M = 0.04
REFIT_MIN_POINTS = 30
FACING_DOT = 0.7
DOOR_PAIR_M = 0.40             # the two faces of one wall are at most this far apart
VOXEL_M = (0.02, 0.05)         # plan and height size used to thin wall points

OPEN, WALL, DOOR = 0, 1, 2


@dataclass
class FaceLine:
    axis: int            # 0: the line u = c (wall normals along u); 1: the line v = c
    c: float
    profile: np.ndarray  # OPEN / WALL / DOOR per along-bin
    doors: list = field(default_factory=list)  # (start, end, width measured?) along the line

    def state(self, lo: float, hi: float, along0: float) -> tuple[float, float]:
        """Share of [lo, hi] that is wall, and that is door."""
        k0 = int(np.floor((lo - along0) / ALONG_BIN_M))
        k1 = int(np.ceil((hi - along0) / ALONG_BIN_M))
        k0, k1 = max(k0, 0), min(max(k1, k0 + 1), len(self.profile))
        if k0 >= k1:
            return 0.0, 0.0
        seg = self.profile[k0:k1]
        return float(np.mean(seg == WALL)), float(np.mean(seg == DOOR))


@dataclass
class LayoutRoomMap:
    """Room labels on the plan grid, compatible with rooms.RoomMap where room_ceilings needs it."""
    grid: PlanGrid
    labels: np.ndarray
    areas_m2: dict[int, float]
    doorways: list = field(default_factory=list)

    @property
    def room_ids(self) -> list[int]:
        return sorted(self.areas_m2)

    def room_of(self, xz: np.ndarray) -> np.ndarray:
        rows, cols = self.grid.index(xz)
        return self.labels[rows, cols]


@dataclass
class Layout:
    room_map: LayoutRoomMap
    outlines: dict[int, RoomOutline]
    openings: list[OpeningOnWall]
    yaw_deg: float
    info: dict


class NotManhattan(ValueError):
    """Walls do not meet at right angles often enough for this method."""


# ---------------------------------------------------------------------------------------------
def _thin(uv, h, nuv):
    keys = np.column_stack([np.floor(uv / VOXEL_M[0]), np.floor(h / VOXEL_M[1])]).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep.sort()
    return uv[keep], h[keep], nuv[keep]


def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs of True."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(padded)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


class _FreeLookup:
    """Free space (seen floor or walked path) looked up at aligned (u, v) coordinates."""

    def __init__(self, free: np.ndarray, grid: PlanGrid, R: np.ndarray):
        self.free, self.grid, self.R = free, grid, R

    def __call__(self, uv: np.ndarray) -> np.ndarray:
        rows, cols = self.grid.index(uv @ self.R)
        return self.free[rows, cols]


def _path_crossings(path_uv: np.ndarray, axis: int, c: float) -> np.ndarray:
    """Along-line positions where the walked path crosses the line (axis, c)."""
    d = path_uv[:, axis] - c
    i = np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    t = d[i] / (d[i] - d[i + 1])
    return path_uv[i, 1 - axis] + t * (path_uv[i + 1, 1 - axis] - path_uv[i, 1 - axis])


def _face_lines(uv, h, nuv, axis, along0, n_bins, evidence_m: float, floor_seen: _FreeLookup,
                path_uv: np.ndarray) -> list[FaceLine]:
    fam = np.abs(nuv[:, axis]) > FACE_COS
    across, along, hh = uv[fam, axis], uv[fam, 1 - axis], h[fam]
    tall = (hh >= evidence_m) & (hh < DOOR_HEAD_M)
    if tall.sum() < 20:
        return []
    # length-weighted histogram: one vote per (across bin, along bin), so dense close walls do not dominate
    keys = np.unique(np.column_stack([np.floor(across[tall] / ACROSS_BIN_M),
                                      np.floor(along[tall] / ALONG_BIN_M)]).astype(np.int64), axis=0)
    lo_bin = keys[:, 0].min() - 3
    hist = np.bincount(keys[:, 0] - lo_bin).astype(float)
    smooth = ndimage.uniform_filter1d(hist, 3)
    peaks, _ = find_peaks(smooth, height=LINE_MIN_M / ALONG_BIN_M, distance=3)
    order = np.argsort(across)
    s_across, s_along, s_h = across[order], along[order], hh[order]
    s_normal = nuv[fam, axis][order]

    candidates = []
    for p in peaks:
        c0 = (p + lo_bin + 0.5) * ACROSS_BIN_M
        i0, i1 = np.searchsorted(s_across, [c0 - 0.02, c0 + 0.02])
        sel = (s_h[i0:i1] >= evidence_m) & (s_h[i0:i1] < DOOR_HEAD_M)
        if sel.sum() < 10:
            continue
        candidates.append((float(np.median(s_across[i0:i1][sel])), float(smooth[p])))
    candidates.sort(key=lambda x: -x[1])
    chosen: list[float] = []
    for c, _ in candidates:
        if all(abs(c - o) >= LINE_MERGE_M for o in chosen):
            chosen.append(c)

    centres = along0 + (np.arange(n_bins) + 0.5) * ALONG_BIN_M
    lines = []
    for c in sorted(chosen):
        i0, i1 = np.searchsorted(s_across, [c - LINE_BAND_M, c + LINE_BAND_M])
        a, z = s_along[i0:i1], s_h[i0:i1]
        k = np.clip(np.floor((a - along0) / ALONG_BIN_M).astype(int), 0, n_bins - 1)
        tall_count = np.bincount(k[(z >= evidence_m) & (z < DOOR_HEAD_M)], minlength=n_bins)
        busy = np.bincount(k[(z >= PASSAGE_BAND_M[0]) & (z <= PASSAGE_BAND_M[1])], minlength=n_bins) >= 2
        # continuous faces at any height, judged as a whole: a face is wall if any part of it was seen at
        # wall height. Furniture hides the low part of a wall in places, and the phone saw the high part
        # only in places; a furniture face never reaches wall height anywhere.
        seen = np.bincount(k[(z >= PASSAGE_BAND_M[0]) & (z < DOOR_HEAD_M)], minlength=n_bins) >= TALL_MIN_POINTS
        merged: list[list[int]] = []
        for st, en in _intervals(seen):
            if merged and (st - merged[-1][1]) * ALONG_BIN_M <= FILL_GAP_M:
                merged[-1][1] = en
            else:
                merged.append([st, en])
        merged = [m for m in merged if (m[1] - m[0]) * ALONG_BIN_M >= 0.1
                  and (tall_count[m[0]:m[1]] >= TALL_MIN_POINTS).sum() * ALONG_BIN_M >= WALL_RUN_TALL_M]
        if sum(e - s_ for s_, e in merged) * ALONG_BIN_M < LINE_MIN_M:
            continue
        # a passage through the line: the walked path crosses it, or the floor is seen right at the line
        # (under a solid wall it cannot be); free floor on both sides is not enough: every inner wall has that
        through = np.zeros(n_bins, bool)
        crossing = np.clip(np.floor((_path_crossings(path_uv, axis, c) - along0) / ALONG_BIN_M).astype(int), 0, n_bins - 1)
        through[crossing] = True
        for side in (-1.0, 1.0):
            probe = np.column_stack([np.full(n_bins, c + side * 0.04), centres])
            if axis == 1:
                probe = probe[:, ::-1]
            if side < 0:
                seen_both = floor_seen(probe)
            else:
                seen_both &= floor_seen(probe)
        line = FaceLine(axis, c, np.zeros(n_bins, np.int8), [])
        line.busy, line.through, line.seen_floor = busy, through, seen_both
        line.facing = float(np.sign(np.median(s_normal[i0:i1])))
        line.along0, line.points_along, line.points_h = along0, a, z
        for st, en in merged:
            line.profile[st:en] = WALL
        for (_, e0), (s1, _) in zip(merged[:-1], merged[1:]):
            _classify_gap(line, e0, s1)
        lines.append(line)
    return lines


def _classify_gap(line: FaceLine, k0: int, k1: int, end_gap: bool = False) -> None:
    """Decide what the bins [k0, k1) between two wall runs (or a wall end and the wall it meets) are.

    - Floor seen right at the line over most of the gap (under a solid wall it cannot be): the whole gap
      is a seen opening, measured wall end to wall end; a door if 0.55-1.4 m wide, open if wider.
    - Otherwise, walked through: a door at the crossing. Its width is measured if both jambs are seen
      within reach, else nominal (and reported with a wide range).
    - Otherwise wall: hidden behind something. A walk through a barely-seen wall must not turn the whole
      unseen stretch into one long opening."""
    if k1 <= k0:
        return
    gap = slice(k0, k1)
    seen = ndimage.binary_closing(np.pad(line.seen_floor[gap], 2), structure=np.ones(5))[2:-2]
    crossings = np.nonzero(line.through[gap])[0]
    a, z = line.points_along, line.points_h
    near = (z >= PASSAGE_BAND_M[0]) & (z <= PASSAGE_BAND_M[1])

    def jambs(lo, hi):
        left = a[near & (a >= lo - JAMB_SEARCH_M) & (a < (lo + hi) / 2)]
        right = a[near & (a <= hi + JAMB_SEARCH_M) & (a > (lo + hi) / 2)]
        return (float(left.max()) if len(left) else None), (float(right.min()) if len(right) else None)

    g_lo, g_hi = line.along0 + k0 * ALONG_BIN_M, line.along0 + k1 * ALONG_BIN_M
    if np.mean(seen) >= SEEN_OPENING_SHARE:
        j_lo, j_hi = jambs(g_lo, g_hi)
        d_lo, d_hi = (j_lo if j_lo is not None else g_lo), (j_hi if j_hi is not None else g_hi)
        width = d_hi - d_lo
        if width > DOOR_MAX_M:
            line.profile[gap] = OPEN
        elif width >= DOOR_MIN_M:
            line.profile[gap] = DOOR
            line.doors.append((d_lo, d_hi, True))
        else:
            line.profile[gap] = WALL
        return
    line.profile[gap] = WALL
    if len(crossings) == 0:
        if end_gap and (k1 - k0) * ALONG_BIN_M > CORNER_SNAP_M:
            line.profile[gap] = OPEN  # the wall may simply end here
        return
    reach = int(round(CROSSING_REACH_M / ALONG_BIN_M))
    done = np.zeros(k1 - k0, bool)
    for k in crossings:
        if done[k]:
            continue
        o0, o1 = max(k - reach, 0), min(k + reach + 1, k1 - k0)
        done[o0:o1] = True
        c_lo, c_hi = line.along0 + (k0 + o0) * ALONG_BIN_M, line.along0 + (k0 + o1) * ALONG_BIN_M
        j_lo, j_hi = jambs(c_lo, c_hi)
        line.profile[k0 + o0:k0 + o1] = DOOR
        if j_lo is not None and j_hi is not None and DOOR_MIN_M <= j_hi - j_lo <= DOOR_MAX_M:
            line.doors.append((j_lo, j_hi, True))
        else:
            mid = line.along0 + (k0 + k + 0.5) * ALONG_BIN_M
            line.doors.append((mid - NOMINAL_DOOR_M / 2, mid + NOMINAL_DOOR_M / 2, False))


def _pair_wall_faces(lines: list[FaceLine]) -> int:
    """The two faces of one wall (facing opposite ways, WALL_THICKNESS_M apart) share what either saw:
    one room may have seen the left half of a wall and the other room the right half. Without this,
    rooms leak into each other along the inside of the wall. Returns the number of pairs."""
    pairs = 0
    ordered = sorted(lines, key=lambda l: l.c)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            gap = b.c - a.c
            if gap > WALL_THICKNESS_M[1]:
                break
            if gap < WALL_THICKNESS_M[0] or not (a.facing < 0 < b.facing):
                continue  # a faces towards smaller c and b towards larger c: the wall is between them
            both = np.maximum(a.profile, b.profile)   # DOOR > WALL > OPEN
            overlap = (a.profile == WALL) & (b.profile == WALL)
            if overlap.sum() * ALONG_BIN_M < 0.2:
                continue  # not seen as the same wall anywhere
            a.profile[:], b.profile[:] = both, both
            pairs += 1
    return pairs


def _close_ends(lines: list[FaceLine], perpendicular: list[FaceLine]) -> None:
    """Carry each wall run's ends to the wall they meet (a corner or a T), deciding the gap as above."""
    coords = np.array(sorted(l.c for l in perpendicular)) if perpendicular else np.zeros(0)
    for line in lines:
        runs = _intervals(line.profile == WALL)
        if not runs or len(coords) == 0:
            continue
        first, last = runs[0][0], runs[-1][1]
        start = line.along0 + first * ALONG_BIN_M
        end = line.along0 + last * ALONG_BIN_M
        before = coords[(coords < start) & (coords >= start - DOOR_MAX_M - 0.1)]
        after = coords[(coords > end) & (coords <= end + DOOR_MAX_M + 0.1)]
        if len(before):
            k = int(np.floor((before.max() - line.along0) / ALONG_BIN_M))
            _classify_gap(line, max(k, 0), first, end_gap=True)
        if len(after):
            k = int(np.ceil((after.min() - line.along0) / ALONG_BIN_M))
            _classify_gap(line, last, min(k, len(line.profile)), end_gap=True)


def _min_cut(n: int, t_source: np.ndarray, t_sink: np.ndarray, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Cells on the source (inside) side of the minimum cut."""
    S, T = n, n + 1
    rows = [np.full(n, S), np.arange(n), edges[:, 0], edges[:, 1]]
    cols = [np.arange(n), np.full(n, T), edges[:, 1], edges[:, 0]]
    caps = [t_source, t_sink, weights, weights]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    caps = np.maximum(np.round(np.concatenate(caps) * FLOW_SCALE), 0).astype(np.int32)
    cap = csr_matrix((caps, (rows, cols)), shape=(n + 2, n + 2))
    cap.sum_duplicates()
    flow = maximum_flow(cap, S, T, method="dinic").flow
    residual = (cap - flow).tocsr()
    residual.data = np.where(residual.data > 0, 1, 0).astype(np.int32)
    residual.eliminate_zeros()
    reach = breadth_first_order(residual, S, directed=True, return_predecessors=False)
    inside = np.zeros(n, bool)
    inside[reach[reach < n]] = True
    return inside


def _rectilinear_cleanup(coords: np.ndarray) -> np.ndarray:
    """Drop repeated and collinear vertices, then remove short jogs between parallel edges."""
    def simplify(p):
        p = [tuple(v) for v in p]
        changed = True
        while changed and len(p) > 4:
            changed = False
            out = []
            n = len(p)
            for k in range(n):
                a, b, c = np.array(p[k - 1]), np.array(p[k]), np.array(p[(k + 1) % n])
                if np.allclose(a, b, atol=1e-9):
                    changed = True
                    continue
                cross = (b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0]
                if abs(cross) < 1e-12:
                    changed = True
                    continue
                out.append(tuple(b))
            p = out
        return np.array(p)

    p = simplify(coords)
    while len(p) > 4:
        n = len(p)
        lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        k = int(np.argmin(lengths))
        if lengths[k] >= JOG_M:
            break
        # edge k runs p[k] -> p[k+1]; its neighbours (k-1) and (k+1) are parallel: move the shorter onto the longer
        a, b = p[k], p[(k + 1) % n]
        prev_len, next_len = lengths[k - 1], lengths[(k + 1) % n]
        axis = 0 if abs(a[0] - b[0]) > abs(a[1] - b[1]) else 1  # edge k runs along this axis
        if prev_len >= next_len:
            p[(k + 1) % n][axis] = a[axis]
            p[(k + 2) % n][axis] = a[axis]
        else:
            p[k][axis] = b[axis]
            p[k - 1][axis] = b[axis]
        p = simplify(p)
    return p


# ---------------------------------------------------------------------------------------------
def build_layout(points: PointSet, floor: HorizontalPlane, camera_positions: np.ndarray) -> Layout:
    maps = plan_maps(points, floor, camera_positions)
    grid = maps.grid
    free_raw = ndimage.binary_closing(maps.floor | maps.path, structure=_disk(CLOSE_RADIUS_M / grid.cell))

    h_all = points.xyz[:, 1] - floor.height
    vertical = (np.abs(points.normal[:, 1]) < 0.2) & (h_all > 0.05) & (h_all < MAX_WALL_HEIGHT_M)
    nxz = points.normal[vertical][:, [0, 2]].astype(float)
    nxz /= np.linalg.norm(nxz, axis=1, keepdims=True) + 1e-9
    yaw, _ = dominant_yaw(nxz)
    R = _to_aligned(yaw)
    nuv = nxz @ R.T
    share = float(np.mean((np.abs(nuv) > FACE_COS).any(axis=1)))
    if share < MANHATTAN_MIN_SHARE:
        raise NotManhattan(f"only {share:.0%} of wall normals follow the two main directions")
    uv, h, nuv = _thin(points.xyz[vertical][:, [0, 2]].astype(float) @ R.T, h_all[vertical].astype(float), nuv)
    free = _FreeLookup(free_raw, grid, R)
    floor_seen = _FreeLookup(maps.floor, grid, R)
    path_uv = np.asarray(camera_positions, float)[:, [0, 2]] @ R.T
    aligned = (np.abs(nuv) > FACE_COS).any(axis=1) & (h >= PASSAGE_BAND_M[0]) & (h < DOOR_HEAD_M)
    evidence_m = float(np.clip(np.percentile(h[aligned], WALL_EVIDENCE_PERCENTILE), *WALL_EVIDENCE_RANGE_M))

    # extent: free space (in aligned coordinates) plus a margin
    rows, cols = np.nonzero(free_raw)
    free_uv = grid.centers(rows, cols) @ R.T
    lo = free_uv.min(axis=0) - BOX_MARGIN_M
    hi = free_uv.max(axis=0) + BOX_MARGIN_M
    n_bins_v = int(np.ceil((hi[1] - lo[1]) / ALONG_BIN_M)) + 1
    n_bins_u = int(np.ceil((hi[0] - lo[0]) / ALONG_BIN_M)) + 1
    lines_u = [l for l in _face_lines(uv, h, nuv, 0, lo[1], n_bins_v, evidence_m, floor_seen, path_uv)
               if lo[0] < l.c < hi[0]]
    lines_v = [l for l in _face_lines(uv, h, nuv, 1, lo[0], n_bins_u, evidence_m, floor_seen, path_uv)
               if lo[1] < l.c < hi[1]]
    _close_ends(lines_u, lines_v)
    _close_ends(lines_v, lines_u)
    paired = _pair_wall_faces(lines_u) + _pair_wall_faces(lines_v)
    us = np.array([lo[0]] + [l.c for l in lines_u] + [hi[0]])
    vs = np.array([lo[1]] + [l.c for l in lines_v] + [hi[1]])
    nu, nv = len(us) - 1, len(vs) - 1
    cell_id = np.arange(nu * nv).reshape(nu, nv)

    # free share per cell from an aligned raster (integral image)
    res = grid.cell
    au = lo[0] + (np.arange(int(np.ceil((hi[0] - lo[0]) / res))) + 0.5) * res
    av = lo[1] + (np.arange(int(np.ceil((hi[1] - lo[1]) / res))) + 0.5) * res
    AU, AV = np.meshgrid(au, av, indexing="ij")
    aligned_free = free(np.column_stack([AU.ravel(), AV.ravel()])).reshape(AU.shape).astype(np.float64)
    integral = np.pad(aligned_free.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    iu = np.clip(np.round((us - lo[0]) / res).astype(int), 0, len(au))
    iv = np.clip(np.round((vs - lo[1]) / res).astype(int), 0, len(av))
    free_share = np.zeros((nu, nv))
    for i in range(nu):
        a0, a1 = iu[i], max(iu[i + 1], iu[i] + 1)
        for j in range(nv):
            b0, b1 = iv[j], max(iv[j + 1], iv[j] + 1)
            a1c, b1c = min(a1, len(au)), min(b1, len(av))
            a0c, b0c = min(a0, a1c - 1), min(b0, b1c - 1)
            total = integral[a1c, b1c] - integral[a0c, b1c] - integral[a1c, b0c] + integral[a0c, b0c]
            free_share[i, j] = total / max((a1c - a0c) * (b1c - b0c), 1)
    area = np.outer(np.diff(us), np.diff(vs))

    # edges between neighbouring cells, with wall and door shares from the face line they lie on
    edges, lengths, wall_share, door_share = [], [], [], []
    for i in range(1, nu):             # vertical edges on u = us[i]
        line = lines_u[i - 1]
        for j in range(nv):
            w, d = line.state(vs[j], vs[j + 1], lo[1])
            edges.append((cell_id[i - 1, j], cell_id[i, j])); lengths.append(vs[j + 1] - vs[j])
            wall_share.append(w); door_share.append(d)
    for j in range(1, nv):             # horizontal edges on v = vs[j]
        line = lines_v[j - 1]
        for i in range(nu):
            w, d = line.state(us[i], us[i + 1], lo[0])
            edges.append((cell_id[i, j - 1], cell_id[i, j])); lengths.append(us[i + 1] - us[i])
            wall_share.append(w); door_share.append(d)
    edges = np.array(edges, int).reshape(-1, 2)
    lengths, wall_share, door_share = map(np.asarray, (lengths, wall_share, door_share))
    boundary = wall_share + door_share

    # inside / outside by minimum cut
    f = free_share.ravel()
    A = area.ravel()
    t_source = A * f * W_FREE                         # cost of calling the cell outside
    t_sink = A * (1 - f) * W_UNSEEN                   # cost of calling it inside
    rim = np.zeros((nu, nv))                          # cells on the box edge touch the unknown outside
    rim[0, :] += np.diff(vs); rim[-1, :] += np.diff(vs); rim[:, 0] += np.diff(us); rim[:, -1] += np.diff(us)
    t_sink = t_sink + rim.ravel() * W_OPEN
    weights = lengths * (W_OPEN * (1 - np.clip(boundary, 0, 1)) + W_WALL * np.clip(boundary, 0, 1))
    inside = _min_cut(nu * nv, t_source, t_sink, edges, weights)

    # rooms: inside cells joined across edges that are neither wall nor door
    passable = (boundary < 0.5) & inside[edges[:, 0]] & inside[edges[:, 1]]
    graph = csr_matrix((np.ones(passable.sum()), (edges[passable, 0], edges[passable, 1])), shape=(nu * nv,) * 2)
    _, comp = connected_components(graph, directed=False)
    comp = np.where(inside, comp, -1)
    comp, split_doors = _split_by_narrowing(comp, maps, grid, R, us, vs, AU, AV, res, edges[passable])
    room_cells = {}
    for c in np.unique(comp[comp >= 0]):
        cells = np.nonzero(comp == c)[0]
        i, j = cells // nv, cells % nv
        longest = max(us[i.max() + 1] - us[i.min()], vs[j.max() + 1] - vs[j.min()])
        if A[cells].sum() >= MIN_ROOM_M2 and A[cells].sum() / longest >= MIN_ROOM_WIDTH_M:
            room_cells[c] = cells
    order = sorted(room_cells, key=lambda c: -A[room_cells[c]].sum())
    room_of_cell = np.zeros(nu * nv, int)
    for rid, c in enumerate(order, start=1):
        room_of_cell[room_cells[c]] = rid

    # outlines
    wall_uv, wall_h, wall_n = uv, h, nuv
    outlines: dict[int, RoomOutline] = {}
    polys_uv: dict[int, np.ndarray] = {}
    for rid in range(1, len(order) + 1):
        cells = np.nonzero(room_of_cell == rid)[0]
        shape = unary_union([box(us[k // nv], vs[k % nv], us[k // nv + 1], vs[k % nv + 1]) for k in cells])
        if shape.geom_type != "Polygon":
            shape = max(shape.geoms, key=lambda g: g.area)
        shape = orient(Polygon(shape.exterior), sign=1.0)
        p = _rectilinear_cleanup(np.array(shape.exterior.coords)[:-1])
        if len(p) < 4:
            continue
        outline = _refit(rid, p, wall_uv, wall_h, wall_n, R, yaw)
        if outline is not None:
            outlines[rid] = outline
            polys_uv[rid] = np.array(outline.vertices) @ R.T

    # room labels on the plan grid (for per-room ceilings)
    labels = np.zeros((grid.rows, grid.cols), np.int32)
    areas = {}
    for rid, outline in outlines.items():
        px = np.column_stack([(outline.vertices[:, 0] - grid.x0) / grid.cell,
                              (outline.vertices[:, 1] - grid.z0) / grid.cell])
        cv2.fillPoly(labels, [np.round(px).astype(np.int32)], int(rid))
        areas[rid] = outline.area_m2
    room_map = LayoutRoomMap(grid, labels, areas)

    openings = _openings(lines_u, lines_v, outlines, polys_uv, us, vs, room_of_cell, nv, lo)
    comp_to_room = {c: rid for rid, c in enumerate(order, start=1)}
    openings += _split_openings(split_doors, comp_to_room, outlines, len(openings))
    openings += _walked_openings(np.asarray(camera_positions, float)[:, [0, 2]], room_map, outlines, openings)
    info = {"method": "walls_first", "yaw_deg": round(yaw, 2), "manhattan_share": round(share, 3),
            "wall_evidence_m": round(evidence_m, 2),
            "face_lines": len(lines_u) + len(lines_v), "cells": nu * nv,
            "inside_cells": int(inside.sum()), "rooms": len(outlines),
            "doors_found": sum(len(l.doors) for l in lines_u + lines_v), "paired_wall_faces": paired}
    return Layout(room_map, outlines, openings, yaw, info)


def _split_by_narrowing(comp, maps, grid, R, us, vs, AU, AV, res, passable_edges):
    """Split walls-first rooms along the floor-first rooms they contain. Returns the new component id per
    cell and the doorways between split parts: (centre xz, width, component a, component b)."""
    ff = segment_rooms(maps)
    nu, nv = len(us) - 1, len(vs) - 1
    if ff.labels.max() == 0:
        return comp, []
    pix_uv = np.column_stack([AU.ravel(), AV.ravel()])
    rows, cols = grid.index(pix_uv @ R)
    pix_label = ff.labels[rows, cols]
    pix_cell = (np.clip(np.searchsorted(us, pix_uv[:, 0]) - 1, 0, nu - 1) * nv
                + np.clip(np.searchsorted(vs, pix_uv[:, 1]) - 1, 0, nv - 1))
    n_ff = int(ff.labels.max()) + 1
    counts = np.bincount(pix_cell * n_ff + pix_label, minlength=nu * nv * n_ff).reshape(nu * nv, n_ff)
    counts[:, 0] = 0
    neighbours: dict[int, list[int]] = {}
    for a, b in passable_edges:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))

    comp = comp.copy()
    next_id = int(comp.max()) + 1
    split_doors = []
    for c in np.unique(comp[comp >= 0]):
        cells = np.nonzero(comp == c)[0]
        overlap = counts[cells].sum(axis=0) * res * res
        significant = [int(l) for l in np.nonzero(overlap >= SPLIT_MIN_M2)[0]]
        if len(significant) < 2:
            continue
        assign = {}
        for cell in cells:
            best = max(significant, key=lambda l: counts[cell, l])
            if counts[cell, best] > 0:
                assign[int(cell)] = best
        frontier = list(assign)
        members = set(int(x) for x in cells)
        while frontier:                      # unlabelled cells join the part they touch first
            nxt = []
            for cell in frontier:
                for nb in neighbours.get(cell, []):
                    if nb in members and nb not in assign:
                        assign[nb] = assign[cell]
                        nxt.append(nb)
            frontier = nxt
        new_ids = {l: next_id + i for i, l in enumerate(significant)}
        next_id += len(significant)
        for cell, l in assign.items():
            comp[cell] = new_ids[l]
        for d in ff.doorways:
            a, b = d.rooms
            if a in new_ids and b in new_ids:
                split_doors.append((np.array(d.center_xz), d.width_m, new_ids[a], new_ids[b]))
    return comp, split_doors


def _split_openings(split_doors, comp_to_room, outlines, first_index) -> list[OpeningOnWall]:
    openings = []
    for k, (centre, width, ca, cb) in enumerate(split_doors):
        ra, rb = comp_to_room.get(ca), comp_to_room.get(cb)
        if ra not in outlines or rb not in outlines:
            continue
        for rid, other in ((ra, rb), (rb, ra)):
            best = None
            for j, wall in enumerate(outlines[rid].walls):
                seg = wall.end - wall.start
                t = float(np.clip(np.dot(centre - wall.start, seg) / max(np.dot(seg, seg), 1e-9), 0, 1))
                dist = float(np.linalg.norm(wall.start + t * seg - centre))
                if best is None or dist < best[0]:
                    best = (dist, j, t * wall.length_m)
            if best and best[0] <= 0.6:
                openings.append(OpeningOnWall(rid, best[1], best[2], float(np.clip(width, DOOR_MIN_M, DOOR_MAX_M)),
                                              other, first_index + k, measured=False))
    return openings


def _nearest_wall(outline: RoomOutline, point: np.ndarray, max_m: float = 0.6):
    best = None
    for j, wall in enumerate(outline.walls):
        seg = wall.end - wall.start
        t = float(np.clip(np.dot(point - wall.start, seg) / max(np.dot(seg, seg), 1e-9), 0, 1))
        dist = float(np.linalg.norm(wall.start + t * seg - point))
        if best is None or dist < best[0]:
            best = (dist, j, t * wall.length_m)
    return best if best and best[0] <= max_m else None


def _walked_openings(path_xz, room_map, outlines, existing) -> list[OpeningOnWall]:
    """The walk went from room A to room B, so there is a way between them: add a door of nominal width
    where it crossed, unless the two rooms are already linked. A walk-through shows every room it enters."""
    linked = {tuple(sorted((o.room_id, o.other_room))) for o in existing if o.other_room is not None}
    rooms = room_map.room_of(path_xz)
    openings, last_room, last_index = [], 0, 0
    next_index = max((o.doorway_index for o in existing), default=-1) + 1
    for i, r in enumerate(rooms):
        if r == 0:
            continue
        if last_room and r != last_room:
            pair = tuple(sorted((int(last_room), int(r))))
            if pair not in linked and last_room in outlines and r in outlines:
                crossing = (path_xz[last_index] + path_xz[i]) / 2
                added = []
                for rid, other in ((int(last_room), int(r)), (int(r), int(last_room))):
                    best = _nearest_wall(outlines[rid], crossing)
                    if best:
                        added.append(OpeningOnWall(rid, best[1], best[2], NOMINAL_DOOR_M, other, next_index,
                                                   measured=False))
                if len(added) == 2:
                    openings += added
                    linked.add(pair)
                    next_index += 1
        last_room, last_index = r, i
    return openings


def _refit(rid, p, wall_uv, wall_h, wall_n, R, yaw) -> RoomOutline | None:
    """Move every edge of a rectilinear outline onto the wall points facing into the room."""
    n = len(p)
    coords, spreads, supports = [], [], []
    for k in range(n):
        a, b = p[k], p[(k + 1) % n]
        axis = 0 if abs(a[0] - b[0]) < 1e-9 else 1       # 0: vertical edge u = const
        c = a[axis]
        d = b - a
        inward = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-12)  # left of a CCW edge
        lo_a, hi_a = sorted((a[1 - axis], b[1 - axis]))
        sel = ((np.abs(wall_uv[:, axis] - c) < REFIT_BAND_M)
               & (wall_uv[:, 1 - axis] > lo_a + 0.05) & (wall_uv[:, 1 - axis] < hi_a - 0.05)
               & (wall_n @ inward > FACING_DOT) & (wall_h > 0.2))
        if sel.sum() >= REFIT_MIN_POINTS:
            faces = wall_uv[sel, axis]
            med = float(np.median(faces))
            coords.append(med)
            spreads.append(float(1.4826 * np.median(np.abs(faces - med))))
            supports.append(int(sel.sum()))
        else:
            coords.append(float(c))
            spreads.append(float("nan"))
            supports.append(0)
    axes = [0 if abs(p[k][0] - p[(k + 1) % n][0]) < 1e-9 else 1 for k in range(n)]
    if any(axes[k] == axes[(k + 1) % n] for k in range(n)):
        return None
    corners = np.zeros((n, 2))
    for k in range(n):  # vertex k joins edge k-1 and edge k
        prev, cur = k - 1, k
        u = coords[prev] if axes[prev] == 0 else coords[cur]
        v = coords[prev] if axes[prev] == 1 else coords[cur]
        corners[k] = (u, v)
    world = corners @ R
    walls = []
    for k in range(n):
        s, e = world[k], world[(k + 1) % n]
        walls.append(WallSegment(s, e, float(np.linalg.norm(e - s)), spreads[k], supports[k]))
    outline = RoomOutline(rid, yaw, world, walls)
    return outline if outline.area_m2 > 0.5 else None


def _openings(lines_u, lines_v, outlines, polys_uv, us, vs, room_of_cell, nv, lo) -> list[OpeningOnWall]:
    def room_at(u, v):
        i = np.searchsorted(us, u) - 1
        j = np.searchsorted(vs, v) - 1
        if 0 <= i < len(us) - 1 and 0 <= j < len(vs) - 1:
            return int(room_of_cell[i * nv + j])
        return 0

    detections = []
    for line in lines_u + lines_v:
        for d_lo, d_hi, measured in line.doors:
            detections.append((line.axis, line.c, d_lo, d_hi, measured))
    # group the two faces of one wall into one physical door
    groups: list[list[tuple]] = []
    for det in detections:
        for g in groups:
            ax, c, lo_, hi_ = g[0][:4]
            overlap = min(hi_, det[3]) - max(lo_, det[2])
            if det[0] == ax and abs(det[1] - c) <= DOOR_PAIR_M and overlap > 0.5 * (det[3] - det[2]):
                g.append(det)
                break
        else:
            groups.append([det])

    openings = []
    for index, group in enumerate(groups):
        axis = group[0][0]
        measured_widths = [d[3] - d[2] for d in group if d[4]]
        width = float(np.mean(measured_widths)) if measured_widths else NOMINAL_DOOR_M
        mid = float(np.mean([(d[2] + d[3]) / 2 for d in group]))
        cs = [d[1] for d in group]
        rooms = set()
        for sign, edge_c in ((-1.0, min(cs)), (1.0, max(cs))):
            for dist in (0.03, 0.10, 0.20, 0.35, 0.50):
                c = edge_c + sign * dist
                r = room_at(c, mid) if axis == 0 else room_at(mid, c)
                if r:
                    rooms.add(r)
                    break
        rooms = sorted(r for r in rooms if r in outlines)
        for rid in rooms:
            poly = polys_uv[rid]
            n = len(poly)
            best = None
            for k in range(n):
                a, b = poly[k], poly[(k + 1) % n]
                if axis == 0 and abs(a[0] - b[0]) < 1e-6:
                    dist, lo_a, hi_a, along_a = min(abs(a[0] - c) for c in cs), min(a[1], b[1]), max(a[1], b[1]), a[1]
                elif axis == 1 and abs(a[1] - b[1]) < 1e-6:
                    dist, lo_a, hi_a, along_a = min(abs(a[1] - c) for c in cs), min(a[0], b[0]), max(a[0], b[0]), a[0]
                else:
                    continue
                if lo_a - 0.05 <= mid <= hi_a + 0.05 and dist <= 0.15 and (best is None or dist < best[0]):
                    best = (dist, k, abs(mid - along_a))
            if best is None:
                continue
            others = [r for r in rooms if r != rid]
            openings.append(OpeningOnWall(rid, best[1], best[2], width, others[0] if others else None, index,
                                          measured=bool(measured_widths)))
    return openings
