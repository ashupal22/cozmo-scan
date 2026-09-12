"""Room outlines: dominant wall direction, right-angled polygons snapped to measured wall faces.

For each room from rooms.segment_rooms:
1. Rotate the plan so the dominant wall direction (wall normals, modulo 90 degrees) lines up with the axes.
2. Trace the room's cells and split the outline into runs that go along one axis. Runs shorter than
   MIN_WALL_M are merged away, which removes doorway notches and furniture bumps.
3. Snap each run to the wall face: the median position of wall points that face into the room,
   within SNAP_WINDOW_M of the run. The spread of those points is kept for the error model.
4. Corners are where neighbouring runs meet, and wall lengths are corner to corner (inside faces).

Assumes walls meet at right angles (a Manhattan layout). Non-right-angle walls come later.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter1d

from cozmo.geometry.fusion import PointSet
from cozmo.geometry.planes import HorizontalPlane
from cozmo.geometry.rooms import (WALL_MAX_ABOVE_FLOOR_M, WALL_MIN_ABOVE_FLOOR_M, WALL_NORMAL_MAX_DOT,
                                  RoomMap)

MIN_WALL_M = 0.25
SNAP_WINDOW_M = 0.30
FACING_DOT = 0.7
MIN_SNAP_POINTS = 50


@dataclass
class WallSegment:
    start: np.ndarray  # world (x, z)
    end: np.ndarray
    length_m: float
    face_spread_m: float  # robust spread of wall points around the fitted face; nan if not snapped
    support: int          # wall points used for snapping (0: outline from floor cells only)


@dataclass
class RoomOutline:
    room_id: int
    yaw_deg: float
    vertices: np.ndarray  # (N, 2) world (x, z), counter-clockwise when viewed with +y up
    walls: list[WallSegment]  # walls[k] runs from vertices[k] to vertices[k + 1]

    @property
    def area_m2(self) -> float:
        x, z = self.vertices[:, 0], self.vertices[:, 1]
        return float(abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))) / 2)

    @property
    def perimeter_m(self) -> float:
        return float(sum(w.length_m for w in self.walls))


def wall_mask(points: PointSet, floor: HorizontalPlane) -> np.ndarray:
    above = points.xyz[:, 1] - floor.height
    return ((np.abs(points.normal[:, 1]) < WALL_NORMAL_MAX_DOT)
            & (above > WALL_MIN_ABOVE_FLOOR_M) & (above < WALL_MAX_ABOVE_FLOOR_M))


def dominant_yaw(normals_xz: np.ndarray) -> tuple[float, float]:
    """Dominant horizontal normal direction in degrees (mod 90), and the share of normals within 3 degrees of it."""
    theta = np.degrees(np.arctan2(normals_xz[:, 1], normals_xz[:, 0])) % 90
    hist, _ = np.histogram(theta, bins=360, range=(0, 90))
    k = int(np.argmax(uniform_filter1d(hist.astype(float), 5, mode="wrap")))
    center = np.radians((k + 0.5) * 0.25 * 4)
    diff = np.angle(np.exp(1j * (np.radians(theta * 4) - center)))
    near = np.abs(diff) < np.radians(12)
    yaw = float((np.degrees(center + np.angle(np.exp(1j * diff[near]).mean())) / 4) % 90)
    return yaw, float(near.mean())


def dominant_yaw_deg(normals_xz: np.ndarray) -> float:
    """Dominant horizontal normal direction in degrees, modulo 90."""
    return dominant_yaw(normals_xz)[0]


def _to_aligned(yaw_deg: float) -> np.ndarray:
    """2x2 rotation taking world (x, z) to aligned (u, v)."""
    t = np.radians(yaw_deg)
    return np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])


def _runs(contour: np.ndarray) -> list[tuple[bool, np.ndarray]]:
    """Split a closed outline (K, 2) into runs along u (True) or along v (False)."""
    k = len(contour)
    step = min(4, max(1, k // 8))
    d = contour[(np.arange(k) + step) % k] - contour
    along_u = np.abs(d[:, 0]) >= np.abs(d[:, 1])
    window = 7
    padded = np.r_[along_u[-window:], along_u, along_u[:window]].astype(float)
    along_u = np.convolve(padded, np.ones(window) / window, "same")[window:-window] > 0.5
    changes = np.nonzero(along_u != np.roll(along_u, 1))[0]
    if len(changes) == 0:
        return []
    order = np.roll(np.arange(k), -changes[0])
    along_u, points = along_u[order], contour[order]
    runs, start = [], 0
    for i in range(1, k + 1):
        if i == k or along_u[i] != along_u[start]:
            runs.append((bool(along_u[start]), points[start:i]))
            start = i
    return runs


def _extent(run: tuple[bool, np.ndarray]) -> float:
    axis = 0 if run[0] else 1
    return float(np.ptp(run[1][:, axis]))


def _simplify(runs: list[tuple[bool, np.ndarray]]) -> list[tuple[bool, np.ndarray]]:
    """Merge short runs into a neighbour, then join neighbouring runs along the same axis."""
    runs = list(runs)
    while True:
        joined = []
        for run in runs:
            if joined and joined[-1][0] == run[0]:
                joined[-1] = (run[0], np.vstack([joined[-1][1], run[1]]))
            else:
                joined.append(run)
        if len(joined) > 1 and joined[0][0] == joined[-1][0]:
            joined[0] = (joined[0][0], np.vstack([joined[-1][1], joined[0][1]]))
            joined.pop()
        runs = joined
        if len(runs) <= 4:
            return runs
        short = min(range(len(runs)), key=lambda i: _extent(runs[i]))
        if _extent(runs[short]) >= MIN_WALL_M:
            return runs
        prev = (short - 1) % len(runs)
        runs[prev] = (runs[prev][0], np.vstack([runs[prev][1], runs[short][1]]))
        runs.pop(short)


def outline_room(room_map: RoomMap, room_id: int, points: PointSet, floor: HorizontalPlane,
                 yaw_deg: float | None = None) -> RoomOutline | None:
    grid = room_map.grid
    walls = points.subset(wall_mask(points, floor))
    if yaw_deg is None:
        yaw_deg = dominant_yaw_deg(walls.normal[:, [0, 2]])
    R = _to_aligned(yaw_deg)

    rows, cols = np.nonzero(room_map.labels == room_id)
    if len(rows) < 20:
        return None
    uv_cells = grid.centers(rows, cols) @ R.T
    cell = grid.cell
    lo = uv_cells.min(axis=0) - 5 * cell
    idx = np.floor((uv_cells - lo) / cell).astype(int)
    mask = np.zeros((idx[:, 1].max() + 6, idx[:, 0].max() + 6), np.uint8)
    mask[idx[:, 1], idx[:, 0]] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)[:, 0, :].astype(float)
    contour_uv = (contour + 0.5) * cell + lo  # (col, row) -> (u, v)
    runs = _simplify(_runs(contour_uv))
    if len(runs) < 4 or len(runs) % 2:
        return None

    wall_uv = walls.xyz[:, [0, 2]] @ R.T
    wall_n = walls.normal[:, [0, 2]] @ R.T

    def inside(u: float, v: float) -> bool:
        c, r = int((u - lo[0]) / cell), int((v - lo[1]) / cell)
        return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])

    lines = []  # (along_u, coordinate, spread, support)
    for along_u, pts in runs:
        axis_along, axis_across = (0, 1) if along_u else (1, 0)
        coord = float(np.median(pts[:, axis_across]))
        lo_span, hi_span = pts[:, axis_along].min(), pts[:, axis_along].max()
        mid = (lo_span + hi_span) / 2
        probe = [mid, coord + 0.15] if along_u else [coord + 0.15, mid]
        into_room = 1.0 if inside(*probe) else -1.0
        near = ((np.abs(wall_uv[:, axis_across] - coord) < SNAP_WINDOW_M)
                & (wall_uv[:, axis_along] > lo_span + 0.1) & (wall_uv[:, axis_along] < hi_span - 0.1)
                & (wall_n[:, axis_across] * into_room > FACING_DOT))
        if near.sum() >= MIN_SNAP_POINTS:
            faces = wall_uv[near, axis_across]
            med = float(np.median(faces))
            lines.append((along_u, med, float(1.4826 * np.median(np.abs(faces - med))), int(near.sum())))
        else:
            lines.append((along_u, coord, float("nan"), 0))

    n = len(lines)
    corners_uv = []
    for k in range(n):
        a, b = lines[k], lines[(k + 1) % n]
        u = b[1] if a[0] else a[1]
        v = a[1] if a[0] else b[1]
        corners_uv.append((u, v))
    corners = np.array(corners_uv) @ R  # back to world (x, z)
    segments = [lines[(k + 1) % n] for k in range(n)]  # segment k: corner k -> corner k+1

    signed = np.dot(corners[:, 0], np.roll(corners[:, 1], -1)) - np.dot(corners[:, 1], np.roll(corners[:, 0], -1))
    if signed < 0:
        corners = corners[::-1]
        # reversed segment k runs from old corner n-1-k to old corner n-2-k; those two corners share line n-1-k
        segments = [lines[(n - 1 - k) % n] for k in range(n)]

    wall_list = []
    for k in range(n):
        start, end = corners[k], corners[(k + 1) % n]
        _, _, spread, support = segments[k]
        wall_list.append(WallSegment(start, end, float(np.linalg.norm(end - start)), spread, support))
    return RoomOutline(room_id, yaw_deg, corners, wall_list)


@dataclass
class OpeningOnWall:
    room_id: int
    wall_index: int
    offset_m: float    # from the wall start to the opening centre
    width_m: float
    other_room: int | None
    doorway_index: int
    measured: bool = True  # False: the jambs were not seen, the width is nominal


def attach_doorways(outlines: dict[int, RoomOutline], room_map: RoomMap,
                    max_distance_m: float = 0.5) -> list[OpeningOnWall]:
    """Place each doorway on the nearest wall of both rooms it connects."""
    openings = []
    for index, door in enumerate(room_map.doorways):
        center = np.array(door.center_xz)
        for rid, other in (door.rooms, door.rooms[::-1]):
            outline = outlines.get(rid)
            if outline is None:
                continue
            best = None
            for k, wall in enumerate(outline.walls):
                seg = wall.end - wall.start
                t = float(np.clip(np.dot(center - wall.start, seg) / max(np.dot(seg, seg), 1e-9), 0, 1))
                distance = float(np.linalg.norm(wall.start + t * seg - center))
                if best is None or distance < best[0]:
                    best = (distance, k, t * wall.length_m)
            if best and best[0] <= max_distance_m:
                openings.append(OpeningOnWall(rid, best[1], best[2], door.width_m, other, index))
    return openings


def outline_rooms(room_map: RoomMap, points: PointSet, floor: HorizontalPlane) -> dict[int, RoomOutline]:
    walls = points.subset(wall_mask(points, floor))
    yaw = dominant_yaw_deg(walls.normal[:, [0, 2]])
    outlines = {rid: outline_room(room_map, rid, points, floor, yaw) for rid in room_map.room_ids}
    return {rid: o for rid, o in outlines.items() if o is not None}
