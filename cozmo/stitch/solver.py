"""Stitch separately measured rooms into one plan by matching their doors (photo tier: one folder per room).

Each room comes in its own frame: an outline, and the doors seen on its walls. A door's centre sits on the
inside face of the wall, its normal points out of the room, and its width is known. Two rooms that share a door
see it from the two faces of one wall, so their centres lie a wall's thickness apart and their normals point
opposite ways. One door pairing fixes where a room sits relative to the other, up to measurement noise.

The solver builds the plan room by room (beam search). From a partial plan it takes the next door not yet
explained and tries every compatible door of every room not yet placed, and also "this door leads elsewhere"
(outside, or a room without photos). Each candidate plan is scored:
  + a bonus per door pair, less a penalty for width disagreement
  + further door pairs that line up by themselves (confirmations)
  - overlap with rooms already placed, beyond what outline noise explains (large overlaps are rejected)
  - doors that open into a placed room where that room has no door
  - leaving a door unexplained (small), and starting a separate island when no door fits (large)
Turns are snapped to quarter turns of the neighbouring room when within SNAP_DEG, since the walls of one home
meet at right angles almost everywhere. Scores for shared walls and for compactness were tried and made door choices
worse (bench/README.md), so geometry only rules placements out, through overlaps and doors that open into rooms.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

WALL_THICKNESS_M = 0.15
MAX_WIDTH_GAP_M = 0.25
WIDTH_SIGMA_M = 0.08
SNAP_DEG = 8.0
MATCH_BONUS = 5.0
CONFIRM_BONUS = 3.0
CONFIRM_REACH_M = 0.35
OVERLAP_SHRINK_M = 0.05
OVERLAP_PER_M2 = 4.0
MAX_OVERLAP_SHARE = 0.15
BEYOND_DOOR_M = 0.4
BLOCKED_DOOR_PENALTY = 3.0
LEAVE_OPEN_PENALTY = 1.0
ISLAND_PENALTY = 12.0
ISLAND_GAP_M = 1.0
BEAM_WIDTH = 128               # bench/results/stitch_ablation: 8 -> 128 fixes early wrong choices; 512 adds nothing


def rot(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass(frozen=True)
class Door:
    centre: tuple[float, float]     # on the inside face of the wall, room frame
    normal: tuple[float, float]     # unit, pointing out of the room
    width_m: float


@dataclass
class Room:
    name: str
    polygon: np.ndarray             # (N, 2) outline, room frame
    doors: list[Door] = field(default_factory=list)


@dataclass(frozen=True)
class Placement:
    """Room frame to plan: rotate by `angle`, then shift."""
    angle: float
    shift: tuple[float, float]

    def apply(self, xy) -> np.ndarray:
        return np.asarray(xy, float) @ rot(self.angle).T + np.asarray(self.shift, float)

    def turn(self, v) -> np.ndarray:
        return rot(self.angle) @ np.asarray(v, float)


@dataclass(frozen=True)
class DoorPair:
    room_a: int
    door_a: int
    room_b: int
    door_b: int


@dataclass
class StitchResult:
    placements: dict[int, Placement]    # room index -> placement
    pairs: list[DoorPair]
    islands: int                        # 1: one connected plan
    overlap_m2: float
    score: float

    def adjacency(self) -> set[frozenset]:
        return {frozenset((p.room_a, p.room_b)) for p in self.pairs}


@dataclass
class _State:
    placements: dict
    polygons: dict
    union: object
    pairs: list
    explained: frozenset                # (room, door): paired, or declared to lead elsewhere
    islands: int
    score: float

    def key(self):
        return (frozenset(self.placements), frozenset(dataclasses.astuple(p) for p in self.pairs), self.explained)


def _plan_polygon(room: Room, placement: Placement) -> Polygon:
    """The room's outline in the plan. An outline that touches itself is repaired, and only its largest part
    kept (or its convex hull if nothing usable is left)."""
    raw = Polygon(placement.apply(room.polygon))
    if raw.is_valid:
        return raw
    repaired = raw.buffer(0)
    parts = [repaired] if repaired.geom_type == "Polygon" else \
        [g for g in getattr(repaired, "geoms", []) if g.geom_type == "Polygon"]
    parts = [g for g in parts if not g.is_empty]
    return max(parts, key=lambda g: g.area) if parts else raw.convex_hull


def _door_in_plan(rooms: list[Room], placements: dict, r: int, i: int):
    door, p = rooms[r].doors[i], placements[r]
    return p.apply(door.centre), p.turn(door.normal), door.width_m


def _place(parent: Placement, door: Door, other: Door, centre: np.ndarray, normal: np.ndarray) -> Placement:
    """Placement of a room whose door `other` is the far side of `door` (centre, normal already in the plan)."""
    angle = np.arctan2(-normal[1], -normal[0]) - np.arctan2(other.normal[1], other.normal[0])
    quarter = np.pi / 2
    snapped = parent.angle + quarter * np.round((angle - parent.angle) / quarter)
    if abs(_wrap(angle - snapped)) <= np.radians(SNAP_DEG):
        angle = snapped
    shift = centre + normal * WALL_THICKNESS_M - rot(angle) @ np.asarray(other.centre, float)
    return Placement(_wrap(angle), (float(shift[0]), float(shift[1])))


def _add_room(rooms: list[Room], state: _State, b: int, placement: Placement, pair: DoorPair,
              width_gap: float) -> _State | None:
    poly = _plan_polygon(rooms[b], placement)
    inner = poly.buffer(-OVERLAP_SHRINK_M)
    overlap = inner.intersection(state.union).area if not inner.is_empty else 0.0
    if overlap > MAX_OVERLAP_SHARE * poly.area:
        return None
    union = state.union.union(poly)
    score = state.score + MATCH_BONUS - 0.5 * (width_gap / WIDTH_SIGMA_M) ** 2 - OVERLAP_PER_M2 * overlap
    placements = {**state.placements, b: placement}
    pairs = state.pairs + [pair]
    explained = set(state.explained) | {(pair.room_a, pair.door_a), (pair.room_b, pair.door_b)}
    for j in range(len(rooms[b].doors)):
        if (b, j) in explained:
            continue
        ce, ne, we = _door_in_plan(rooms, placements, b, j)
        for r in state.placements:
            partner = next((i for i in range(len(rooms[r].doors)) if (r, i) not in explained
                            and _meets(*_door_in_plan(rooms, placements, r, i), ce, ne, we)), None)
            if partner is not None:
                pairs.append(DoorPair(r, partner, b, j))
                explained |= {(r, partner), (b, j)}
                score += CONFIRM_BONUS
                break
    for j in range(len(rooms[b].doors)):
        if (b, j) not in explained:
            ce, ne, _ = _door_in_plan(rooms, placements, b, j)
            if state.union.contains(Point(*(ce + ne * BEYOND_DOOR_M))):
                score -= BLOCKED_DOOR_PENALTY
    for r in state.placements:
        for i in range(len(rooms[r].doors)):
            if (r, i) not in explained:
                cd, nd, _ = _door_in_plan(rooms, placements, r, i)
                if poly.contains(Point(*(cd + nd * BEYOND_DOOR_M))):
                    score -= BLOCKED_DOOR_PENALTY
    return _State(placements, {**state.polygons, b: poly}, union, pairs, frozenset(explained), state.islands, score)


def _meets(cd, nd, wd, ce, ne, we) -> bool:
    return (float(nd @ ne) < -0.9 and abs(wd - we) <= MAX_WIDTH_GAP_M
            and float(np.linalg.norm(cd + nd * WALL_THICKNESS_M - ce)) <= CONFIRM_REACH_M)


def _start(rooms: list[Room], order: list[int], k: int, state: _State | None) -> _State:
    """Room k as the first room of a plan, or as a new island beside an existing one."""
    if state is None:
        placement = Placement(0.0, (0.0, 0.0))
        poly = _plan_polygon(rooms[k], placement)
        return _State({k: placement}, {k: poly}, poly, [], frozenset(), 1, 0.0)
    minx, miny, _, _ = Polygon(rooms[k].polygon).bounds
    placement = Placement(0.0, (state.union.bounds[2] + ISLAND_GAP_M - minx, state.union.bounds[1] - miny))
    poly = _plan_polygon(rooms[k], placement)
    union = state.union.union(poly)
    return _State({**state.placements, k: placement}, {**state.polygons, k: poly}, union, list(state.pairs),
                  state.explained, state.islands + 1, state.score - ISLAND_PENALTY)


def _expand(rooms: list[Room], order: list[int], state: _State) -> list[_State] | None:
    open_doors = [(r, i) for r in sorted(state.placements) for i in range(len(rooms[r].doors))
                  if (r, i) not in state.explained]
    unplaced = [k for k in order if k not in state.placements]
    if not open_doors:
        return [_start(rooms, order, unplaced[0], state)] if unplaced else None
    r, i = open_doors[0]
    children = [dataclasses.replace(state, explained=state.explained | {(r, i)}, score=state.score - LEAVE_OPEN_PENALTY)]
    door = rooms[r].doors[i]
    centre, normal, _ = _door_in_plan(rooms, state.placements, r, i)
    for b in unplaced:
        for j, other in enumerate(rooms[b].doors):
            gap = abs(other.width_m - door.width_m)
            if gap > MAX_WIDTH_GAP_M:
                continue
            child = _add_room(rooms, state, b, _place(state.placements[r], door, other, centre, normal),
                              DoorPair(r, i, b, j), gap)
            if child is not None:
                children.append(child)
    return children


def total_overlap_m2(polygons) -> float:
    polys = [p.buffer(-OVERLAP_SHRINK_M) for p in polygons]
    return float(sum(a.intersection(b).area for k, a in enumerate(polys) for b in polys[k + 1:]))


def stitch(rooms: list[Room], beam_width: int | None = None) -> StitchResult:
    """One plan from separately framed rooms. The largest room fixes the plan's frame. `beam_width` defaults
    to BEAM_WIDTH, read at call time so benchmarks can override it."""
    if not rooms:
        return StitchResult({}, [], 0, 0.0, 0.0)
    beam_width = BEAM_WIDTH if beam_width is None else beam_width
    order = sorted(range(len(rooms)), key=lambda k: -_plan_polygon(rooms[k], Placement(0.0, (0.0, 0.0))).area)
    beam = [_start(rooms, order, order[0], None)]
    while True:
        grown, changed = [], False
        for state in beam:
            children = _expand(rooms, order, state)
            if children is None:
                grown.append(state)
            else:
                grown.extend(children)
                changed = True
        if not changed:
            break
        unique: dict = {}
        for s in sorted(grown, key=lambda s: -s.score):
            unique.setdefault(s.key(), s)
        beam = list(unique.values())[:beam_width]
    best = max(beam, key=lambda s: s.score)
    return StitchResult(best.placements, best.pairs, best.islands, total_overlap_m2(best.polygons.values()), best.score)
