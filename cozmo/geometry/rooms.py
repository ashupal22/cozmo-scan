"""Plan-view maps and room segmentation for a LiDAR capture.

1. Rasterise on a 5 cm plan grid: where the floor was seen, where walls are (vertical surfaces
   between 0.3 and 1.8 m above the floor), and where the person walked.
2. Interior = seen floor or walked path, small gaps closed, walls removed, enclosed holes filled
   (furniture footprints inside a room become interior again).
3. Rooms, the morphological method from the room-segmentation survey (Bormann et al., ICRA 2016):
   cells further than DOOR_HALF_WIDTH_M from any non-interior cell form room "cores", because a
   doorway pinches shut at that radius while rooms stay open. Cores grow back over the interior
   (breadth-first, so growth never crosses a wall), and rooms below MIN_ROOM_M2 merge into the
   neighbour they share the longest boundary with.
4. Doorways are the boundaries between two rooms, measured along their long axis. These are coarse
   (5 cm cells); exact opening widths come later from image edges.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from cozmo.geometry.fusion import PointSet
from cozmo.geometry.planes import HorizontalPlane, ceiling_mask, find_ceilings

CELL_M = 0.05
FLOOR_BAND_M = 0.05
FLOOR_NORMAL_DOT = 0.9
WALL_MIN_ABOVE_FLOOR_M = 0.3
WALL_MAX_ABOVE_FLOOR_M = 1.8
WALL_NORMAL_MAX_DOT = 0.2
WALL_MIN_HITS = 3
PATH_RADIUS_M = 0.25
CLOSE_RADIUS_M = 0.15
DOOR_HALF_WIDTH_M = 0.5      # doorways up to ~1.0 m wide separate rooms
MIN_CORE_M2 = 0.1
MIN_ROOM_M2 = 1.5
MIN_DOORWAY_M = 0.5          # narrower links are gaps in wall evidence, not doors


def _disk(radius_cells: float) -> np.ndarray:
    r = max(int(round(radius_cells)), 1)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


@dataclass(frozen=True)
class PlanGrid:
    """Plan-view grid: row index follows world z, column index follows world x."""

    x0: float
    z0: float
    cell: float
    rows: int
    cols: int

    @classmethod
    def around(cls, xz: np.ndarray, cell: float = CELL_M, margin: float = 0.5) -> "PlanGrid":
        lo = xz.min(axis=0) - margin
        hi = xz.max(axis=0) + margin
        cols, rows = np.ceil((hi - lo) / cell).astype(int)
        return cls(float(lo[0]), float(lo[1]), cell, int(rows), int(cols))

    def index(self, xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols = np.clip(((xz[:, 0] - self.x0) / self.cell).astype(int), 0, self.cols - 1)
        rows = np.clip(((xz[:, 1] - self.z0) / self.cell).astype(int), 0, self.rows - 1)
        return rows, cols

    def raster(self, xz: np.ndarray, min_hits: int = 1) -> np.ndarray:
        counts = np.zeros((self.rows, self.cols), np.int32)
        if len(xz):
            rows, cols = self.index(xz)
            np.add.at(counts, (rows, cols), 1)
        return counts >= min_hits

    def centers(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return np.column_stack([self.x0 + (cols + 0.5) * self.cell, self.z0 + (rows + 0.5) * self.cell])

    def area_m2(self, mask: np.ndarray) -> float:
        return float(mask.sum()) * self.cell ** 2


@dataclass
class PlanMaps:
    grid: PlanGrid
    floor: np.ndarray  # floor seen
    walls: np.ndarray  # vertical surfaces at wall height
    path: np.ndarray   # walked area


@dataclass
class Doorway:
    rooms: tuple[int, int]
    center_xz: tuple[float, float]
    width_m: float


@dataclass
class RoomMap:
    maps: PlanMaps
    interior: np.ndarray
    labels: np.ndarray                     # 0 = not a room, 1..n = room id (largest first)
    areas_m2: dict[int, float] = field(default_factory=dict)
    doorways: list[Doorway] = field(default_factory=list)

    @property
    def grid(self) -> PlanGrid:
        return self.maps.grid

    @property
    def room_ids(self) -> list[int]:
        return sorted(self.areas_m2)

    def room_of(self, xz: np.ndarray) -> np.ndarray:
        rows, cols = self.grid.index(xz)
        return self.labels[rows, cols]


def plan_maps(points: PointSet, floor: HorizontalPlane, camera_positions: np.ndarray,
              cell: float = CELL_M) -> PlanMaps:
    grid = PlanGrid.around(np.vstack([points.xyz[:, [0, 2]], camera_positions[:, [0, 2]]]), cell)
    above = points.xyz[:, 1] - floor.height
    on_floor = (points.normal[:, 1] > FLOOR_NORMAL_DOT) & (np.abs(above) < FLOOR_BAND_M)
    on_wall = ((np.abs(points.normal[:, 1]) < WALL_NORMAL_MAX_DOT)
               & (above > WALL_MIN_ABOVE_FLOOR_M) & (above < WALL_MAX_ABOVE_FLOOR_M))
    path = ndimage.binary_dilation(grid.raster(camera_positions[:, [0, 2]]), structure=_disk(PATH_RADIUS_M / cell))
    return PlanMaps(grid, grid.raster(points.xyz[on_floor][:, [0, 2]]),
                    grid.raster(points.xyz[on_wall][:, [0, 2]], WALL_MIN_HITS), path)


def interior_mask(maps: PlanMaps) -> np.ndarray:
    free = ndimage.binary_closing(maps.floor | maps.path, structure=_disk(CLOSE_RADIUS_M / maps.grid.cell))
    return ndimage.binary_fill_holes(free & ~maps.walls)


def _grow(seeds: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multi-source breadth-first growth of labelled seeds inside mask (4-connected)."""
    labels = seeds.copy()
    rows, cols = mask.shape
    queue = deque(zip(*np.nonzero(labels)))
    while queue:
        r, c = queue.popleft()
        lab = labels[r, c]
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= rr < rows and 0 <= cc < cols and mask[rr, cc] and labels[rr, cc] == 0:
                labels[rr, cc] = lab
                queue.append((rr, cc))
    return labels


def _shared_boundary(labels: np.ndarray, lab: int) -> dict[int, int]:
    region = labels == lab
    ring = ndimage.binary_dilation(region) & ~region
    others, counts = np.unique(labels[ring], return_counts=True)
    return {int(o): int(n) for o, n in zip(others, counts) if o > 0}


def _merge_small_rooms(labels: np.ndarray, cell: float, min_room_m2: float) -> np.ndarray:
    while True:
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        small = [(n, int(i)) for i, n in zip(ids, counts) if n * cell ** 2 < min_room_m2]
        if not small:
            return labels
        _, lab = min(small)
        neighbours = _shared_boundary(labels, lab)
        labels[labels == lab] = max(neighbours, key=neighbours.get) if neighbours else 0


def _relabel_by_area(labels: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    lut = np.zeros(labels.max() + 1, labels.dtype)
    lut[order] = np.arange(1, len(order) + 1)
    return lut[labels]


def _doorways(labels: np.ndarray, grid: PlanGrid) -> list[Doorway]:
    boundaries: dict[tuple[int, int], np.ndarray] = {}
    for a, b in ((labels[:-1, :], labels[1:, :]), (labels[:, :-1], labels[:, 1:])):
        touching = (a > 0) & (b > 0) & (a != b)
        for r, c in zip(*np.nonzero(touching)):
            key = tuple(sorted((int(a[r, c]), int(b[r, c]))))
            boundaries.setdefault(key, np.zeros(labels.shape, bool))[r, c] = True

    doorways = []
    for key, img in boundaries.items():
        segments, n = ndimage.label(img, structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            rows, cols = np.nonzero(segments == k)
            xz = grid.centers(rows, cols)
            centred = xz - xz.mean(axis=0)
            axis = np.linalg.svd(centred, full_matrices=False)[2][0] if len(xz) > 1 else np.array([1.0, 0.0])
            width = float(np.ptp(centred @ axis) + grid.cell)
            if width >= MIN_DOORWAY_M:
                doorways.append(Doorway(key, (float(xz[:, 0].mean()), float(xz[:, 1].mean())), width))
    return doorways


def segment_rooms(maps: PlanMaps, door_half_width_m: float = DOOR_HALF_WIDTH_M,
                  min_room_m2: float = MIN_ROOM_M2) -> RoomMap:
    grid = maps.grid
    interior = interior_mask(maps)
    distance = ndimage.distance_transform_edt(interior) * grid.cell
    cores, n = ndimage.label(distance > door_half_width_m)
    if n:
        core_area = ndimage.sum_labels(np.ones_like(cores), cores, index=np.arange(1, n + 1)) * grid.cell ** 2
        keep = np.concatenate([[False], core_area >= MIN_CORE_M2])
        cores[~keep[cores]] = 0
    labels = _grow(cores.astype(np.int32), interior)
    labels = _merge_small_rooms(labels, grid.cell, min_room_m2)
    labels = _relabel_by_area(labels) if labels.max() > 0 else labels
    areas = {int(i): grid.area_m2(labels == i) for i in np.unique(labels[labels > 0])}
    return RoomMap(maps, interior, labels, areas, _doorways(labels, grid))


def build_room_map(points: PointSet, floor: HorizontalPlane, camera_positions: np.ndarray, **kwargs) -> RoomMap:
    return segment_rooms(plan_maps(points, floor, camera_positions), **kwargs)


def room_ceilings(points: PointSet, floor: HorizontalPlane, room_map: RoomMap) -> dict[int, HorizontalPlane | None]:
    """Largest ceiling plane over each room, or None where that room's ceiling was never seen."""
    mask = ceiling_mask(points, floor)
    ceiling_points = points.subset(mask)
    rooms = room_map.room_of(ceiling_points.xyz[:, [0, 2]])
    result = {}
    for rid in room_map.room_ids:
        found = find_ceilings(ceiling_points.subset(rooms == rid), floor)
        result[rid] = found[0] if found else None
    return result


def render(room_map: RoomMap, scale: int = 2) -> np.ndarray:
    """Colour image of rooms (tinted), walls (dark) and doorways (red dots) for visual checks."""
    rng = np.random.default_rng(7)
    palette = np.vstack([[255, 255, 255], rng.integers(90, 230, (room_map.labels.max() + 1, 3))]).astype(np.uint8)
    img = palette[np.where(room_map.labels > 0, room_map.labels + 1, 0)]
    img[room_map.maps.walls] = (40, 40, 40)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    g = room_map.grid
    for d in room_map.doorways:
        c = (int((d.center_xz[0] - g.x0) / g.cell * scale), int((d.center_xz[1] - g.z0) / g.cell * scale))
        cv2.circle(img, c, 3 * scale, (0, 0, 255), -1)
    for rid in room_map.room_ids:
        rows, cols = np.nonzero(room_map.labels == rid)
        cv2.putText(img, str(rid), (int(cols.mean() * scale), int(rows.mean() * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img
