"""Photo tier: one folder of 2 to 8 stills per room, no depth and no poses (docs/capture_protocol.md, tier 3).

Each room:
1. Depth Anything 3 gives the room's cameras and depth in one frame (DA3-BASE, up to scale), and the metric model
   with the photo's EXIF focal length sets the scale. This is the video tier's calibrated chain
   (cozmo/video/capture.py) run on one set of views: the same metric gain and range correction, levelled on the
   floor and walls.
2. The room is a right-angled box. On each of its four sides the wall is the farthest face with at least half the
   best coverage on that side (nearer faces are furniture). A side no photo saw is closed a step behind the
   cameras, because the protocol has the photos taken from the doorway. It is marked unseen and gets a wide
   interval.
3. Doors are gaps in a box wall through which the photos see low surfaces beyond the wall (the next room's floor
   or wall base; windows have sills, so they do not qualify), 0.6 to 1.6 m wide. The doorway the photos were taken
   from is added on the unseen side, at the cameras, with a typical width.
The pipeline then joins the rooms through their doors (cozmo/stitch/solver.py).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cozmo.geometry.fusion import fuse
from cozmo.geometry.planes import HorizontalPlane, find_ceilings, find_floors
from cozmo.geometry.walls import RoomOutline, WallSegment, _to_aligned, dominant_yaw
from cozmo.stitch.solver import Door

MAX_DEPTH_M = 8.0                  # DA3 depth has no sensor range limit; beyond this it is too coarse to use
WALL_HEIGHT_M = (0.3, 2.0)
FACING_COS = float(np.cos(np.radians(12)))
BAND_M = 0.05
MIN_SIDE_POINTS = 50
MIN_COVER_M = 0.3
UNSEEN_GAP_M = 0.3                 # the wall behind a photographer standing in the doorway
UNSEEN_FACE_SIGMA_M = 0.5          # where that wall is, if the protocol was not followed
DOOR_WIDTH_M = (0.6, 1.6)
TYPICAL_DOOR_M = 0.85
THROUGH_BEYOND_M = 0.25            # points this far behind a wall were seen through an opening in it
THROUGH_HEIGHT_M = (0.05, 0.7)     # low: the next room's floor and wall base; windows have sills
BIN_M = 0.05
CAMERA_HEIGHT_PRIOR_M = 1.4
# side (axis, sign) -> wall index of the box outline u0v0 -> u1v0 -> u1v1 -> u0v1
SIDE_WALL = {(1, -1): 0, (0, 1): 1, (1, 1): 2, (0, -1): 3}


@dataclass
class PhotoRoom:
    name: str
    photos: int
    outline: RoomOutline            # room frame (x, z), metres
    doors: list[Door]               # room frame, for the stitch
    door_walls: list[int]           # wall index of each door
    door_offsets: list[float]       # from the wall start to the door centre
    door_measured: list[bool]       # False: the width is typical, not measured
    ceiling_m: float | None         # ceiling above floor, None when no photo saw the ceiling
    ceiling_sigma_m: float
    scale_sigma: float
    focal_source: str
    sides_seen: int
    capture: object = None          # VideoCapture-like views, for damage detection
    points: object = None
    floor: HorizontalPlane | None = None
    warnings: list[str] = field(default_factory=list)


def _file_key(files: list[Path]) -> str:
    h = hashlib.sha1(b"photo-room-v1")
    for f in files:
        size = f.stat().st_size
        h.update(f"{f.name}:{size}".encode())
        with open(f, "rb") as fh:
            h.update(fh.read(1 << 20))
    return h.hexdigest()[:12]


def room_capture(files: list[Path], cache_dir: Path, fx_over_width: float | None = None):
    """The room's photos as a levelled, metric set of views (the VideoCapture interface fusion expects)."""
    from cozmo.ingest.photos import load_photo
    from cozmo.video import capture as vc
    from cozmo.video import da3
    from cozmo.video.focal import estimate_focal

    loaded = [load_photo(p) for p in files]
    images = [rgb for rgb, _ in loaded]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{_file_key(files)}.npz"
    if not cache.is_file():
        vs = da3.run_views(images, process_res=vc.PROCESS_RES, ray_pose=vc.RAY_POSE)
        unit_metric = da3.metric_depth(images, fx_over_width=1.0, process_res=vc.PROCESS_RES)
        np.savez(cache, depth=vs.depth.astype(np.float16), conf=vs.conf.astype(np.float16), w2c=vs.world_to_cam,
                 K=vs.K, size=np.array(vs.size), m1=unit_metric.astype(np.float16))
    z = np.load(cache)              # a first run reads back what it saved, so it matches every later run exactly
    vs = da3.ViewSet(z["depth"].astype(np.float32), z["conf"].astype(np.float32), z["w2c"], z["K"], tuple(z["size"]))
    unit_metric = z["m1"].astype(np.float32)
    n, (h, w) = len(files), vs.depth.shape[1:]

    exif = [f for _, f in loaded if f]
    if fx_over_width is not None:
        focal_source, focal_sigma = "given", vc.FOCAL_SIGMA_GIVEN
    elif exif:
        fx_over_width, focal_source, focal_sigma = float(np.median(exif)), "EXIF", vc.FOCAL_SIGMA_GIVEN
    else:
        est = estimate_focal([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images])
        if est is not None:
            fx_over_width, focal_sigma = est.fx_over_width, max(est.sigma, vc.FOCAL_SIGMA_LINES_MIN)
            focal_source = "room lines (no EXIF focal length)"
        else:
            fx_over_width = float(np.median(vs.K[:, 0, 0] / w))
            focal_source, focal_sigma = "DA3 estimate (no EXIF focal length, too few straight lines)", vc.FOCAL_SIGMA_DA3
    K = vc.camera_matrix(fx_over_width, w, h)
    metric = {i: unit_metric[i] * fx_over_width * vc.METRIC_DEPTH_GAIN for i in range(n)}
    chain = vc.chain_runs([vs], [(0, n)], K, metric)
    conf = np.zeros(chain.depth.shape, np.uint8)
    for i in range(n):
        conf[i][vc._confident(chain.conf[i])] = 2
    scale_sigma = float(np.linalg.norm([vc.METRIC_SIGMA / np.sqrt(max(chain.info["metric_anchors"], 1)),
                                        focal_sigma, vc.METRIC_BIAS_SIGMA]))
    cap = vc.VideoCapture(Path(files[0]).parent, list(files), np.arange(n, dtype=float), chain.c2w[:, :3, 3].copy(),
                          chain.c2w[:, :3, :3].copy(), vc.correct_range(chain.depth), conf, np.repeat(K[None], n, 0),
                          scale_sigma, dict(chain.info))
    return cap, focal_source


def _robust_spread(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values)))) if len(values) else float("nan")


def fit_box(points, floor: HorizontalPlane, cameras: np.ndarray):
    """Right-angled box in the room frame: (yaw, extents {(axis, sign): position}, seen {(axis, sign): cover or None},
    face spread {(axis, sign): m}, face support {(axis, sign): points})."""
    h = points.xyz[:, 1] - floor.height
    nrm = points.normal
    wall = (np.abs(nrm[:, 1]) < 0.2) & (h > WALL_HEIGHT_M[0]) & (h < WALL_HEIGHT_M[1])
    nxz = nrm[wall][:, [0, 2]].astype(float)
    nxz /= np.linalg.norm(nxz, axis=1, keepdims=True) + 1e-9
    yaw, _ = dominant_yaw(nxz) if len(nxz) else (0.0, 0.0)
    R = _to_aligned(yaw)
    uv = points.xyz[wall][:, [0, 2]].astype(float) @ R.T
    nuv = nxz @ R.T
    cam = np.median(cameras[:, [0, 2]] @ R.T, axis=0)
    on_floor = (nrm[:, 1] > 0.9) & (np.abs(h) < 0.08)
    fuv = points.xyz[on_floor][:, [0, 2]].astype(float) @ R.T
    ext, seen, spread, support = {}, {}, {}, {}
    for axis in (0, 1):
        for sign in (1, -1):
            facing = nuv[:, axis] * -sign > FACING_COS
            pos, along = uv[facing, axis], uv[facing, 1 - axis]
            ahead = (pos - cam[axis]) * sign > 0.3
            pos, along = pos[ahead], along[ahead]
            best = None
            if len(pos) >= MIN_SIDE_POINTS:
                centres = np.unique(np.round(pos / 0.02)) * 0.02
                cover = np.array([len(np.unique(np.floor(along[np.abs(pos - c) < BAND_M] / BIN_M))) * BIN_M
                                  for c in centres])
                strong = centres[(cover >= 0.5 * cover.max()) & (cover >= MIN_COVER_M)]
                if len(strong):
                    far = strong[np.argmax(strong * sign)]
                    near = np.abs(pos - far) < BAND_M
                    best = (float(np.median(pos[near])), float(cover.max()), _robust_spread(pos[near]), int(near.sum()))
            if best is not None:
                ext[(axis, sign)], seen[(axis, sign)] = best[0], round(best[1], 2)
                spread[(axis, sign)], support[(axis, sign)] = best[2], best[3]
            else:
                closed = cam[axis] + sign * UNSEEN_GAP_M
                if len(fuv):
                    floor_edge = float(fuv[:, axis].max() if sign > 0 else fuv[:, axis].min())
                    closed = closed if (closed - floor_edge) * sign > 0 else floor_edge
                ext[(axis, sign)], seen[(axis, sign)] = closed, None
                spread[(axis, sign)], support[(axis, sign)] = UNSEEN_FACE_SIGMA_M, 1
    return yaw, ext, seen, spread, support, cam


def find_doors(points, floor: HorizontalPlane, yaw: float, ext: dict, seen: dict) -> list[tuple]:
    """(side, centre along the side, width) of openings seen through on the walls the photos saw."""
    R = _to_aligned(yaw)
    h = points.xyz[:, 1] - floor.height
    uv = points.xyz[:, [0, 2]].astype(float) @ R.T
    low = (h > THROUGH_HEIGHT_M[0]) & (h < THROUGH_HEIGHT_M[1])
    doors = []
    for (axis, sign), pos in ext.items():
        if seen[(axis, sign)] is None:
            continue
        lo, hi = ext[(1 - axis, -1)], ext[(1 - axis, 1)]
        edges = np.arange(lo, hi + BIN_M, BIN_M)
        if len(edges) < 3:
            continue
        beyond = (uv[:, axis] - pos) * sign
        through = low & (beyond > THROUGH_BEYOND_M)
        at_wall = (np.abs(beyond) < 2 * BAND_M) & (h > THROUGH_HEIGHT_M[0]) & (h < WALL_HEIGHT_M[1])
        n_through, _ = np.histogram(uv[through, 1 - axis], edges)
        n_wall, _ = np.histogram(uv[at_wall, 1 - axis], edges)
        open_bin = (n_through >= 5) & (n_wall < np.maximum(5, 0.2 * n_through))
        k = 0
        while k < len(open_bin):
            if not open_bin[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(open_bin) and open_bin[j + 1]:
                j += 1
            width = (j - k + 1) * BIN_M
            if DOOR_WIDTH_M[0] <= width <= DOOR_WIDTH_M[1] and k > 0 and j < len(open_bin) - 1:
                doors.append(((axis, sign), float(edges[k] + width / 2), width))
            k = j + 1
    return doors


def box_outline(yaw: float, ext: dict, spread: dict, support: dict, room_id: int) -> RoomOutline:
    u0, u1, v0, v1 = ext[(0, -1)], ext[(0, 1)], ext[(1, -1)], ext[(1, 1)]
    Rt = _to_aligned(yaw).T
    corners = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]) @ Rt.T
    walls = []
    for k, side in enumerate(((1, -1), (0, 1), (1, 1), (0, -1))):
        a, b = corners[k], corners[(k + 1) % 4]
        walls.append(WallSegment(a, b, float(np.linalg.norm(b - a)), float(spread[side]), int(support[side])))
    return RoomOutline(room_id, yaw, corners, walls)


def _offset(side, along: float, ext: dict) -> float:
    """Distance from the start of the box wall on `side` to the point `along` it."""
    wall = SIDE_WALL[side]
    if wall == 0:
        return along - ext[(0, -1)]
    if wall == 1:
        return along - ext[(1, -1)]
    if wall == 2:
        return ext[(0, 1)] - along
    return ext[(1, 1)] - along


def build_room(name: str, files: list[Path], cache_dir: Path, room_id: int,
               fx_over_width: float | None = None) -> PhotoRoom:
    cap, focal_source = room_capture(files, cache_dir, fx_over_width)
    points = fuse(cap, frames=np.arange(len(cap)), max_depth=MAX_DEPTH_M)
    warnings = []
    floors = find_floors(points)
    if floors:
        floor = floors[0]
    else:
        floor = HorizontalPlane(float(np.median(cap.positions[:, 1])) - CAMERA_HEIGHT_PRIOR_M, 0.1, 1)
        warnings.append(f"{name}: no floor seen; floor placed {CAMERA_HEIGHT_PRIOR_M} m below the camera")
    yaw, ext, seen, spread, support, cam = fit_box(points, floor, cap.positions)
    outline = box_outline(yaw, ext, spread, support, room_id)
    Rt = _to_aligned(yaw).T

    doors, walls, offsets, measured = [], [], [], []

    def add(side, along, width, was_measured):
        axis, sign = side
        uv = np.zeros(2)
        uv[axis], uv[1 - axis] = ext[side], along
        normal = np.zeros(2)
        normal[axis] = sign
        doors.append(Door(tuple(Rt @ uv), tuple(Rt @ normal), float(width)))
        walls.append(SIDE_WALL[side])
        offsets.append(float(_offset(side, along, ext)))
        measured.append(was_measured)

    for side, along, width in find_doors(points, floor, yaw, ext, seen):
        add(side, along, width, True)
    unseen = [side for side, cover in seen.items() if cover is None]
    if unseen:
        # the doorway the photos were taken from: on the unseen side nearest the cameras
        side = min(unseen, key=lambda s: abs(ext[s] - cam[s[0]]))
        lo, hi = ext[(1 - side[0], -1)], ext[(1 - side[0], 1)]
        half = TYPICAL_DOOR_M / 2
        add(side, float(np.clip(cam[1 - side[0]], lo + half, hi - half)) if hi - lo > TYPICAL_DOOR_M else (lo + hi) / 2,
            TYPICAL_DOOR_M, False)
        warnings.append(f"{name}: {len(unseen)} of 4 walls not seen; closed at the camera (photos taken from the "
                        f"doorway), with a wide interval")

    ceilings = find_ceilings(points, floor)
    ceiling_m = ceilings[0].height - floor.height if ceilings else None
    ceiling_sigma = float(np.hypot(ceilings[0].standard_error, floor.standard_error)) if ceilings else float("nan")
    return PhotoRoom(name, len(files), outline, doors, walls, offsets, measured, ceiling_m, ceiling_sigma,
                     cap.scale_sigma, focal_source, sum(c is not None for c in seen.values()), cap, points, floor,
                     warnings)
