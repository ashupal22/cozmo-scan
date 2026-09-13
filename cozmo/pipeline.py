"""`cozmo run`: one capture in, JSON and plan drawing out."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from cozmo.export.document import LIDAR_ERRORS, ErrorModel, build_document, photo_errors, video_errors
from cozmo.export.render import render_svg
from cozmo.export.validate import validate_output
from cozmo.geometry.fusion import fuse
from cozmo.geometry.layout import LIDAR_FACE_SCATTER_M, NotManhattan, build_layout
from cozmo.geometry.planes import HorizontalPlane, find_floors
from cozmo.geometry.rooms import build_room_map, room_ceilings
from cozmo.geometry.walls import OpeningOnWall, RoomOutline, WallSegment, attach_doorways, outline_rooms
from cozmo.ingest.detect import VIDEO_EXT, detect_tier
from cozmo.ingest.stray import CaptureError, StrayCapture
from cozmo.slam.drift import estimate_drift

REPO = Path(__file__).resolve().parents[1]
# Apple LiDAR depth reads short. On ARKitScenes (2020 iPad Pro, laser truth) the median device - laser depth is -11.9 mm
# over six walks, and adding it back brings ceiling height within 15 mm on 6/6 walks (bench/README.md). The transfer to
# iPhone LiDAR is not yet verified, so the ceiling interval keeps its full 13 mm bias term (cozmo/export/document.py).
LIDAR_DEPTH_OFFSET_M = 0.0119


class NotBuiltYet(RuntimeError):
    """The requested tier has no pipeline yet."""


def _code_version() -> str:
    out = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def plan_rooms(points, floor, positions, face_scatter_m: float = LIDAR_FACE_SCATTER_M):
    """Rooms from walls first; if the walls do not meet at right angles, fall back to rooms traced
    from the seen floor. Returns (room map, outlines, openings, note for the warnings or None)."""
    try:
        layout = build_layout(points, floor, positions, face_scatter_m=face_scatter_m)
        if layout.outlines:
            return layout.room_map, layout.outlines, layout.openings, None
        note = "walls-first layout found no rooms; outlines traced from the seen floor instead"
    except NotManhattan as e:
        note = f"walls do not meet at right angles ({e}); outlines traced from the seen floor instead"
    room_map = build_room_map(points, floor, positions)
    outlines = outline_rooms(room_map, points, floor)
    return room_map, outlines, attach_doorways(outlines, room_map), note


@dataclass
class Plan:
    document: dict
    yaw_deg: float
    points: object      # drift-corrected PointSet
    floor: object       # HorizontalPlane
    outlines: dict      # room id -> RoomOutline
    correction: object = None  # FrameCorrection (per-frame corrected camera positions), None without drift correction
    photo_rooms: list = None   # photo tier: the per-room models (cozmo/photo/room.py), each in its own frame
    capture: object = None     # LiDAR and video tiers: the capture, for damage detection on its images


def plan_capture(capture, path: Path, tier: str, t0: float, drift: bool = True, jumps=None, relocalized: bool = True,
                 errors: ErrorModel = LIDAR_ERRORS, face_scatter_m: float = LIDAR_FACE_SCATTER_M,
                 depth_offset_m: float = 0.0) -> Plan:
    """Shared by every tier that gives per-frame depth and poses: fuse, correct drift, find the floor,
    lay out rooms, measure ceilings, write the document."""
    points = fuse(capture, depth_offset_m=depth_offset_m)
    positions = capture.positions
    drift_summary = correction = None
    if drift:
        correction, report = estimate_drift(capture, points, jumps=jumps, relocalized=relocalized)
        points = correction.apply(points)
        positions = correction.positions[correction.valid]
        drift_summary = report.to_schema()
    floors = find_floors(points)
    if not floors:
        raise CaptureError(f"{path}: no floor found; the capture must show the floor")
    floor = floors[0]
    room_map, outlines, openings, method_note = plan_rooms(points, floor, positions, face_scatter_m)
    if not outlines:
        raise CaptureError(f"{path}: no room outline could be built")
    ceilings = room_ceilings(points, floor, room_map)
    info = {"id": Path(path).stem if tier == "video" else Path(path).name, "tier": tier, "device": None,
            "input_path": str(path), "pipeline_version": _code_version()}
    document, _ = build_document(info, floor, room_map, outlines, ceilings, openings, time.time() - t0,
                                 drift=drift_summary, errors=errors)
    for warning in capture.warnings:
        document["quality"]["warnings"].append(f"capture: {warning}")
    if method_note:
        document["quality"]["warnings"].append(method_note)
    yaw = next(iter(outlines.values())).yaw_deg
    return Plan(document, yaw, points, floor, outlines, correction, capture=capture)


def run_lidar(path: Path, drift: bool = True) -> Plan:
    t0 = time.time()
    plan = plan_capture(StrayCapture(path), path, "lidar", t0, drift=drift, depth_offset_m=LIDAR_DEPTH_OFFSET_M)
    plan.document["quality"]["warnings"].append(
        f"LiDAR depth corrected by +{1000 * LIDAR_DEPTH_OFFSET_M:.1f} mm (calibrated on a 2020 iPad Pro against laser scans; "
        f"not yet verified on iPhone, so ceiling intervals keep the full bias allowance)")
    return plan


VIDEO_FACE_SCATTER_M = 0.048  # video wall points across their wall, true poses, c00a170fe1 (LiDAR: 0.020)
# Fix-loop parts 1 and 2 (docs/fix_loop.md): declaring tracking breaks made every walk worse (footprint 1a83 -9.4 ->
# -35.3%, c7d2 -4.9 -> -48.2%), so it is off. Read at call time: the after-run regenerates with it switched on.
DECLARE_TRACKING_BREAKS = False


def plan_video_capture(capture, path: Path, t0: float, drift: bool = True) -> Plan:
    """Video walks are corrected like LiDAR walks: tracking breaks (capture.breaks) are reported but not declared to
    drift correction, which measured best (docs/fix_loop.md). With DECLARE_TRACKING_BREAKS the walk is cut there and
    loop closures across a break try every quarter turn."""
    jumps = list(capture.breaks) if DECLARE_TRACKING_BREAKS else []
    plan = plan_capture(capture, path, "video", t0, drift=drift, jumps=jumps, relocalized=False,
                        errors=video_errors(capture.scale_sigma), face_scatter_m=VIDEO_FACE_SCATTER_M)
    document, info = plan.document, capture.info
    document["quality"]["warnings"] += [
        f"no depth sensor: wall positions come from a learned depth model (Depth Anything 3) on {info['frames']} "
        f"key frames; camera tracking was doubtful at {len(capture.breaks)} place(s), where rooms seen before and after "
        f"may be misplaced (a break can turn the rest of the walk by up to a quarter turn)",
        f"real size set by a monocular metric-depth model, focal length from {info['focal_source']}: scale "
        f"uncertainty {100 * capture.scale_sigma:.1f}% (1 sigma), applied to every length",
    ]
    from cozmo.video.capture import KEYFRAME_FPS
    if info["keyframe_fps"] < KEYFRAME_FPS - 1e-6:
        document["quality"]["warnings"].append(
            f"long clip: key frames taken at {info['keyframe_fps']:.2f}/s instead of {KEYFRAME_FPS:.1f}/s, to keep the "
            f"run inside the 10 minute budget. Frames are further apart, so the camera path is less certain than on a "
            f"shorter clip of the same space")
    document["quality"]["low_confidence"] = True
    return plan


def run_video(path: Path, work_dir: Path, drift: bool = True, fx_over_width: float | None = None) -> Plan:
    from cozmo.video.capture import load_video  # torch and the DA3 models load only for videos
    t0 = time.time()
    return plan_video_capture(load_video(path, work_dir, fx_over_width=fx_over_width), path, t0, drift=drift)


def run_photos(path: Path, work_dir: Path, fx_over_width: float | None = None) -> Plan:
    """One folder per room: each room measured on its own (cozmo/photo/room.py), then joined through its doors."""
    import math

    from cozmo.ingest.photos import photo_rooms, photo_warnings
    from cozmo.photo.room import build_room
    from cozmo.stitch.solver import Room, stitch
    t0 = time.time()
    folders = photo_rooms(path)
    rooms = [build_room(name, files, Path(work_dir) / "cache", k + 1, fx_over_width)
             for k, (name, files) in enumerate(folders.items())]
    result = stitch([Room(r.name, r.outline.vertices, r.doors) for r in rooms])
    other = {}
    for pair in result.pairs:
        other[(pair.room_a, pair.door_a)] = pair.room_b
        other[(pair.room_b, pair.door_b)] = pair.room_a
    outlines, openings, ceilings = {}, [], {}
    for k, r in enumerate(rooms):
        at = result.placements[k]
        walls = [WallSegment(at.apply(w.start), at.apply(w.end), w.length_m, w.face_spread_m, w.support)
                 for w in r.outline.walls]
        outlines[k + 1] = RoomOutline(k + 1, r.outline.yaw_deg + math.degrees(at.angle), at.apply(r.outline.vertices), walls)
        for d in range(len(r.doors)):
            o = other.get((k, d))
            openings.append(OpeningOnWall(k + 1, r.door_walls[d], r.door_offsets[d], r.doors[d].width_m,
                                          None if o is None else o + 1, d, r.door_measured[d]))
        ceilings[k + 1] = HorizontalPlane(r.ceiling_m, r.ceiling_sigma_m, 1) if r.ceiling_m is not None else None
    info = {"id": Path(path).name, "tier": "photo", "device": None, "input_path": str(path),
            "pipeline_version": _code_version()}
    document, _ = build_document(info, HorizontalPlane(0.0, 0.0, 1), None, outlines, ceilings, openings,
                                 time.time() - t0, drift=None, errors=photo_errors(max(r.scale_sigma for r in rooms)))
    for room, r in zip(document["rooms"], rooms):
        room["label"] = r.name
    plan = document["plan"]
    plan["drift"] = {"enabled": False, "correction": [], "notes": "not applicable: each room is measured from its own photos"}
    plan["stitch"] = {"method": "door_matching", "low_confidence": True,
                      "notes": f"{len(result.pairs)} door pair(s) join {len(rooms)} rooms into {result.islands} group(s); "
                               f"groups not joined by a door are set beside the plan, where they really are is unknown"}
    warnings = document["quality"]["warnings"]
    warnings[:] = [w for w in warnings if not w.startswith("drift correction switched off")]
    warnings += photo_warnings(folders) + [w for r in rooms for w in r.warnings] + [
        "no depth sensor and no camera poses: each room's walls come from a learned depth model (Depth Anything 3) "
        "on its own photos, sized by a monocular metric-depth model; focal length from "
        + ", ".join(sorted({r.focal_source for r in rooms})),
        f"rooms joined through their doors: {result.islands} group(s); the photo tier's stitch is not yet reliable "
        f"(bench/README.md), so check the adjacency"]
    document["quality"]["low_confidence"] = True
    yaw = next(iter(outlines.values())).yaw_deg
    return Plan(document, yaw, None, None, outlines, None, rooms)


def run(path, out_dir, drift: bool = True, damage: bool = True) -> Path:
    path, out_dir = Path(path), Path(out_dir)
    tier = detect_tier(path)
    if tier == "lidar":
        plan = run_lidar(path, drift=drift)
    elif tier == "video":
        video = path if path.is_file() else next(c for c in sorted(path.iterdir()) if c.suffix.lower() in VIDEO_EXT)
        plan = run_video(video, out_dir / "video_work", drift=drift)
    else:
        plan = run_photos(path, out_dir / "photo_work")
    document, yaw = plan.document, plan.yaw_deg
    out_dir.mkdir(parents=True, exist_ok=True)
    if damage:
        from cozmo.damage.assess import assess
        assess(plan, tier, out_dir)
    else:
        document["quality"]["warnings"].append("damage detection switched off (--no-damage)")
    svg_path = out_dir / "plan.svg"
    document["render"] = {"plan_svg": str(svg_path)}
    problems = validate_output(document)
    if problems:
        raise RuntimeError("output failed validation:\n  " + "\n  ".join(problems))
    (out_dir / "result.json").write_text(json.dumps(document, indent=2) + "\n")
    svg_path.write_text(render_svg(document, yaw))
    return out_dir / "result.json"
