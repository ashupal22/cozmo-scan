"""`cozmo run`: one capture in, JSON and plan drawing out."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from cozmo.export.document import LIDAR_ERRORS, ErrorModel, build_document, video_errors
from cozmo.export.render import render_svg
from cozmo.export.validate import validate_output
from cozmo.geometry.fusion import fuse
from cozmo.geometry.layout import LIDAR_FACE_SCATTER_M, NotManhattan, build_layout
from cozmo.geometry.planes import find_floors
from cozmo.geometry.rooms import build_room_map, room_ceilings
from cozmo.geometry.walls import attach_doorways, outline_rooms
from cozmo.ingest.detect import VIDEO_EXT, detect_tier
from cozmo.ingest.stray import CaptureError, StrayCapture
from cozmo.slam.drift import estimate_drift

REPO = Path(__file__).resolve().parents[1]


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


def plan_capture(capture, path: Path, tier: str, t0: float, drift: bool = True, jumps=None, relocalized: bool = True,
                 errors: ErrorModel = LIDAR_ERRORS, face_scatter_m: float = LIDAR_FACE_SCATTER_M) -> Plan:
    """Shared by every tier that gives per-frame depth and poses: fuse, correct drift, find the floor,
    lay out rooms, measure ceilings, write the document."""
    points = fuse(capture)
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
    return Plan(document, yaw, points, floor, outlines, correction)


def run_lidar(path: Path, drift: bool = True) -> Plan:
    t0 = time.time()
    return plan_capture(StrayCapture(path), path, "lidar", t0, drift=drift)


VIDEO_FACE_SCATTER_M = 0.048  # video wall points across their wall, true poses, c00a170fe1 (LiDAR: 0.020)


def plan_video_capture(capture, path: Path, t0: float, drift: bool = True) -> Plan:
    """Video walks are not cut at their tracking breaks (capture.breaks). On our three walks, cutting there and
    re-attaching the pieces made the wall map agree better with LiDAR (1a8384c3f6 0.55 -> 0.63, c7d28f72c6
    0.39 -> 0.49) but did not improve the plans (bench/README.md), so the breaks are only reported."""
    plan = plan_capture(capture, path, "video", t0, drift=drift, jumps=[], relocalized=False,
                        errors=video_errors(capture.scale_sigma), face_scatter_m=VIDEO_FACE_SCATTER_M)
    document, info = plan.document, capture.info
    document["quality"]["warnings"] += [
        f"no depth sensor: wall positions come from a learned depth model (Depth Anything 3) on {info['frames']} "
        f"key frames; camera tracking was doubtful at {len(capture.breaks)} place(s), where parts of the plan may be "
        f"turned or shifted",
        f"real size set by a monocular metric-depth model, focal length from {info['focal_source']}: scale "
        f"uncertainty {100 * capture.scale_sigma:.1f}% (1 sigma), applied to every length",
    ]
    document["quality"]["low_confidence"] = True
    return plan


def run_video(path: Path, work_dir: Path, drift: bool = True, fx_over_width: float | None = None) -> Plan:
    from cozmo.video.capture import load_video  # torch and the DA3 models load only for videos
    t0 = time.time()
    return plan_video_capture(load_video(path, work_dir, fx_over_width=fx_over_width), path, t0, drift=drift)


def run(path, out_dir, drift: bool = True) -> Path:
    path, out_dir = Path(path), Path(out_dir)
    tier = detect_tier(path)
    if tier == "lidar":
        plan = run_lidar(path, drift=drift)
    elif tier == "video":
        video = path if path.is_file() else next(c for c in sorted(path.iterdir()) if c.suffix.lower() in VIDEO_EXT)
        plan = run_video(video, out_dir / "video_work", drift=drift)
    else:
        raise NotBuiltYet(f"the {tier} tier is not built yet")
    document, yaw = plan.document, plan.yaw_deg
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "plan.svg"
    document["render"] = {"plan_svg": str(svg_path)}
    problems = validate_output(document)
    if problems:
        raise RuntimeError("output failed validation:\n  " + "\n  ".join(problems))
    (out_dir / "result.json").write_text(json.dumps(document, indent=2) + "\n")
    svg_path.write_text(render_svg(document, yaw))
    return out_dir / "result.json"
