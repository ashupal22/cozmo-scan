"""`cozmo run`: one capture in, JSON and plan drawing out."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from cozmo.export.document import build_document
from cozmo.export.render import render_svg
from cozmo.export.validate import validate_output
from cozmo.geometry.fusion import fuse
from cozmo.geometry.layout import NotManhattan, build_layout
from cozmo.geometry.planes import find_floors
from cozmo.geometry.rooms import build_room_map, room_ceilings
from cozmo.geometry.walls import attach_doorways, outline_rooms
from cozmo.ingest.detect import detect_tier
from cozmo.ingest.stray import CaptureError, StrayCapture
from cozmo.slam.drift import estimate_drift

REPO = Path(__file__).resolve().parents[1]


class NotBuiltYet(RuntimeError):
    """The requested tier has no pipeline yet."""


def _code_version() -> str:
    out = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def plan_rooms(points, floor, positions):
    """Rooms from walls first; if the walls do not meet at right angles, fall back to rooms traced
    from the seen floor. Returns (room map, outlines, openings, note for the warnings or None)."""
    try:
        layout = build_layout(points, floor, positions)
        if layout.outlines:
            return layout.room_map, layout.outlines, layout.openings, None
        note = "walls-first layout found no rooms; outlines traced from the seen floor instead"
    except NotManhattan as e:
        note = f"walls do not meet at right angles ({e}); outlines traced from the seen floor instead"
    room_map = build_room_map(points, floor, positions)
    outlines = outline_rooms(room_map, points, floor)
    return room_map, outlines, attach_doorways(outlines, room_map), note


def run_lidar(path: Path, drift: bool = True) -> tuple[dict, float]:
    t0 = time.time()
    capture = StrayCapture(path)
    points = fuse(capture)
    positions = capture.positions
    drift_summary = None
    if drift:
        correction, report = estimate_drift(capture, points)
        points = correction.apply(points)
        positions = correction.positions[correction.valid]
        drift_summary = report.to_schema()
    floors = find_floors(points)
    if not floors:
        raise CaptureError(f"{path}: no floor found; the capture must show the floor")
    floor = floors[0]
    room_map, outlines, openings, method_note = plan_rooms(points, floor, positions)
    if not outlines:
        raise CaptureError(f"{path}: no room outline could be built")
    ceilings = room_ceilings(points, floor, room_map)
    info = {"id": Path(path).name, "tier": "lidar", "device": None, "input_path": str(path),
            "pipeline_version": _code_version()}
    document, _ = build_document(info, floor, room_map, outlines, ceilings, openings, time.time() - t0,
                                 drift=drift_summary)
    for warning in capture.warnings:
        document["quality"]["warnings"].append(f"capture: {warning}")
    if method_note:
        document["quality"]["warnings"].append(method_note)
    yaw = next(iter(outlines.values())).yaw_deg
    return document, yaw


def run(path, out_dir, drift: bool = True) -> Path:
    path, out_dir = Path(path), Path(out_dir)
    tier = detect_tier(path)
    if tier != "lidar":
        raise NotBuiltYet(f"the {tier} tier is not built yet; only LiDAR captures run for now")
    document, yaw = run_lidar(path, drift=drift)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "plan.svg"
    document["render"] = {"plan_svg": str(svg_path)}
    problems = validate_output(document)
    if problems:
        raise RuntimeError("output failed validation:\n  " + "\n  ".join(problems))
    (out_dir / "result.json").write_text(json.dumps(document, indent=2) + "\n")
    svg_path.write_text(render_svg(document, yaw))
    return out_dir / "result.json"
