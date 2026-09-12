"""Drift ablation (brief: "an ablation shows the stitched footprint with it on and off").

Runs the LiDAR pipeline on each walk twice, drift correction off and on, and compares:
  - the plan itself: rooms, room areas, footprint, doorways
  - map sharpness: area covered by wall points (walls seen twice collapse onto each other when drift is gone)
  - wall heading spread across the walk
  - repeatability: walls of two walks of the same flat, after one best rigid fit (no ground truth needed)

    COZMO_DATA=/path/to/captures python bench/drift_ablation.py
Writes bench/results/drift_ablation.json and plan drawings bench/results/drift_<capture>_{off,on}.svg.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cozmo.export.document import build_document  # noqa: E402
from cozmo.export.render import render_svg  # noqa: E402
from cozmo.geometry.planes import find_floors  # noqa: E402
from cozmo.geometry.rooms import build_room_map, room_ceilings  # noqa: E402
from cozmo.geometry.fusion import fuse  # noqa: E402
from cozmo.geometry.walls import attach_doorways, outline_rooms, wall_mask  # noqa: E402
from cozmo.ingest.stray import StrayCapture  # noqa: E402
from cozmo.slam.drift import estimate_drift  # noqa: E402
from cozmo.slam.matching import match_walls  # noqa: E402

DATA = Path(os.environ.get("COZMO_DATA", ROOT / "data" / "captures"))
OUT = ROOT / "bench" / "results"
CAPTURES = ["c00a170fe1", "1a8384c3f6", "c7d28f72c6"]
SAME_FLAT = ("c7d28f72c6", "1a8384c3f6")


def plan(capture_id, points, positions, drift_summary):
    floor = find_floors(points)[0]
    room_map = build_room_map(points, floor, positions)
    outlines = outline_rooms(room_map, points, floor)
    ceilings = room_ceilings(points, floor, room_map)
    openings = attach_doorways(outlines, room_map)
    info = {"id": capture_id, "tier": "lidar", "device": None, "input_path": str(DATA / capture_id),
            "pipeline_version": "bench"}
    doc, _ = build_document(info, floor, room_map, outlines, ceilings, openings, 0.0, drift=drift_summary)
    walls = points.subset(wall_mask(points, floor))
    xz, keep = np.unique(np.floor(walls.xyz[:, [0, 2]] / 0.02).astype(np.int64), axis=0, return_index=True)
    wall_xz = walls.xyz[np.sort(keep)][:, [0, 2]].astype(float)
    wall_n = walls.normal[np.sort(keep)][:, [0, 2]].astype(float)
    wall_n /= np.linalg.norm(wall_n, axis=1, keepdims=True) + 1e-9
    summary = {
        "rooms": len(doc["rooms"]),
        "room_areas_m2": sorted((r["floor_area_m2"]["value"] for r in doc["rooms"]), reverse=True),
        "footprint_m2": doc["plan"]["footprint_area_m2"]["value"],
        "doorways": len(room_map.doorways),
        "walls_per_room": [len(r["walls"]) for r in doc["rooms"]],
        "wall_map_area_m2": round(len(xz) * 0.02 ** 2, 2),
    }
    yaw = next(iter(outlines.values())).yaw_deg if outlines else 0.0
    return summary, render_svg(doc, yaw), (wall_xz, wall_n)


def repeatability(a, b):
    """Two walks of the same flat: fit b onto a once, then measure how far the walls of each sit from
    the other's. Symmetric, so a sharper map is not penalised: with one-way distances, doubled (drifted)
    walls in the target gave the source two chances to match, and sharpening the target looked worse."""
    (axz, an), (bxz, _) = a, b
    m = match_walls(axz, an, bxz, yaw_range_deg=(-180, 179), yaw_step_deg=1.0, max_shift_m=None,
                    cell_m=0.05, center=True)
    moved = m.apply(bxz)
    d_ba, _ = cKDTree(axz).query(moved)
    d_ab, _ = cKDTree(moved).query(axz)
    points = np.vstack([moved, axz])
    d = np.concatenate([d_ba, d_ab])
    tiles: dict[tuple, list] = {}
    for p, di in zip(points, d):
        if di < 0.15:
            tiles.setdefault(tuple(np.floor(p).astype(int)), []).append(di)
    tile_medians = np.array([np.median(v) for v in tiles.values() if len(v) > 30])
    return {
        "fit_yaw_deg": round(m.yaw_deg, 2),
        "within_2cm": round(float(np.mean(d < 0.02)), 3),
        "within_5cm": round(float(np.mean(d < 0.05)), 3),
        "matched_median_cm": round(100 * float(np.median(d[d < 0.10])), 2),
        "tile_median_p50_cm": round(100 * float(np.median(tile_medians)), 2),
        "tile_median_p90_cm": round(100 * float(np.percentile(tile_medians, 90)), 2),
        "tile_median_max_cm": round(100 * float(tile_medians.max()), 2),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results, maps = {}, {}
    for cid in CAPTURES:
        if not (DATA / cid).is_dir():
            print(f"skip {cid}: not found under {DATA}")
            continue
        t0 = time.time()
        capture = StrayCapture(DATA / cid)
        points = fuse(capture)
        off, svg_off, map_off = plan(cid, points, capture.positions, None)
        correction, report = estimate_drift(capture, points)
        fixed = correction.apply(points)
        del points
        gc.collect()
        on, svg_on, map_on = plan(cid, fixed, correction.positions[correction.valid], report.to_schema())
        del fixed
        gc.collect()
        (OUT / f"drift_{cid}_off.svg").write_text(svg_off)
        (OUT / f"drift_{cid}_on.svg").write_text(svg_on)
        maps[cid] = {"off": map_off, "on": map_on}
        results[cid] = {"off": off, "on": on, "drift": report.__dict__, "seconds": round(time.time() - t0)}
        print(cid, json.dumps({"off": off, "on": on}, default=float))
        print("   drift:", json.dumps(report.__dict__, default=float))

    comparison = {}
    if all(c in maps for c in SAME_FLAT):
        for mode in ("off", "on"):
            comparison[mode] = repeatability(maps[SAME_FLAT[0]][mode], maps[SAME_FLAT[1]][mode])
            print("same flat, drift", mode, comparison[mode])

    commit = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    report = {"benchmark": "drift_ablation", "code_commit": commit, "captures": results,
              "same_flat_repeatability": {"walks": list(SAME_FLAT), **comparison}}
    (OUT / "drift_ablation.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print("wrote", (OUT / "drift_ablation.json").relative_to(ROOT))


if __name__ == "__main__":
    main()
