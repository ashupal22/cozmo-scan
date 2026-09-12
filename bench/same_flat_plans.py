"""Same flat, same plan? Two walks of one flat compared room by room (brief: repeatability gate).

Both walks go through the LiDAR pipeline (drift correction on). The second walk's plan is placed onto the
first by one rigid fit of their wall maps; rooms are paired when they overlap by IoU >= 0.5. Reported for
the old floor-first outlines and the walls-first layout:
  - rooms per walk, footprint per walk and the difference
  - paired rooms: IoU, area of each, area difference, and the difference of their two main dimensions
  - share of each walk's floor area that found a partner

    COZMO_DATA=/path/to/captures python bench/same_flat_plans.py
Writes bench/results/same_flat_plans.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cozmo.geometry.fusion import fuse  # noqa: E402
from cozmo.geometry.layout import build_layout  # noqa: E402
from cozmo.geometry.planes import find_floors  # noqa: E402
from cozmo.geometry.rooms import build_room_map  # noqa: E402
from cozmo.geometry.walls import _to_aligned, outline_rooms, wall_mask  # noqa: E402
from cozmo.ingest.stray import StrayCapture  # noqa: E402
from cozmo.slam.drift import estimate_drift  # noqa: E402
from cozmo.slam.matching import match_walls  # noqa: E402

DATA = Path(os.environ.get("COZMO_DATA", ROOT / "data" / "captures"))
OUT = ROOT / "bench" / "results" / "same_flat_plans.json"
WALKS = ("c7d28f72c6", "1a8384c3f6")
MIN_IOU = 0.5


def code_commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                          capture_output=True, text=True).stdout.strip()


def walk_plans(cid):
    capture = StrayCapture(DATA / cid)
    points = fuse(capture)
    correction, _ = estimate_drift(capture, points)
    points = correction.apply(points)
    positions = correction.positions[correction.valid]
    floor = find_floors(points)[0]
    walls = points.subset(wall_mask(points, floor))
    _, keep = np.unique(np.floor(walls.xyz[:, [0, 2]] / 0.02).astype(np.int64), axis=0, return_index=True)
    wall_xz = walls.xyz[np.sort(keep)][:, [0, 2]].astype(float)
    wall_n = walls.normal[np.sort(keep)][:, [0, 2]].astype(float)
    wall_n /= np.linalg.norm(wall_n, axis=1, keepdims=True) + 1e-9
    floor_first = outline_rooms(build_room_map(points, floor, positions), points, floor)
    layout = build_layout(points, floor, positions)
    return {"floor_first": floor_first, "walls_first": layout.outlines}, (wall_xz, wall_n), layout.yaw_deg


def main_dimensions(poly: Polygon, yaw_deg: float) -> np.ndarray:
    xz = np.array(poly.exterior.coords)[:-1] @ _to_aligned(yaw_deg).T
    return np.sort(np.ptp(xz, axis=0))


def compare(rooms_a, rooms_b, fit, yaw_a) -> dict:
    theta = np.radians(fit.yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    # match_walls: target = rot2(yaw) source + shift, rot2 = [[c, s], [-s, c]]
    params = [c, s, -s, c, float(fit.shift[0]), float(fit.shift[1])]
    polys_a = {r: Polygon(o.vertices) for r, o in rooms_a.items()}
    polys_b = {r: affine_transform(Polygon(o.vertices), params) for r, o in rooms_b.items()}
    pairs, used = [], set()
    for ra, pa in sorted(polys_a.items(), key=lambda kv: -kv[1].area):
        best = None
        for rb, pb in polys_b.items():
            if rb in used or not pa.intersects(pb):
                continue
            iou = pa.intersection(pb).area / pa.union(pb).area
            if best is None or iou > best[0]:
                best = (iou, rb)
        if best and best[0] >= MIN_IOU:
            used.add(best[1])
            pb = polys_b[best[1]]
            dims_a, dims_b = main_dimensions(pa, yaw_a), main_dimensions(pb, yaw_a)
            pairs.append({"room_a": ra, "room_b": best[1], "iou": round(best[0], 3),
                          "area_a": round(pa.area, 2), "area_b": round(pb.area, 2),
                          "area_diff_m2": round(abs(pa.area - pb.area), 2),
                          "dimension_diffs_cm": [round(100 * float(d), 1) for d in np.abs(dims_a - dims_b)]})
    area_a = sum(p.area for p in polys_a.values())
    area_b = sum(p.area for p in polys_b.values())
    dims = [d for p in pairs for d in p["dimension_diffs_cm"]]
    return {
        "rooms": [len(polys_a), len(polys_b)],
        "footprint_m2": [round(area_a, 2), round(area_b, 2)],
        "footprint_diff_pct": round(100 * abs(area_a - area_b) / max(area_a, area_b), 1),
        "paired_rooms": len(pairs),
        "paired_area_share": [round(sum(p["area_a"] for p in pairs) / area_a, 3),
                              round(sum(p["area_b"] for p in pairs) / area_b, 3)],
        "median_iou": round(float(np.median([p["iou"] for p in pairs])), 3) if pairs else None,
        "dimension_diff_cm_median": round(float(np.median(dims)), 1) if dims else None,
        "dimensions_within_1cm_or_0.5pct": (f"{sum(d <= 1.0 for d in dims)}/{len(dims)}" if dims else "0/0"),
        "pairs": pairs,
    }


def main():
    commit = code_commit()
    a_id, b_id = WALKS
    plans_a, walls_a, yaw_a = walk_plans(a_id)
    plans_b, walls_b, _ = walk_plans(b_id)
    fit = match_walls(walls_a[0], walls_a[1], walls_b[0], yaw_range_deg=(-180, 179), yaw_step_deg=1.0,
                      max_shift_m=None, cell_m=0.05, center=True)
    report = {"benchmark": "same_flat_plans", "code_commit": commit, "walks": list(WALKS),
              "fit": {"yaw_deg": round(fit.yaw_deg, 2), "overlap_median_cm": round(100 * fit.overlap_median_m, 2)}}
    for method in ("floor_first", "walls_first"):
        report[method] = compare(plans_a[method], plans_b[method], fit, yaw_a)
        summary = {k: v for k, v in report[method].items() if k != "pairs"}
        print(method, json.dumps(summary))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
