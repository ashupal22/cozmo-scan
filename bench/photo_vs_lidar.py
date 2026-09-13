"""Photo tier against LiDAR of the same walk: wall lengths (G-WALL-PHOTO), stitch, and interval calibration.

We have no iPhone photo sets with tape truth. Stand-in sets are cut from our walks by bench/make_photo_sets.py
(--sweep: overlapping views turning in place, as the capture protocol asks), one folder per LiDAR room. Each
photo room is compared with the LiDAR plan of the same walk. The stills are video frames and carry no EXIF, so
the true focal length is passed in to stand for the EXIF value every iPhone photo carries.

Per room: the photo box's two dimensions against the LiDAR room's extents along its own axes (both sorted, so no
alignment is needed), floor area, and whether the 90% intervals hold. The split-conformal widening factor for
PHOTO_INTERVAL_SCALE is computed from the dimension errors.

    COZMO_DATA=/path/to/captures python bench/photo_vs_lidar.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from video_vs_lidar import DATA, DERIVED, code_commit  # noqa: E402

from cozmo.export.document import PHOTO_INTERVAL_SCALE, Z90  # noqa: E402
from cozmo.geometry.walls import _to_aligned  # noqa: E402
from cozmo.ingest.stray import StrayCapture, upright_rotate_code  # noqa: E402
from cozmo.pipeline import run_lidar, run_photos  # noqa: E402

OUT = ROOT / "bench" / "results" / "photo_vs_lidar.json"
WALKS = ("c00a170fe1", "1a8384c3f6")   # c7d28f72c6 has no photo set: its LiDAR plan is not reliable enough to cut rooms from


def true_focal(cid: str) -> float:
    cap = StrayCapture(DATA / cid)
    codes = [upright_rotate_code(cap.rotation(i)) for i in range(0, len(cap), 30)]
    code = max(set(codes), key=codes.count)
    k = cap.intrinsics(0, "rgb")
    sideways = code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return k.fy / cap.rgb_size[1] if sideways else k.fx / cap.rgb_size[0]


def lidar_room(outlines: dict, rid: int, area_m2: float):
    """The LiDAR room the photo set was cut from: its id when the area still matches, else the closest area."""
    if rid in outlines and abs(outlines[rid].area_m2 / area_m2 - 1) < 0.05:
        return outlines[rid]
    return min(outlines.values(), key=lambda o: abs(o.area_m2 - area_m2))


def sigma_of(m: dict) -> float:
    """The model sigma before widening, recovered from the interval (log-scale intervals: see measurement())."""
    v, lo, hi = m["value"], m["ci_low"], m["ci_high"]
    if lo > 0 and abs((hi - v) - (v - lo)) > 0.002:
        return v * ((hi / v) ** (1 / Z90) - 1) / PHOTO_INTERVAL_SCALE
    return (hi - v) / Z90 / PHOTO_INTERVAL_SCALE


def run_walk(cid: str) -> dict:
    photos = DERIVED / f"{cid}_photo_sweeps"
    manifest = json.loads((photos / "manifest.json").read_text())
    lidar = run_lidar(DATA / cid)
    plan = run_photos(photos, DERIVED / f"{cid}_photo_work", fx_over_width=true_focal(cid))
    doc = plan.document
    rooms, dims = [], []
    for room, r in zip(doc["rooms"], plan.photo_rooms):
        info = manifest["rooms"][r.name]
        lid = lidar_room(lidar.outlines, int(info["lidar_room"][1:]), info["area_m2"])
        extent = np.sort(np.ptp(np.asarray(lid.vertices) @ _to_aligned(lid.yaw_deg).T, axis=0))
        walls = sorted((room["walls"][0]["length_m"], room["walls"][1]["length_m"]), key=lambda m: m["value"])
        pairs = []
        for m, truth in zip(walls, extent):
            pairs.append({"photo": m["value"], "lidar": round(float(truth), 3), "interval": [m["ci_low"], m["ci_high"]],
                          "error_pct": round(100 * (m["value"] / truth - 1), 1),
                          "holds": bool(m["ci_low"] <= truth <= m["ci_high"]),
                          "z": round(abs(m["value"] - truth) / sigma_of(m), 2)})
        dims += pairs
        area = room["floor_area_m2"]
        rooms.append({"room": r.name, "photos": r.photos, "sides_seen": r.sides_seen, "focal": r.focal_source,
                      "lidar_room_area_m2": round(lid.area_m2, 2), "photo_area_m2": area["value"],
                      "area_interval": [area["ci_low"], area["ci_high"]],
                      "area_error_pct": round(100 * (area["value"] / lid.area_m2 - 1), 1),
                      "doors": len(r.doors), "dimensions": pairs})
        print(cid, r.name, json.dumps(rooms[-1]), flush=True)
    truth_area = sum(r["lidar_room_area_m2"] for r in rooms)
    fp = doc["plan"]["footprint_area_m2"]
    return {"rooms": rooms, "dimensions_within_8pct": f"{sum(abs(d['error_pct']) <= 8 for d in dims)}/{len(dims)}",
            "dimension_abs_error_pct_median": round(float(np.median([abs(d["error_pct"]) for d in dims])), 1),
            "intervals_hold": f"{sum(d['holds'] for d in dims)}/{len(dims)}",
            "footprint": {"photo": fp["value"], "interval": [fp["ci_low"], fp["ci_high"]], "lidar_rooms_with_photos": round(truth_area, 2),
                          "error_pct": round(100 * (fp["value"] / truth_area - 1), 1)},
            "stitch": doc["plan"]["stitch"], "adjacency": doc["plan"]["adjacency"],
            "seconds": doc["capture"]["runtime_s"], "z": [d["z"] for d in dims]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    report = {"benchmark": "photo_vs_lidar", "code_commit": code_commit(), "photo_interval_scale": PHOTO_INTERVAL_SCALE,
              "walks": {cid: run_walk(cid) for cid in args.walks}}
    z = np.sort([v for w in report["walks"].values() for v in w["z"]])
    n = len(z)
    in_sample = int(np.ceil(0.9 * n))                 # the rule VIDEO_INTERVAL_SCALE used (9 walls: the largest)
    conformal = min(int(np.ceil(0.9 * (n + 1))), n)   # split-conformal, finite-sample corrected
    report["calibration"] = {"dimensions": n, "z_sorted": [round(float(v), 2) for v in z],
                             "scale_in_sample_90": round(float(z[in_sample - 1] / Z90), 2),
                             "scale_split_conformal_90": round(float(z[conformal - 1] / Z90), 2),
                             "note": "z = |error| / model sigma before widening. scale_in_sample_90: smallest factor "
                                     "with the nominal 90% interval holding on at least 90% of these dimensions "
                                     "(used for PHOTO_INTERVAL_SCALE). With 18 dimensions the split-conformal rule "
                                     "needs the single worst one."}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["calibration"]))
    for cid, w in report["walks"].items():
        print(cid, {k: w[k] for k in ("dimensions_within_8pct", "dimension_abs_error_pct_median", "intervals_hold", "footprint")},
              w["stitch"]["notes"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
