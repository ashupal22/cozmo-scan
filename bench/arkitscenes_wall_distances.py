"""LiDAR wall-to-wall distances against laser ground truth (A-WALL-LIDAR), on ARKitScenes.

The walls-first layout cannot run on these scans: the phone stays within about 1 m, and rooms are grown from where the
phone walked (docs/technical_report.md, failure modes). This measures what the layout is built on instead: where the
depth sensor puts the walls. On each walk, device LiDAR depth and laser-rendered depth are fused over the same frames
with the same poses, so pose errors cancel and only the sensor and fusion are tested.

Wall planes: points facing within 12 degrees of one of the room's two wall directions, 0.3-2.0 m above the floor, in
1 cm position bins. A plane needs 0.5 m of coverage along the wall and 0.8 m of height (furniture fronts are lower).
Two planes facing each other across the room give a wall-to-wall distance, what a laser measurer reads. Each laser
distance is compared with the distance between the matching device planes (each within 10 cm of its laser plane),
with device depth as recorded and with the shipped +11.9 mm correction.

    python bench/arkitscenes_wall_distances.py      # needs: python scripts/fetch_external.py arkitscenes; ~10 min
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from video_vs_lidar import code_commit  # noqa: E402

from cozmo.geometry.fusion import fuse  # noqa: E402
from cozmo.geometry.planes import find_floors  # noqa: E402
from cozmo.geometry.walls import _to_aligned, dominant_yaw  # noqa: E402
from cozmo.ingest.arkitscenes import ARKitScenesWalk  # noqa: E402
from cozmo.pipeline import LIDAR_DEPTH_OFFSET_M  # noqa: E402

DATA = ROOT / "data" / "external" / "arkitscenes"
OUT = ROOT / "bench" / "results" / "arkitscenes_wall_distances.json"
FACING_COS = float(np.cos(np.radians(12)))
BAND_M = (0.3, 2.0)
BIN_M = 0.01
PLANE_HALF_M = 0.02
SUPPRESS_BINS = 10
MIN_COVER_M, MIN_SPAN_M = 0.5, 0.8
MATCH_M = 0.10
MIN_DISTANCE_M = 0.8
MIN_OVERLAP_M = 0.3


def wall_points(points, floor):
    h = points.xyz[:, 1] - floor.height
    keep = (np.abs(points.normal[:, 1]) < 0.2) & (h > BAND_M[0]) & (h < BAND_M[1])
    n = points.normal[keep][:, [0, 2]].astype(float)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return points.xyz[keep][:, [0, 2]].astype(float), n, h[keep]


def find_planes(xz, nxz, h, R) -> list[dict]:
    uv, nuv = xz @ R.T, nxz @ R.T
    found = []
    for axis in (0, 1):
        for sign in (1, -1):
            m = nuv[:, axis] * sign > FACING_COS
            pos, along, hh = uv[m, axis], uv[m, 1 - axis], h[m]
            if len(pos) < 100:
                continue
            edges = np.arange(np.floor(pos.min() / BIN_M) * BIN_M, pos.max() + 2 * BIN_M, BIN_M)
            hist, _ = np.histogram(pos, edges)
            smooth = np.convolve(hist, np.ones(3), mode="same")
            taken = np.zeros(len(smooth), bool)
            for k in np.argsort(-smooth):
                if smooth[k] < 30:
                    break
                if taken[max(k - SUPPRESS_BINS, 0):k + SUPPRESS_BINS + 1].any():
                    continue
                taken[k] = True
                near = np.abs(pos - (edges[k] + BIN_M / 2)) < PLANE_HALF_M
                if near.sum() < 50:
                    continue
                a = along[near]
                cover = len(np.unique(np.floor(a / 0.05))) * 0.05
                span = float(np.percentile(hh[near], 95) - np.percentile(hh[near], 5))
                if cover >= MIN_COVER_M and span >= MIN_SPAN_M:
                    found.append({"axis": axis, "sign": sign, "pos": float(np.median(pos[near])),
                                  "along": [float(np.percentile(a, 5)), float(np.percentile(a, 95))],
                                  "cover_m": round(cover, 2), "span_m": round(span, 2), "points": int(near.sum())})
    return found


def match(plane: dict, others: list[dict]) -> dict | None:
    same = [o for o in others if o["axis"] == plane["axis"] and o["sign"] == plane["sign"]
            and abs(o["pos"] - plane["pos"]) <= MATCH_M]
    return min(same, key=lambda o: abs(o["pos"] - plane["pos"])) if same else None


def facing_pairs(planes: list[dict]) -> list[tuple[dict, dict, float]]:
    """For each plane facing +axis, the nearest plane facing back at it across open space."""
    pairs = []
    for p in planes:
        if p["sign"] != 1:
            continue
        best = None
        for q in planes:
            if q["axis"] != p["axis"] or q["sign"] != -1:
                continue
            d = q["pos"] - p["pos"]
            overlap = min(p["along"][1], q["along"][1]) - max(p["along"][0], q["along"][0])
            if d >= MIN_DISTANCE_M and overlap >= MIN_OVERLAP_M and (best is None or d < best[2]):
                best = (p, q, d)
        if best:
            pairs.append(best)
    return pairs


def run_walk(vid: str) -> dict:
    walk = ARKitScenesWalk(DATA / "Validation" / vid)
    frames = walk.gt_indices
    t0 = time.time()
    laser = fuse(walk, frames=frames, ground_truth=True, stride=8)
    floor = find_floors(laser)[0]
    lx, ln, lh = wall_points(laser, floor)
    yaw, share = dominant_yaw(ln)
    R = _to_aligned(yaw)
    lp = find_planes(lx, ln, lh, R)
    pairs = facing_pairs(lp)
    result = {"frames": int(len(frames)), "wall_yaw_deg": round(yaw, 2), "manhattan_share": round(share, 2),
              "laser_planes": len(lp), "laser_distances": len(pairs)}
    for name, offset in (("recorded", 0.0), ("corrected", LIDAR_DEPTH_OFFSET_M)):
        device = fuse(walk, frames=frames, depth_offset_m=offset)
        dx, dn, dh = wall_points(device, floor)
        dp = find_planes(dx, dn, dh, R)
        offsets = [1000 * (m["pos"] - p["pos"]) * p["sign"] for p in lp if (m := match(p, dp))]
        rows = []
        for p, q, d in pairs:
            mp, mq = match(p, dp), match(q, dp)
            if mp and mq:
                dev = mq["pos"] - mp["pos"]
                rows.append({"laser_m": round(d, 3), "device_m": round(dev, 3), "error_cm": round(100 * (dev - d), 2),
                             "within_gate": bool(abs(dev - d) <= max(0.02, 0.01 * d))})
        result[name] = {"planes_matched": f"{len(offsets)}/{len(lp)}",
                        "wall_face_offset_mm_median": round(float(np.median(offsets)), 1) if offsets else None,
                        "distances": rows}
    result["seconds"] = round(time.time() - t0, 1)
    return result


def summary(report: dict, name: str) -> dict:
    rows = [r for w in report["walks"].values() for r in w[name]["distances"]]
    err = np.array([r["error_cm"] for r in rows])
    offs = [w[name]["wall_face_offset_mm_median"] for w in report["walks"].values() if w[name]["wall_face_offset_mm_median"] is not None]
    return {"distances": len(rows), "within_max_2cm_1pct": f"{sum(r['within_gate'] for r in rows)}/{len(rows)}",
            "error_cm_median": round(float(np.median(err)), 2) if len(err) else None,
            "abs_error_cm_median": round(float(np.median(np.abs(err))), 2) if len(err) else None,
            "abs_error_cm_p90": round(float(np.percentile(np.abs(err), 90)), 2) if len(err) else None,
            "wall_face_offset_mm_median_of_walks": round(float(np.median(offs)), 1) if offs else None}


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    report = {"benchmark": "arkitscenes_wall_distances", "code_commit": code_commit(),
              "depth_offset_m": LIDAR_DEPTH_OFFSET_M, "walks": {}}
    for m in manifest:
        vid = m["video_id"]
        report["walks"][vid] = {"visit_id": m["visit_id"], **run_walk(vid)}
        w = report["walks"][vid]
        print(vid, {k: w[k] for k in ("laser_planes", "laser_distances", "seconds")},
              {n: {"matched": w[n]["planes_matched"], "offset_mm": w[n]["wall_face_offset_mm_median"],
                   "errors_cm": [r["error_cm"] for r in w[n]["distances"]]} for n in ("recorded", "corrected")}, flush=True)
    report["summary"] = {n: summary(report, n) for n in ("recorded", "corrected")}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("summary", json.dumps(report["summary"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
