"""Benchmark: floor and ceiling heights from device LiDAR against laser ground truth (ARKitScenes).

For each walk it reports:
  pipeline    what cozmo picks on its own: the most supported floor and ceiling (device depth, keyframes)
  truth       the same planes found in laser-rendered depth (highres_depth frames)
  matched     device points re-fitted around the laser plane heights. This separates
              "picked the wrong surface" from "measured the right surface badly"
  depth bias  device minus laser depth per pixel, by distance
and, per venue, the spread of ceiling heights across walks (the repeatability gate).

Gates (docs/gates.md): G-CEIL error <= 15 mm; G-CEIL-SPREAD spread <= 10 mm.

    python bench/arkitscenes_planes.py                 # all walks in data/external/arkitscenes
    python bench/arkitscenes_planes.py --walks 41069048
    python bench/arkitscenes_planes.py --bias-correction leave-one-venue-out
Writes bench/results/arkitscenes_planes.json, or arkitscenes_planes_bias_corrected.json with the correction:
every walk's device depth gets the offset measured on the other venue's walks (median device - laser depth
between 0.3 and 2 m), so no walk is corrected with its own ground truth.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cozmo.geometry.fusion import PointSet, fuse  # noqa: E402
from cozmo.geometry.planes import HORIZONTAL_DOT, find_ceilings, find_floors, robust_height  # noqa: E402
from cozmo.ingest.arkitscenes import ARKitScenesWalk  # noqa: E402

DATA = ROOT / "data" / "external" / "arkitscenes"
OUT = ROOT / "bench" / "results" / "arkitscenes_planes.json"
RANGES_M = [(0.3, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.5)]
CEIL_GATE_MM, SPREAD_GATE_MM = 15.0, 10.0


def depth_bias_mm(walk: ARKitScenesWalk, every: int = 5) -> dict:
    diffs = {r: [] for r in RANGES_M}
    for i in walk.gt_indices[::every]:
        i = int(i)
        d, c = walk.depth(i), walk.confidence(i)
        g = cv2.resize(walk.gt_depth(i), walk.depth_size, interpolation=cv2.INTER_NEAREST)
        ok = (c == 2) & (d > 0) & (g > 0)
        for lo, hi in RANGES_M:
            m = ok & (g >= lo) & (g < hi)
            if m.sum() > 100:
                diffs[(lo, hi)].append(float(np.median(d[m] - g[m])))
    return {f"{lo}-{hi} m": (round(1000 * float(np.median(v)), 1) if v else None) for (lo, hi), v in diffs.items()}


def device_bias_m(walk: ARKitScenesWalk, every: int = 5, near: float = 0.3, far: float = 2.0) -> float | None:
    """Median over frames of the median device - laser depth, for laser depth between near and far."""
    per_frame = []
    for i in walk.gt_indices[::every]:
        i = int(i)
        d, c = walk.depth(i), walk.confidence(i)
        g = cv2.resize(walk.gt_depth(i), walk.depth_size, interpolation=cv2.INTER_NEAREST)
        m = (c == 2) & (d > 0) & (g >= near) & (g < far)
        if m.sum() > 100:
            per_frame.append(float(np.median(d[m] - g[m])))
    return float(np.median(per_frame)) if per_frame else None


def refit(points: PointSet, facing: int, height: float):
    """Device plane fitted only around a known height; facing +1 = up (floor), -1 = down (ceiling)."""
    mask = points.normal[:, 1] * facing > HORIZONTAL_DOT
    return robust_height(points.xyz[mask, 1], height, window=0.05)


def evaluate(walk: ARKitScenesWalk, depth_offset_m: float = 0.0) -> dict:
    device = fuse(walk, depth_offset_m=depth_offset_m)
    laser = fuse(walk, walk.gt_indices, ground_truth=True, stride=8, max_depth=6.0)
    result = {"frames": len(walk), "gt_frames": int(len(walk.gt_indices)), "depth_offset_mm": round(1000 * depth_offset_m, 1)}

    true_floors = find_floors(laser)
    if not true_floors:
        result["error"] = "no floor in ground truth"
        return result
    tf = true_floors[0]
    true_ceilings = find_ceilings(laser, tf)
    true_height = true_ceilings[0].height - tf.height if true_ceilings else None
    result["truth"] = {"ceiling_height_m": true_height}

    floors = find_floors(device)
    if floors:
        pf = floors[0]
        ceilings = find_ceilings(device, pf)
        height = ceilings[0].height - pf.height if ceilings else None
        result["pipeline"] = {
            "floor_error_mm": round(1000 * (pf.height - tf.height), 1),
            "ceiling_height_m": height,
            "ceiling_error_mm": round(1000 * (height - true_height), 1) if height and true_height else None,
        }

    mf = refit(device, +1, tf.height)
    mc = refit(device, -1, true_ceilings[0].height) if true_ceilings else None
    if mf and mc:
        result["matched"] = {
            "floor_error_mm": round(1000 * (mf.height - tf.height), 1),
            "ceiling_plane_error_mm": round(1000 * (mc.height - true_ceilings[0].height), 1),
            "ceiling_error_mm": round(1000 * ((mc.height - mf.height) - true_height), 1),
        }
    result["depth_bias_mm"] = depth_bias_mm(walk)
    return result


def summarize(walks: dict, visits: dict) -> dict:
    summary = {}
    for kind in ("pipeline", "matched"):
        errors = [w[kind]["ceiling_error_mm"] for w in walks.values() if w.get(kind, {}).get("ceiling_error_mm") is not None]
        per_visit = {}
        for visit, ids in visits.items():
            heights = [walks[v][kind]["ceiling_height_m"] if kind == "pipeline" else
                       walks[v]["truth"]["ceiling_height_m"] + walks[v][kind]["ceiling_error_mm"] / 1000
                       for v in ids if walks[v].get(kind, {}).get("ceiling_error_mm") is not None]
            if len(heights) >= 2:
                per_visit[str(visit)] = round(1000 * (max(heights) - min(heights)), 1)
        summary[kind] = {
            "ceiling_errors_mm": errors,
            "mean_error_mm": round(float(np.mean(errors)), 1) if errors else None,
            "walks_within_gate": f"{sum(abs(e) <= CEIL_GATE_MM for e in errors)}/{len(errors)}",
            "spread_across_walks_mm": per_visit,
            "venues_within_spread_gate": f"{sum(s <= SPREAD_GATE_MM for s in per_visit.values())}/{len(per_visit)}",
        }
    return summary


def code_commit() -> str:
    """Read before any output is written: a run's own tracked result files would otherwise mark the
    tree "-dirty". "-dirty" means uncommitted code changes at the start of the run."""
    return subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--walks", nargs="+", help="video ids (default: all in manifest.json)")
    parser.add_argument("--bias-correction", choices=["none", "leave-one-venue-out"], default="none",
                        help="add the depth offset measured on the other venue's walks (default: depth as recorded)")
    args = parser.parse_args()

    commit = code_commit()
    manifest = json.loads((DATA / "manifest.json").read_text())
    chosen = [m for m in manifest if not args.walks or m["video_id"] in args.walks]
    walks, visits, offsets, bias = {}, {}, {}, {}
    if args.bias_correction == "leave-one-venue-out":
        for m in chosen:
            bias[m["video_id"]] = (m["visit_id"], device_bias_m(ARKitScenesWalk(DATA / "Validation" / m["video_id"])))
        for m in chosen:
            others = [b for visit, b in bias.values() if visit != m["visit_id"] and b is not None]
            offsets[m["video_id"]] = -float(np.mean(others)) if others else 0.0
    for m in chosen:
        t0 = time.time()
        walk = ARKitScenesWalk(DATA / "Validation" / m["video_id"])
        walks[m["video_id"]] = {"visit_id": m["visit_id"], **evaluate(walk, offsets.get(m["video_id"], 0.0)),
                                "seconds": round(time.time() - t0)}
        visits.setdefault(m["visit_id"], []).append(m["video_id"])
        print(m["video_id"], json.dumps(walks[m["video_id"]]))

    report = {"benchmark": "arkitscenes_planes", "code_commit": commit, "bias_correction": args.bias_correction,
              "gates_mm": {"ceiling_error": CEIL_GATE_MM, "ceiling_spread": SPREAD_GATE_MM},
              "summary": summarize(walks, visits), "walks": walks}
    if bias:
        report["device_bias_mm"] = {vid: round(1000 * b, 1) if b is not None else None for vid, (_, b) in bias.items()}
        measured = [b for _, b in bias.values() if b is not None]
        report["device_bias_all_walks_mm"] = round(1000 * float(np.mean(measured)), 1) if measured else None
    out = OUT if args.bias_correction == "none" else OUT.with_name("arkitscenes_planes_bias_corrected.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
