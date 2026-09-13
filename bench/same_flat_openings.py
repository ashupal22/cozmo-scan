"""Opening widths, walk against walk: a repeatability check for G-OPEN, since we have no tape truth.

Both walks of the same flat go through the shipped LiDAR pipeline (`cozmo run`). The second plan is placed on the
first by one rigid fit of their wall maps. Openings are paired when their centres lie within PAIR_M of each other.
Each doorway appears once per room it joins. For each pair the benchmark gives both widths and whether each was
measured jamb to jamb or reported at a typical width.

    COZMO_DATA=/path/to/captures python bench/same_flat_openings.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from video_vs_lidar import DATA, code_commit, wall_map  # noqa: E402

from cozmo.pipeline import run_lidar  # noqa: E402
from cozmo.slam.matching import match_walls  # noqa: E402

OUT = ROOT / "bench" / "results" / "same_flat_openings.json"
WALKS = ("c7d28f72c6", "1a8384c3f6")
PAIR_M = 0.3


def openings(doc: dict) -> list[dict]:
    out = []
    for room in doc["rooms"]:
        walls = {w["id"]: w for w in room["walls"]}
        for op in room["openings"]:
            w = walls[op["wall_id"]]
            s, e = np.array(w["start"]), np.array(w["end"])
            d = (e - s) / max(np.linalg.norm(e - s), 1e-9)
            out.append({"id": op["id"], "centre": s + d * op["offset_along_wall_m"]["value"],
                        "width": op["width_m"]["value"], "interval": [op["width_m"]["ci_low"], op["width_m"]["ci_high"]],
                        "measured": op["width_m"].get("observed", True)})
    return out


def main():
    a, b = (run_lidar(DATA / cid) for cid in WALKS)
    wa, na = wall_map(a)
    wb, _ = wall_map(b)
    fit = match_walls(wa, na, wb, yaw_range_deg=(-180, 179), yaw_step_deg=1.0, max_shift_m=None, cell_m=0.05, center=True)
    oa, ob = openings(a.document), openings(b.document)
    centres_b = np.array([fit.apply(o["centre"][None])[0] for o in ob]) if ob else np.zeros((0, 2))
    pairs = []
    for o in oa:
        if not len(centres_b):
            break
        dist = np.linalg.norm(centres_b - o["centre"], axis=1)
        j = int(np.argmin(dist))
        if dist[j] <= PAIR_M:
            q = ob[j]
            pairs.append({"a": o["id"], "b": q["id"], "centre_gap_m": round(float(dist[j]), 3), "width_a": o["width"],
                          "width_b": q["width"], "difference_cm": round(100 * (o["width"] - q["width"]), 1),
                          "both_measured": bool(o["measured"] and q["measured"]),
                          "intervals_overlap": bool(o["interval"][0] <= q["interval"][1] and q["interval"][0] <= o["interval"][1])})
    measured = [p for p in pairs if p["both_measured"]]
    diff = np.abs([p["difference_cm"] for p in measured])
    report = {"benchmark": "same_flat_openings", "code_commit": code_commit(), "walks": list(WALKS),
              "openings": {WALKS[0]: len(oa), WALKS[1]: len(ob)}, "paired": len(pairs),
              "both_measured": len(measured),
              "abs_difference_cm_median": round(float(np.median(diff)), 1) if len(diff) else None,
              "within_2cm": f"{int((diff <= 2).sum())}/{len(diff)}",
              "intervals_overlap": f"{sum(p['intervals_overlap'] for p in pairs)}/{len(pairs)}",
              "pairs": pairs}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}))
    for p in pairs:
        print(p)


if __name__ == "__main__":
    main()
