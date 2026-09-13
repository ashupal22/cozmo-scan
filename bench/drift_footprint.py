"""G-DRIFT with the shipped pipeline: the stitched footprint with drift correction on and off (`cozmo run` and
`cozmo run --no-drift`), on each walk, and how far the two walks of the same flat disagree either way.
(bench/drift_ablation.py measures map sharpness and wall agreement; it predates the walls-first layout.)

    COZMO_DATA=/path/to/captures python bench/drift_footprint.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from video_vs_lidar import DATA, WALKS, code_commit  # noqa: E402

from cozmo.pipeline import run_lidar  # noqa: E402

OUT = ROOT / "bench" / "results" / "drift_footprint.json"


def main():
    report = {"benchmark": "drift_footprint", "code_commit": code_commit(), "walks": {}}
    for cid in WALKS:
        row = {}
        for name, drift in (("off", False), ("on", True)):
            t0 = time.time()
            doc = run_lidar(DATA / cid, drift=drift).document
            fp = doc["plan"]["footprint_area_m2"]
            row[name] = {"rooms": len(doc["rooms"]), "footprint_m2": fp["value"], "interval": [fp["ci_low"], fp["ci_high"]],
                         "adjacencies": len(doc["plan"]["adjacency"]), "seconds": round(time.time() - t0, 1),
                         "drift": doc["plan"]["drift"]}
        report["walks"][cid] = row
        print(cid, {k: {x: v[x] for x in ("rooms", "footprint_m2", "adjacencies")} for k, v in row.items()}, flush=True)
    same = {}
    for name in ("off", "on"):
        a, b = (report["walks"][c][name]["footprint_m2"] for c in ("c7d28f72c6", "1a8384c3f6"))
        same[name] = {"c7d28f72c6": a, "1a8384c3f6": b, "difference_pct": round(100 * abs(a - b) / ((a + b) / 2), 1)}
    report["same_flat"] = same
    print("same flat", same)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
