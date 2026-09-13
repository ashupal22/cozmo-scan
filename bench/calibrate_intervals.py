"""Calibrate the video tier's 90% intervals on the errors it actually makes (brief: calibration is scored at
every tier; confident garbage on thin input caps the score).

Input: bench/results/video_vs_lidar.json (run bench/video_vs_lidar.py first). For every paired wall the error
|video - LiDAR| is divided by the video wall's own sigma, the half-width of its 90% interval / 1.645. All paired
walls count, not only the gate walls (LiDAR walls of at least 1 m whose two neighbouring faces were fitted to
wall points): on our walks only 5 gate walls paired, too few to calibrate on. Walls with a weaker LiDAR reference
make the factor more conservative, not less. A video wall matched to two parallel LiDAR walls counts once.
Split-conformal calibration: with n such ratios, the ceil((n+1) x 0.9)-th smallest, divided by 1.645, is the
factor by which every video interval must widen so that 90% of intervals hold the truth. LiDAR's own error
(1-2 cm) is counted as video error, which makes the factor slightly conservative.

Checked leave-one-walk-out: the factor from the other walks, applied to the held-out walk's walls. Missed
walls (no video partner) have no error to calibrate on; they are detection misses, reported by the benchmark.
Caveat: three walks of two flats is a small calibration set, and the walls of one walk are not independent.

    python bench/calibrate_intervals.py [--variant video]
Writes bench/results/video_intervals.json. The pipeline's VIDEO_INTERVAL_SCALE (cozmo/export/document.py) is
set to the factor for all walks, times the scale that was in effect when the benchmark ran.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "video_vs_lidar.json"
OUT = ROOT / "bench" / "results" / "video_intervals.json"
Z90 = 1.645
LEVEL = 0.9
GATE_WALL_M = 1.0


def ratios(walk_result: dict, gate_only: bool = False) -> np.ndarray:
    """|video - LiDAR| / video sigma for the paired walls of one walk (only gate walls with gate_only)."""
    out, seen = [], set()
    for w in walk_result.get("wall_pairs", []):
        if w.get("video_m") is None or (gate_only and (not w["reference_ok"] or w["lidar_m"] < GATE_WALL_M)):
            continue
        key = (w["room"], w["video_m"], tuple(w["video_ci"]))
        if key in seen:
            continue
        seen.add(key)
        sigma = (w["video_ci"][1] - w["video_ci"][0]) / (2 * Z90)
        out.append(abs(w["video_m"] - w["lidar_m"]) / sigma if sigma > 0 else np.inf)
    return np.array(out)


def conformal_factor(z: np.ndarray, level: float = LEVEL) -> float | None:
    """Widening factor for nominal-`level` intervals; None when there are too few errors to say."""
    n = len(z)
    k = int(np.ceil((n + 1) * level))
    if n == 0 or k > n:
        return None
    return float(np.sort(z)[k - 1] / Z90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="video")
    args = ap.parse_args()
    results = json.loads(RESULTS.read_text())
    walks = {cid: r[args.variant] for cid, r in results["walks"].items() if args.variant in r}
    z = {cid: ratios(r) for cid, r in walks.items()}
    in_effect = {cid: r.get("video_interval_scale", 1.0) for cid, r in walks.items()}
    report = {"benchmark": "video_intervals", "source": str(RESULTS.relative_to(ROOT)),
              "source_commit": results.get("code_commit"), "variant": args.variant,
              "walls_per_walk": {cid: int(len(v)) for cid, v in z.items()},
              "gate_walls_per_walk": {cid: int(len(ratios(r, gate_only=True))) for cid, r in walks.items()},
              "error_over_sigma": {cid: [round(float(x), 2) for x in np.sort(v)] for cid, v in z.items()},
              "covered_as_is": {cid: f"{int(np.sum(v <= Z90))}/{len(v)}" for cid, v in z.items()}}
    loo = {}
    for cid in walks:
        train = np.concatenate([z[o] for o in walks if o != cid]) if len(walks) > 1 else np.array([])
        factor = conformal_factor(train)
        held = z[cid]
        loo[cid] = {"factor_from_others": round(factor, 3) if factor else None,
                    "held_out_covered": f"{int(np.sum(held <= Z90 * factor))}/{len(held)}" if factor else None}
    report["leave_one_out"] = loo
    everything = np.concatenate(list(z.values())) if z else np.array([])
    factor = conformal_factor(everything)
    scale_in_effect = float(np.median(list(in_effect.values()))) if in_effect else 1.0
    report["factor_all_walks"] = round(factor, 3) if factor else None
    report["scale_in_effect"] = scale_in_effect
    report["recommended_VIDEO_INTERVAL_SCALE"] = round(factor * scale_in_effect, 2) if factor else None
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
