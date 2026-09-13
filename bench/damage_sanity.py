"""Damage detector checks we can run without a damaged room (A-DMG-DETECT needs staged damage; not yet captured).

1. False alarms: every tile of every image of our three undamaged LiDAR walks (one image per second) is scored;
   the report gives the highest damage score and how many tiles pass the threshold P_MIN.
2. Painted stains: a soft brown stain is painted into the centre-left tile of every 4th image, and the report
   counts how often that tile is flagged. Painted stains are not real damage: this checks that the detector can fire
   at all, not how well it finds real water damage.

    COZMO_DATA=/path/to/captures python bench/damage_sanity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from video_vs_lidar import DATA, DERIVED, WALKS, code_commit  # noqa: E402

import cozmo.damage.detect as dd  # noqa: E402
from cozmo.ingest.stray import StrayCapture  # noqa: E402

OUT = ROOT / "bench" / "results" / "damage_sanity.json"


def paint_stain(image: np.ndarray, rng) -> np.ndarray:
    h, w = image.shape[:2]
    cy, cx = int(h * 0.5), int(w * 0.375)
    mask = np.zeros((h, w), np.float32)
    for _ in range(8):
        cv2.ellipse(mask, (cx + int(rng.normal(0, 25)), cy + int(rng.normal(0, 20))),
                    (int(rng.uniform(20, 45)), int(rng.uniform(15, 35))), rng.uniform(0, 180), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)[..., None]
    return (image * (1 - 0.55 * mask) + np.array([120, 85, 40], np.float32) * 0.55 * mask).astype(np.uint8)


def main():
    threshold = dd.P_MIN
    dd.P_MIN = 0.0                                  # score every tile
    rng = np.random.default_rng(1)
    report = {"benchmark": "damage_sanity", "code_commit": code_commit(), "model": dd.MODEL, "p_min": threshold, "walks": {}}
    for cid in WALKS:
        views = dd.lidar_views(StrayCapture(DATA / cid), DERIVED / f"{cid}_damage_frames")
        scores = np.array([h[3] for h in dd.classify(views)])
        painted = [dd.View(paint_stain(v.image, rng), v.depth, v.K, v.R, v.t, v.frame, v.upright) for v in views[::4]]
        hit = {}
        for k, box, cls, p in dd.classify(painted):
            if box[0] <= 0.375 < box[2] and box[1] <= 0.5 < box[3]:
                hit[k] = (cls, p)
        found = sum(p >= threshold for _, p in hit.values())
        report["walks"][cid] = {"images": len(views), "tiles": int(len(scores)),
                                "max_damage_score": round(float(scores.max()), 3),
                                "tiles_over_threshold": int((scores >= threshold).sum()),
                                "painted_images": len(painted), "painted_found": found,
                                "painted_scores": [round(p, 3) for _, p in sorted(hit.values(), key=lambda x: -x[1])]}
        print(cid, json.dumps(report["walks"][cid]), flush=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
