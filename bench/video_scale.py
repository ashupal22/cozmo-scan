"""How far off is the video tier's real-world scale, and can it be calibrated? (fix loop for G-WALL-VIDEO)

The video tier sets its scale with DA3METRIC depth, which depends on the focal length. Per walk this
measures, against ARKit and LiDAR of the same walk:
  - focal length: room lines (cozmo/video/focal.py) and DA3's own estimate, against ARKit's calibration
  - end-to-end depth scale: LiDAR depth / video-tier depth on every key frame (median), with the
    metric gain switched off, so the number is the raw bias the gain must remove
Then leave-one-walk-out: the gain from the other walks, applied to the held-out walk.
Caveat: c7d28f72c6 and 1a8384c3f6 are the same flat, so there are two independent flats, not three.

    COZMO_DATA=/path/to/captures python bench/video_scale.py
Writes bench/results/video_scale.json.
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

from video_vs_lidar import DATA, DERIVED, WALKS, code_commit, key_rows, upright_video  # noqa: E402

from cozmo.ingest.stray import StrayCapture, upright_rotate_code  # noqa: E402
from cozmo.video.capture import load_video  # noqa: E402

OUT = ROOT / "bench" / "results" / "video_scale.json"


def depth_scale(vcap, cap: StrayCapture, rows: np.ndarray, code) -> np.ndarray:
    """LiDAR / video depth per key frame (median over LiDAR high-confidence pixels 0.3-4.5 m)."""
    out = []
    for k, r in enumerate(rows):
        L = cap.depth(r)
        C = cap.confidence(r)
        if code is not None:
            L, C = cv2.rotate(L, code), cv2.rotate(C, code)
        d = cv2.resize(vcap.depths[k], (L.shape[1], L.shape[0]), interpolation=cv2.INTER_AREA)
        m = (C == 2) & (L > 0.3) & (L < 4.5) & (d > 0)
        out.append(float(np.median(L[m] / d[m])) if m.sum() > 200 else np.nan)
    return np.array(out)


def main():
    report = {"benchmark": "video_scale", "code_commit": code_commit(), "walks": {}}
    for cid in WALKS:
        cap = StrayCapture(DATA / cid)
        codes = [upright_rotate_code(cap.rotation(i)) for i in range(0, len(cap), 30)]
        code = max(set(codes), key=codes.count)
        k = cap.intrinsics(0, "rgb")
        sideways = code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
        true_fx = k.fy / cap.rgb_size[1] if sideways else k.fx / cap.rgb_size[0]
        video = upright_video(cid)
        vcap = load_video(video, DERIVED / f"{cid}_video_work", metric_gain=1.0)
        rows = key_rows(video, vcap.frame_files, cap)
        ratio = depth_scale(vcap, cap, rows, code)
        info = vcap.info
        report["walks"][cid] = {
            "true_fx_over_width": round(true_fx, 4),
            "focal_room_lines": info["fx_over_width"], "focal_source": info["focal_source"],
            "focal_room_lines_error_pct": round(100 * (info["fx_over_width"] / true_fx - 1), 2),
            "focal_da3_error_pct": round(100 * (info["fx_over_width_da3_median"] / true_fx - 1), 2),
            "depth_scale_lidar_over_video": round(float(np.nanmedian(ratio)), 4),
            "depth_scale_p10_p90": [round(float(np.nanpercentile(ratio, 10)), 3), round(float(np.nanpercentile(ratio, 90)), 3)],
            "frames": int(np.isfinite(ratio).sum()),
        }
        print(cid, json.dumps(report["walks"][cid]), flush=True)
    walks = report["walks"]
    loo = {}
    for cid in walks:
        others = [walks[o]["depth_scale_lidar_over_video"] for o in walks if o != cid]
        gain = float(np.mean(others))
        loo[cid] = {"gain_from_others": round(gain, 4),
                    "residual_scale_error_pct": round(100 * (walks[cid]["depth_scale_lidar_over_video"] / gain - 1), 2)}
    report["leave_one_out"] = loo
    report["gain_all_walks"] = round(float(np.mean([w["depth_scale_lidar_over_video"] for w in walks.values()])), 4)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(loo), "gain", report["gain_all_walks"])
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
