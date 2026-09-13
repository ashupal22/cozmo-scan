"""How far off is the video tier's real-world scale, and can it be calibrated? (fix loop for G-WALL-VIDEO)

The video tier sets its scale with DA3METRIC depth, which depends on the focal length. Per walk this
measures, against ARKit and LiDAR of the same walk:
  - focal length: room lines (cozmo/video/focal.py) and DA3's own estimate, against ARKit's calibration
  - overall depth scale: LiDAR depth / video-tier depth on every key frame (median), with the metric gain
    and the range correction switched off, so the number is the raw bias the gain must remove
  - range bias after the gain: log(LiDAR / video) against log(video depth), on pixels both measured well,
    binned by video depth (what is known when running); fitted as a + b log(depth)
Both calibrations are checked leave-one-walk-out: fitted on the other walks, applied to the held-out walk.
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
from cozmo.video import capture as vc  # noqa: E402

OUT = ROOT / "bench" / "results" / "video_scale.json"
RANGE_EDGES = np.array([0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.1, 2.6, 3.2, 4.0])
PIXELS_PER_FRAME = 300


def lidar_frame(cap: StrayCapture, row: int, code):
    L, C = cap.depth(row), cap.confidence(row)
    return (cv2.rotate(L, code), cv2.rotate(C, code)) if code is not None else (L, C)


def depth_scale(vcap, cap: StrayCapture, rows: np.ndarray, code) -> np.ndarray:
    """LiDAR / video depth per key frame (median over LiDAR high-confidence pixels 0.3-4.5 m)."""
    out = []
    for k, r in enumerate(rows):
        L, C = lidar_frame(cap, r, code)
        d = cv2.resize(vcap.depths[k], (L.shape[1], L.shape[0]), interpolation=cv2.INTER_AREA)
        m = (C == 2) & (L > 0.3) & (L < 4.5) & (d > 0)
        out.append(float(np.median(L[m] / d[m])) if m.sum() > 200 else np.nan)
    return np.array(out)


def depth_pairs(vcap, cap: StrayCapture, rows: np.ndarray, code, seed: int = 0):
    """(LiDAR depth, video depth) on a sample of pixels that both measured well, away from depth edges."""
    rng = np.random.default_rng(seed)
    Ls, Vs = [], []
    for k, r in enumerate(rows):
        L, C = lidar_frame(cap, r, code)
        d = cv2.resize(vcap.depths[k], (L.shape[1], L.shape[0]), interpolation=cv2.INTER_NEAREST)
        good = cv2.resize(vcap.confidences[k], (L.shape[1], L.shape[0]), interpolation=cv2.INTER_NEAREST) == 2
        ok = (C == 2) & (L > 0.3) & (L < 4.5) & (d > 0.2) & good & (np.abs(cv2.medianBlur(L, 5) - L) < 0.02 * L)
        idx = np.flatnonzero(ok)
        idx = rng.choice(idx, min(PIXELS_PER_FRAME, len(idx)), replace=False) if len(idx) else idx
        Ls.append(L.ravel()[idx])
        Vs.append(d.ravel()[idx])
    return np.concatenate(Ls), np.concatenate(Vs)


def binned(L: np.ndarray, V: np.ndarray) -> np.ndarray:
    """(log bin centre, median log(L / V), count) per video-depth bin with enough pixels."""
    out = []
    for lo, hi in zip(RANGE_EDGES[:-1], RANGE_EDGES[1:]):
        m = (V >= lo) & (V < hi)
        if m.sum() > 100:
            out.append((np.log(np.sqrt(lo * hi)), float(np.median(np.log(L[m] / V[m]))), int(m.sum())))
    return np.array(out)


def fit_range(L: np.ndarray, V: np.ndarray) -> tuple[float, float]:
    B = binned(L, V)
    w = np.sqrt(B[:, 2])
    A = np.column_stack([np.ones(len(B)), B[:, 0]])
    a, b = np.linalg.lstsq(A * w[:, None], B[:, 1] * w, rcond=None)[0]
    return float(a), float(b)


def _bins_pct(B: np.ndarray) -> list:
    return [[round(float(np.exp(x)), 2), round(100 * float(np.exp(y) - 1), 1), int(n)] for x, y, n in B]


def main():
    report = {"benchmark": "video_scale", "code_commit": code_commit(), "walks": {}}
    pairs = {}
    for cid in WALKS:
        cap = StrayCapture(DATA / cid)
        codes = [upright_rotate_code(cap.rotation(i)) for i in range(0, len(cap), 30)]
        code = max(set(codes), key=codes.count)
        k = cap.intrinsics(0, "rgb")
        sideways = code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
        true_fx = k.fy / cap.rgb_size[1] if sideways else k.fx / cap.rgb_size[0]
        video = upright_video(cid)
        raw = vc.load_video(video, DERIVED / f"{cid}_video_work", metric_gain=1.0, range_correction=False, max_keyframes=None)
        rows = key_rows(video, raw.frame_files, cap)
        ratio = depth_scale(raw, cap, rows, code)
        L, V = depth_pairs(raw, cap, rows, code)
        pairs[cid] = (L, V * vc.METRIC_DEPTH_GAIN)   # the gain scales every run, so every depth, exactly
        info = raw.info
        report["walks"][cid] = {
            "true_fx_over_width": round(true_fx, 4),
            "focal_room_lines": info["fx_over_width"], "focal_source": info["focal_source"],
            "focal_room_lines_error_pct": round(100 * (info["fx_over_width"] / true_fx - 1), 2),
            "focal_da3_error_pct": round(100 * (info["fx_over_width_da3_median"] / true_fx - 1), 2),
            "depth_scale_lidar_over_video": round(float(np.nanmedian(ratio)), 4),
            "depth_scale_p10_p90": [round(float(np.nanpercentile(ratio, 10)), 3), round(float(np.nanpercentile(ratio, 90)), 3)],
            "frames": int(np.isfinite(ratio).sum()), "pixel_pairs": int(len(L)),
        }
        print(cid, json.dumps(report["walks"][cid]), flush=True)

    walks = report["walks"]
    loo = {}
    for cid in walks:
        others = [o for o in walks if o != cid]
        gain = float(np.mean([walks[o]["depth_scale_lidar_over_video"] for o in others]))
        a, b = fit_range(np.concatenate([pairs[o][0] for o in others]), np.concatenate([pairs[o][1] for o in others]))
        L, V = pairs[cid]
        corrected = V * np.exp(a) * np.clip(V, *vc.RANGE_CALIBRATED_M) ** b
        loo[cid] = {"gain_from_others": round(gain, 4),
                    "residual_scale_error_pct": round(100 * (walks[cid]["depth_scale_lidar_over_video"] / gain - 1), 2),
                    "range_fit_from_others": [round(a, 4), round(b, 4)],
                    "range_bias_before_pct": _bins_pct(binned(L, V)),
                    "range_bias_after_pct": _bins_pct(binned(L, corrected))}
        print(cid, "leave-one-out:", json.dumps(loo[cid]), flush=True)
    report["leave_one_out"] = loo
    report["gain_all_walks"] = round(float(np.mean([w["depth_scale_lidar_over_video"] for w in walks.values()])), 4)
    a, b = fit_range(np.concatenate([p[0] for p in pairs.values()]), np.concatenate([p[1] for p in pairs.values()]))
    report["range_fit_all_walks"] = [round(a, 4), round(b, 4)]
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("gain", report["gain_all_walks"], "range fit a, b", report["range_fit_all_walks"])
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
