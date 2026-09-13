"""Video tier against the LiDAR tier on the same walk (brief: G-WALL-VIDEO, calibration at every tier).

The only reference we have for the video tier is the LiDAR tier of the same walk. Every Stray capture
holds the phone's video, so the video tier runs on that video, turned upright the way the Camera app
stores it, and its plan is compared with the LiDAR plan. LiDAR walls are good to about 1-2 cm, well
inside the video gate of 3%, so they serve as the reference. Caveat: this video is ARKit's 4:3 camera
stream without stabilisation, not a Camera-app clip.

Per walk:
  - rooms and footprint of both plans; the video plan is placed on the LiDAR plan by one rigid fit of
    their wall maps (rotation and shift only: a scale error must show, not be fitted away)
  - rooms paired by IoU >= 0.5, then walls paired by direction and position
  - wall length error, share within 3% (G-WALL-VIDEO), and whether each video wall's 90% interval holds
    the LiDAR length (calibration)
Diagnostic variants (--variants):
  truefocal   the video tier given the true focal length instead of estimating it
  oracle      ARKit's camera poses with the video tier's depth (true focal): what depth alone limits

    COZMO_DATA=/path/to/captures python bench/video_vs_lidar.py [walk ...] [--variants truefocal,oracle]
Writes bench/results/video_vs_lidar.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cozmo.geometry.walls import wall_mask  # noqa: E402
from cozmo.ingest.stray import StrayCapture, upright_rotate_code  # noqa: E402
from cozmo.pipeline import plan_video_capture, run_lidar  # noqa: E402
from cozmo.slam.matching import match_walls  # noqa: E402

DATA = Path(os.environ.get("COZMO_DATA", ROOT / "data" / "captures"))
DERIVED = ROOT / "data" / "derived"
OUT = ROOT / "bench" / "results" / "video_vs_lidar.json"
WALKS = ("c00a170fe1", "1a8384c3f6", "c7d28f72c6")
MIN_IOU = 0.5
MIN_WALL_M = 0.5
GATE = 0.03
VIDEO_POSE_LAG_S = 0.075   # Stray's video frames lag its poses by 4-5 frames (found by feature-based rotations)
TRANSPOSE = {cv2.ROTATE_90_CLOCKWISE: "transpose=1", cv2.ROTATE_90_COUNTERCLOCKWISE: "transpose=2",
             cv2.ROTATE_180: "transpose=1,transpose=1"}


def code_commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                          capture_output=True, text=True).stdout.strip()


def upright_video(cid: str) -> Path:
    """The capture's rgb.mp4 turned the way the phone was mostly held, with its timestamps kept."""
    out = DERIVED / f"{cid}_upright.mp4"
    if out.is_file():
        return out
    cap = StrayCapture(DATA / cid)
    codes = [upright_rotate_code(cap.rotation(i)) for i in range(0, len(cap), 30)]
    code = max(set(codes), key=codes.count)
    DERIVED.mkdir(parents=True, exist_ok=True)
    vf = ["-vf", TRANSPOSE[code]] if code is not None else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(DATA / cid / "rgb.mp4"), *vf, "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-an", str(out)], check=True)
    return out


def key_rows(video: Path, frames: list[Path], cap: StrayCapture) -> np.ndarray:
    """Odometry row of every key frame: the key frame is matched to its source frame by image, and the
    source frame to the pose by timestamp plus the video lag."""
    pts = np.array([float(v.strip(",")) for v in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0",
         str(video)], capture_output=True, text=True).stdout.split() if v.strip(",")])
    w, h = 72, 96
    first = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    if first.shape[1] > first.shape[0]:
        w, h = h, w
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={w}:{h}", "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
    src = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
    t = cap.timestamps - cap.timestamps[0]
    rows = []
    for k, f in enumerate(frames):
        key = cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        cand = np.argsort(np.abs(pts - k / 3.0))[:6]
        j = int(cand[int(np.argmin([np.mean(np.abs(src[c] - key)) for c in cand]))])
        rows.append(int(np.argmin(np.abs(t - (pts[j] + VIDEO_POSE_LAG_S)))))
    return np.array(rows)


def oracle_capture(vcap, cap: StrayCapture, rows: np.ndarray, code):
    """The video capture with ARKit's poses (upright camera axes) in place of the estimated ones."""
    from cozmo.video.capture import VideoCapture
    Q = {cv2.ROTATE_90_CLOCKWISE: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]),
         cv2.ROTATE_90_COUNTERCLOCKWISE: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]),
         cv2.ROTATE_180: np.diag([-1.0, -1.0, 1.0]), None: np.eye(3)}[code]
    rotations = np.array([cap.rotation(r) @ Q for r in rows])
    return VideoCapture(vcap.path, vcap.frame_files, vcap.timestamps, cap.positions[rows].copy(), rotations,
                        vcap.depths, vcap.confidences, vcap.Ks, vcap.scale_sigma, dict(vcap.info))


def wall_map(plan):
    walls = plan.points.subset(wall_mask(plan.points, plan.floor))
    _, keep = np.unique(np.floor(walls.xyz[:, [0, 2]] / 0.02).astype(np.int64), axis=0, return_index=True)
    keep = np.sort(keep)
    n = walls.normal[keep][:, [0, 2]].astype(float)
    return walls.xyz[keep][:, [0, 2]].astype(float), n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def _direction(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    return d / (np.linalg.norm(d) + 1e-12)


def compare(lidar, video, fit) -> dict:
    """Pair rooms and walls of the video plan (moved by `fit`) with the LiDAR plan."""
    doc_v = {r["id"]: r for r in video.document["rooms"]}
    rooms_l = {rid: Polygon(o.vertices) for rid, o in lidar.outlines.items()}
    rooms_v = {rid: Polygon(fit.apply(o.vertices)) for rid, o in video.outlines.items()}
    pairs, used = [], set()
    for rl, pl in sorted(rooms_l.items(), key=lambda kv: -kv[1].area):
        best = None
        for rv, pv in rooms_v.items():
            if rv in used or not pl.intersects(pv):
                continue
            iou = pl.intersection(pv).area / pl.union(pv).area
            if best is None or iou > best[0]:
                best = (iou, rv)
        if best and best[0] >= MIN_IOU:
            used.add(best[1])
            pairs.append((rl, best[1], best[0]))
    walls = []
    for rl, rv, iou in pairs:
        ol, ov = lidar.outlines[rl], video.outlines[rv]
        vs = [(k, fit.apply(np.array([w.start])), fit.apply(np.array([w.end])), w) for k, w in enumerate(ov.walls)]
        for w in ol.walls:
            if w.length_m < MIN_WALL_M:
                continue
            d = _direction(w.start, w.end)
            normal = np.array([-d[1], d[0]])
            best = None
            for k, s, e, wv in vs:
                s, e = s[0], e[0]
                dv = _direction(s, e)
                if abs(d @ dv) < np.cos(np.radians(10)):
                    continue
                offset = abs((0.5 * (s + e) - 0.5 * (np.asarray(w.start) + np.asarray(w.end))) @ normal)
                a0, a1 = sorted([(s - w.start) @ d, (e - w.start) @ d])
                overlap = min(a1, w.length_m) - max(a0, 0.0)
                if offset > 0.35 or overlap < 0.5 * min(w.length_m, wv.length_m):
                    continue
                if best is None or offset < best[0]:
                    best = (offset, k, wv)
            if best is None:
                walls.append({"room": f"R{rl}", "lidar_m": round(w.length_m, 3), "video_m": None})
                continue
            _, k, wv = best
            m = doc_v[f"R{rv}"]["walls"][k]["length_m"]
            err = (wv.length_m - w.length_m) / w.length_m
            walls.append({"room": f"R{rl}", "lidar_m": round(w.length_m, 3), "video_m": round(wv.length_m, 3),
                          "error_pct": round(100 * err, 1), "within_gate": bool(abs(err) <= GATE),
                          "interval_holds_lidar": bool(m["ci_low"] <= w.length_m <= m["ci_high"]),
                          "offset_m": round(best[0], 3)})
    matched = [w for w in walls if w["video_m"] is not None]
    errs = np.array([w["error_pct"] for w in matched]) if matched else np.array([])
    area_l = sum(p.area for p in rooms_l.values())
    area_v = sum(p.area for p in rooms_v.values())
    fp = video.document["plan"]["footprint_area_m2"]
    return {
        "rooms": {"lidar": len(rooms_l), "video": len(rooms_v), "paired": len(pairs),
                  "paired_iou": [round(p[2], 3) for p in pairs]},
        "footprint_m2": {"lidar": round(area_l, 2), "video": round(area_v, 2),
                         "error_pct": round(100 * (area_v - area_l) / area_l, 1),
                         "video_interval": [fp["ci_low"], fp["ci_high"]],
                         "interval_holds_lidar": bool(fp["ci_low"] <= area_l <= fp["ci_high"])},
        "walls": {"lidar_walls_over_0.5m": len(walls), "paired": len(matched),
                  "within_3pct": f"{int(np.sum(np.abs(errs) <= 100 * GATE))}/{len(walls)}",
                  "error_pct_median": round(float(np.median(errs)), 1) if len(errs) else None,
                  "abs_error_pct_median": round(float(np.median(np.abs(errs))), 1) if len(errs) else None,
                  "interval_holds_lidar": f"{sum(w['interval_holds_lidar'] for w in matched)}/{len(matched)}"},
        "fit": {"yaw_deg": round(fit.yaw_deg, 1), "overlap_fraction": round(fit.overlap_fraction, 3),
                "overlap_median_cm": round(100 * fit.overlap_median_m, 1)},
        "wall_pairs": walls,
    }


def run_walk(cid: str, variants: list[str]) -> dict:
    from cozmo.video.capture import load_video
    lidar = run_lidar(DATA / cid)
    target, target_n = wall_map(lidar)
    video_path = upright_video(cid)
    work = DERIVED / f"{cid}_video_work"
    cap = StrayCapture(DATA / cid)
    codes = [upright_rotate_code(cap.rotation(i)) for i in range(0, len(cap), 30)]
    code = max(set(codes), key=codes.count)
    k = cap.intrinsics(0, "rgb")
    sideways = code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
    true_fx = k.fy / cap.rgb_size[1] if sideways else k.fx / cap.rgb_size[0]  # focal / upright frame width
    out = {}
    for name in ["video"] + variants:
        t0 = time.time()
        vcap = load_video(video_path, work, fx_over_width=true_fx if name in ("truefocal", "oracle") else None)
        if name == "oracle":
            vcap = oracle_capture(vcap, cap, key_rows(video_path, vcap.frame_files, cap), code)
        video = plan_video_capture(vcap, video_path, t0)
        source, _ = wall_map(video)
        fit = match_walls(target, target_n, source, yaw_range_deg=(-180, 179), yaw_step_deg=1.0, max_shift_m=None,
                          cell_m=0.05, center=True)
        result = compare(lidar, video, fit) if fit else {"fit": None}
        result["video_capture"] = {key: v for key, v in vcap.info.items() if key != "fx_over_width_per_run"}
        result["true_fx_over_width"] = round(true_fx, 4)
        result["seconds"] = round(time.time() - t0, 1)
        out[name] = result
        summary = {k: result.get(k) for k in ("rooms", "footprint_m2")} | {"walls": {k: v for k, v in result.get("walls", {}).items()}}
        print(cid, name, json.dumps(summary), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variants", default="", help="comma-separated: truefocal, oracle")
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    report = {"benchmark": "video_vs_lidar", "code_commit": code_commit(), "walks": {}}
    if OUT.is_file():
        old = json.loads(OUT.read_text())
        report["walks"] = old.get("walks", {})
    for cid in args.walks:
        report["walks"][cid] = run_walk(cid, variants)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
