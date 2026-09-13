"""Video tier against the LiDAR tier on the same walk (brief: G-WALL-VIDEO, calibration at every tier).

The only reference we have for the video tier is the LiDAR tier of the same walk. Every Stray capture
holds the phone's video, so the video tier runs on that video, turned upright the way the Camera app
stores it, and its plan is compared with the LiDAR plan. LiDAR walls are good to about 1-2 cm, well
inside the video gate of 3%, so they serve as the reference. Caveat: this video is ARKit's 4:3 camera
stream without stabilisation, not a Camera-app clip.

Per walk:
  - rooms and footprint of both plans; the video plan is placed on the LiDAR plan by one rigid fit of
    their wall maps (rotation and shift only: a scale error must show, not be fitted away)
  - rooms paired by IoU >= 0.5 after a local alignment per room (wall lengths are local measurements;
    placement in the stitched plan is judged by the footprint)
  - walls paired corner to corner; the gate (G-WALL-VIDEO, 3%) is scored on LiDAR walls of at least 1 m
    whose two neighbouring faces were fitted to wall points, and a LiDAR wall with no video partner is
    a miss; plus whether each video wall's 90% interval holds the LiDAR length (calibration)
  - main dimensions of paired rooms
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

from cozmo.export.document import VIDEO_INTERVAL_SCALE  # noqa: E402
from cozmo.geometry.walls import _to_aligned, wall_mask  # noqa: E402
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
GATE_WALL_M = 1.0          # shorter walls: LiDAR's own 1-2 cm is already 1-2% of the length
PAIR_REACH_M = 1.5         # room centroids this close after the global fit are candidates
CORNER_MATCH_M = 0.6       # corners this close drive the local shift
CORNER_REACH_M = 0.20      # a video wall pairs when both its corners lie this close to the LiDAR wall's...
CORNER_REACH_SHARE = 0.15  # ...or within this share of the wall's length
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


def _dominant_angle(vertices: np.ndarray) -> float:
    """Wall direction of an outline, degrees modulo 90 (length-weighted, 4-fold symmetric mean)."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    length = np.linalg.norm(edges, axis=1)
    angle = np.arctan2(edges[:, 1], edges[:, 0])
    return float(np.degrees(np.angle(np.sum(length * np.exp(4j * angle))) / 4))


def _rotate(points: np.ndarray, degrees: float, about: np.ndarray) -> np.ndarray:
    t = np.radians(degrees)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (points - about) @ R.T + about


def local_align(ref: np.ndarray, test: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Test room vertices (already placed by the global fit) moved onto the reference room: the small
    rotation that makes their wall directions agree, then the median shift between nearest corners.
    Wall lengths are local measurements, so a room misplaced in the stitched plan still counts."""
    turn = (_dominant_angle(ref) - _dominant_angle(test) + 45.0) % 90.0 - 45.0
    moved = _rotate(test, turn, test.mean(axis=0))
    shift = np.zeros(2)
    for _ in range(iterations):
        cur = moved + shift
        dist = np.linalg.norm(ref[:, None, :] - cur[None, :, :], axis=2)
        nearest = dist.argmin(axis=1)
        ok = dist[np.arange(len(ref)), nearest] < CORNER_MATCH_M
        if ok.sum() < 2:
            break
        shift += np.median(ref[ok] - cur[nearest[ok]], axis=0)
    return moved + shift


def main_dimensions(outline) -> np.ndarray:
    return np.sort(np.ptp(np.asarray(outline.vertices) @ _to_aligned(outline.yaw_deg).T, axis=0))


def _iou(a: Polygon, b: Polygon) -> float:
    a, b = (a if a.is_valid else a.buffer(0)), (b if b.is_valid else b.buffer(0))
    union = a.union(b).area
    return a.intersection(b).area / union if union > 0 else 0.0


def pair_rooms(lidar_outlines: dict, video_outlines: dict, fit) -> list[tuple[int, int, float, np.ndarray]]:
    """(LiDAR room, video room, IoU after local alignment, aligned video vertices), largest rooms first.
    Candidates must lie near each other after the global fit."""
    pairs, used = [], set()
    for rl, ol in sorted(lidar_outlines.items(), key=lambda kv: -kv[1].area_m2):
        ref = np.asarray(ol.vertices, float)
        ref_poly = Polygon(ref)
        reach = max(PAIR_REACH_M, 0.5 * np.sqrt(ol.area_m2))
        best = None
        for rv, ov in video_outlines.items():
            if rv in used:
                continue
            placed = fit.apply(np.asarray(ov.vertices, float))
            if np.linalg.norm(np.asarray(Polygon(placed).centroid.coords[0]) - np.asarray(ref_poly.centroid.coords[0])) > reach:
                continue
            aligned = local_align(ref, placed)
            iou = _iou(ref_poly, Polygon(aligned))
            if best is None or iou > best[2]:
                best = (rl, rv, iou, aligned)
        if best is not None and best[2] >= MIN_IOU:
            used.add(best[1])
            pairs.append(best)
    return pairs


def pair_walls(ol, aligned: np.ndarray, ov, doc_room: dict, rl: int, lidar_room: dict) -> list[dict]:
    """Every LiDAR wall of at least MIN_WALL_M against the video wall whose two corners lie near its two
    corners. A LiDAR wall's length is set by its two neighbouring faces; it is a trustworthy reference
    only when both were fitted to wall points (support > 0)."""
    n_ref, n_test = len(ol.walls), len(ov.walls)
    rows = []
    for k, w in enumerate(ol.walls):
        if w.length_m < MIN_WALL_M:
            continue
        s, e = np.asarray(ol.vertices[k], float), np.asarray(ol.vertices[(k + 1) % n_ref], float)
        reach = max(CORNER_REACH_M, CORNER_REACH_SHARE * w.length_m)
        best = None
        for j in range(n_test):
            ts, te = aligned[j], aligned[(j + 1) % n_test]
            cost = min(max(np.linalg.norm(s - ts), np.linalg.norm(e - te)),
                       max(np.linalg.norm(s - te), np.linalg.norm(e - ts)))
            if cost <= reach and (best is None or cost < best[0]):
                best = (cost, j)
        ml = lidar_room["walls"][k]["length_m"]
        row = {"room": f"L{rl}", "lidar_m": round(w.length_m, 3), "lidar_ci": [ml["ci_low"], ml["ci_high"]],
               "reference_ok": bool(ol.walls[k - 1].support > 0 and ol.walls[(k + 1) % n_ref].support > 0)}
        if best is None:
            rows.append(row | {"video_m": None})
            continue
        cost, j = best
        length = float(ov.walls[j].length_m)
        m = doc_room["walls"][j]["length_m"]
        err = (length - w.length_m) / w.length_m
        rows.append(row | {"video_m": round(length, 3), "error_pct": round(100 * err, 1),
                           "within_gate": bool(abs(err) <= GATE),
                           "interval_holds_lidar": bool(m["ci_low"] <= w.length_m <= m["ci_high"]),
                           "video_ci": [m["ci_low"], m["ci_high"]],
                           "corner_offset_m": round(float(cost), 3)})
    return rows


def _wall_summary(rows: list[dict]) -> dict:
    paired = [r for r in rows if r["video_m"] is not None]
    errs = np.array([r["error_pct"] for r in paired])
    return {"walls": len(rows), "paired": len(paired),
            "within_3pct": f"{int(np.sum(np.abs(errs) <= 100 * GATE))}/{len(rows)}",
            "error_pct_median": round(float(np.median(errs)), 1) if len(errs) else None,
            "abs_error_pct_median": round(float(np.median(np.abs(errs))), 1) if len(errs) else None,
            "abs_error_pct_p90": round(float(np.percentile(np.abs(errs), 90)), 1) if len(errs) else None,
            "interval_holds_lidar": f"{sum(r['interval_holds_lidar'] for r in paired)}/{len(paired)}"}


def compare(lidar, video, fit) -> dict:
    """Rooms, walls and main dimensions of the video plan against the LiDAR plan of the same walk."""
    doc_v = {r["id"]: r for r in video.document["rooms"]}
    doc_l = {r["id"]: r for r in lidar.document["rooms"]}
    pairs = pair_rooms(lidar.outlines, video.outlines, fit)
    walls, dims = [], []
    for rl, rv, iou, aligned in pairs:
        ol, ov = lidar.outlines[rl], video.outlines[rv]
        walls += pair_walls(ol, aligned, ov, doc_v[f"R{rv}"], rl, doc_l[f"R{rl}"])
        for a, b in zip(main_dimensions(ol), main_dimensions(ov)):
            dims.append({"room": f"L{rl}", "lidar_m": round(float(a), 3), "video_m": round(float(b), 3),
                         "error_pct": round(100 * float(b - a) / float(a), 1)})
    area_l = sum(o.area_m2 for o in lidar.outlines.values())
    area_v = sum(o.area_m2 for o in video.outlines.values())
    fp = video.document["plan"]["footprint_area_m2"]
    dim_err = np.array([d["error_pct"] for d in dims])
    reference = [w for w in walls if w["reference_ok"] and w["lidar_m"] >= GATE_WALL_M]
    return {
        "rooms": {"lidar": len(lidar.outlines), "video": len(video.outlines), "paired": len(pairs),
                  "paired_iou": [round(p[2], 3) for p in pairs],
                  "paired_area_share": round(sum(lidar.outlines[p[0]].area_m2 for p in pairs) / area_l, 3)},
        "footprint_m2": {"lidar": round(area_l, 2), "video": round(area_v, 2),
                         "error_pct": round(100 * (area_v - area_l) / area_l, 1),
                         "video_interval": [fp["ci_low"], fp["ci_high"]],
                         "interval_holds_lidar": bool(fp["ci_low"] <= area_l <= fp["ci_high"])},
        "gate_walls": _wall_summary(reference),
        "all_walls": _wall_summary(walls),
        "dimensions": {"count": len(dims), "within_3pct": f"{int(np.sum(np.abs(dim_err) <= 100 * GATE))}/{len(dims)}",
                       "abs_error_pct_median": round(float(np.median(np.abs(dim_err))), 1) if len(dims) else None,
                       "error_pct_median": round(float(np.median(dim_err)), 1) if len(dims) else None},
        "fit": {"yaw_deg": round(fit.yaw_deg, 1), "overlap_fraction": round(fit.overlap_fraction, 3),
                "overlap_median_cm": round(100 * fit.overlap_median_m, 1)},
        "wall_pairs": walls,
        "dimension_pairs": dims,
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
        vcap = load_video(video_path, work, fx_over_width=true_fx if name in ("truefocal", "oracle") else None,
                                  max_keyframes=None)   # every key frame, as the committed results were produced
        if name == "oracle":
            vcap = oracle_capture(vcap, cap, key_rows(video_path, vcap.frame_files, cap), code)
        video = plan_video_capture(vcap, video_path, t0)
        source, _ = wall_map(video)
        fit = match_walls(target, target_n, source, yaw_range_deg=(-180, 179), yaw_step_deg=1.0, max_shift_m=None,
                          cell_m=0.05, center=True)
        result = compare(lidar, video, fit) if fit else {"fit": None}
        result["video_capture"] = {key: v for key, v in vcap.info.items() if key != "fx_over_width_per_run"}
        result["true_fx_over_width"] = round(true_fx, 4)
        result["video_interval_scale"] = VIDEO_INTERVAL_SCALE
        result["seconds"] = round(time.time() - t0, 1)
        out[name] = result
        summary = {k: result.get(k) for k in ("rooms", "footprint_m2", "gate_walls", "dimensions")}
        print(cid, name, json.dumps(summary), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variants", default="", help="comma-separated: truefocal, oracle")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    variants = [v for v in args.variants.split(",") if v]
    report = {"benchmark": "video_vs_lidar", "code_commit": code_commit(), "walks": {}}
    if out.is_file():
        old = json.loads(out.read_text())
        report["walks"] = old.get("walks", {})
    for cid in args.walks:
        report["walks"][cid] = run_walk(cid, variants)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
