"""A plain video turned into the same frame interface as a LiDAR capture.

1. Key frames: ffmpeg at KEYFRAME_FPS, upright (ffmpeg applies the phone's rotation tag), long side
   FRAME_LONG_SIDE px.
2. Depth Anything 3 (cozmo.video.da3) on overlapping runs of RUN frames that share OVERLAP frames: camera
   poses and depth per run, each run in its own frame and scale. Inside a run DA3 is good (camera
   positions within ~2 cm of ARKit over 1.5 m on our walks); the work is in joining runs.
3. One camera for every frame: focal length given (e.g. from metadata) or the median of DA3's estimates.
   DA3's focal differs from run to run by up to 15%, and with each run's own focal the same frame's
   point clouds differ by a stretch, not a scale.
4. Scale of every run, solved at once: neighbouring runs must agree on the depth of the frames they
   share (a per-pixel depth ratio, independent of the focal length), and every run must agree with
   DA3METRIC's depth in metres on the frames it samples. The metric anchors stop scale from creeping
   along the video; the shared frames keep neighbours consistent.
5. Level every run on its own floor and walls (floor normals parallel to up, wall normals perpendicular),
   as an IMU would. Without this, DA3's small tilt errors add up from run to run.
6. Join runs by the camera poses of the shared frames: heading from the average of their rotations,
   position from their camera centres. Heading errors that remain are corrected later by the drift
   step (wall directions, loop closures), which works on heading and position only.

The result has timestamps, positions, rotation(i), depth(i), confidence(i) and intrinsics(i, "depth"),
so fusion, drift correction and the room layout run unchanged. There are no ARKit relocalisations, so
drift correction runs without jump detection.

Model outputs are cached in the work folder: a second run on the same video skips the networks and
gives the same result.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cozmo.ingest.stray import CaptureError, Intrinsics
from cozmo.video import da3

KEYFRAME_FPS = 3.0
FRAME_LONG_SIDE = 960
RUN, OVERLAP = 12, 4            # longer runs fold opposite white walls together on our walks (bench/README.md)
PROCESS_RES = 504
RAY_POSE = True                 # cameras solved from DA3's ray output: focal 1-12% off on c00a, camera head 6-18%
CONF_KEEP = 0.6                 # keep the most confident 60% of each depth map
ALIGN_PIXELS = 4000
METRIC_SAMPLE_EVERY = 4         # key frames between metric-depth samples
METRIC_SIGMA = 0.07             # per-frame scatter of DA3METRIC's scale against LiDAR on c00a (6.3%)
SEAM_SIGMA_MIN = 0.01
MIN_FLOOR_NORMALS = 300         # below this a run is not levelled on its own floor (c00a: 36 -> 36 deg wrong, 417 -> 1.8 deg)
FOCAL_SIGMA_ESTIMATED = 0.05    # provisional: DA3 focal, median over the video
FOCAL_SIGMA_GIVEN = 0.02
METRIC_BIAS_SIGMA = 0.05        # provisional until calibrated against the LiDAR walks


@dataclass
class VideoCapture:
    path: Path
    frame_files: list[Path]
    timestamps: np.ndarray
    positions: np.ndarray        # (N, 3) camera centres, metres, y up
    rotations: np.ndarray        # (N, 3, 3) camera (OpenCV) to world
    depths: np.ndarray           # (N, h, w) metres
    confidences: np.ndarray      # (N, h, w) uint8, 2 = keep
    Ks: np.ndarray               # (N, 3, 3) for the depth maps
    scale_sigma: float           # relative uncertainty of the metric scale
    info: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def depth_size(self) -> tuple[int, int]:
        return self.depths.shape[2], self.depths.shape[1]

    def rotation(self, i: int) -> np.ndarray:
        return self.rotations[i]

    def depth(self, i: int) -> np.ndarray:
        return self.depths[i]

    def confidence(self, i: int) -> np.ndarray:
        return self.confidences[i]

    def intrinsics(self, i: int, image: str = "depth") -> Intrinsics:
        K = self.Ks[i]
        w, h = self.depth_size
        return Intrinsics(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), w, h)


# ---- frames and model outputs ----------------------------------------------------------------

def extract_keyframes(video: Path, out_dir: Path, fps: float = KEYFRAME_FPS, long_side: int = FRAME_LONG_SIDE) -> list[Path]:
    if not shutil.which("ffmpeg"):
        raise CaptureError("ffmpeg is needed to read videos: brew install ffmpeg")
    stamp = out_dir / "source.json"
    source = {"video": str(video.resolve()), "bytes": video.stat().st_size, "fps": fps, "long_side": long_side}
    files = sorted(out_dir.glob("*.jpg"))
    if files and stamp.is_file() and json.loads(stamp.read_text()) == source:
        return files
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in files:
        old.unlink()
    scale = f"scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps},{scale}", "-q:v", "2",
                    str(out_dir / "%05d.jpg")], check=True)
    files = sorted(out_dir.glob("*.jpg"))
    if len(files) < RUN:
        raise CaptureError(f"{video}: only {len(files)} key frames; the clip is too short")
    stamp.write_text(json.dumps(source))
    return files


def runs_for(n: int) -> list[tuple[int, int]]:
    """Frame ranges [a, b) of RUN frames, consecutive runs sharing OVERLAP frames."""
    step = RUN - OVERLAP
    runs = [(s, min(s + RUN, n)) for s in range(0, max(n - OVERLAP, 1), step)]
    if len(runs) > 1 and runs[-1][1] - runs[-1][0] < OVERLAP + 2:
        runs[-2] = (runs[-2][0], runs[-1][1])
        runs.pop()
    return runs


def pose_cache(work_dir: Path) -> Path:
    camera = "ray" if RAY_POSE else "head"
    return work_dir / "da3" / f"{da3.POSE_MODEL.split('/')[-1]}-{camera}-{PROCESS_RES}-{RUN}x{OVERLAP}"


def _views(frames: list[Path], a: int, b: int, cache: Path) -> da3.ViewSet:
    f = cache / f"views_{a:05d}_{b:05d}.npz"
    if not f.is_file():
        vs = da3.run_views([str(p) for p in frames[a:b]], process_res=PROCESS_RES, ray_pose=RAY_POSE)
        cache.mkdir(parents=True, exist_ok=True)
        # half precision keeps the cache small (relative precision 0.05%) and replays identically
        np.savez(f, depth=vs.depth.astype(np.float16), conf=vs.conf.astype(np.float16), world_to_cam=vs.world_to_cam,
                 K=vs.K, size=np.array(vs.size))
    z = np.load(f)
    return da3.ViewSet(z["depth"].astype(np.float32), z["conf"].astype(np.float32), z["world_to_cam"], z["K"], tuple(z["size"]))


def _metric(frames: list[Path], indices: list[int], fx_over_width: float, cache: Path) -> dict[int, np.ndarray]:
    """Metric depth for the given frames. The model output is cached for a focal length of 300 px, so
    another focal length is a plain rescale."""
    out = {}
    cache.mkdir(parents=True, exist_ok=True)
    for i in indices:
        f = cache / f"metric_{i:05d}.npy"
        if not f.is_file():
            d = da3.metric_depth([str(frames[i])], fx_over_width=1.0, process_res=PROCESS_RES)[0]
            np.save(f, (d * (300.0 / d.shape[1])).astype(np.float16))
        d = np.load(f).astype(np.float32)
        out[i] = d * (fx_over_width * d.shape[1] / 300.0)
    return out


# ---- geometry helpers -------------------------------------------------------------------------

def camera_matrix(fx_over_width: float, w: int, h: int) -> np.ndarray:
    f = fx_over_width * w
    return np.array([[f, 0.0, (w - 1) / 2], [0.0, f, (h - 1) / 2], [0.0, 0.0, 1.0]])


def _c2w(w2c: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = w2c[:3, :3].T
    T[:3, 3] = -w2c[:3, :3].T @ w2c[:3, 3]
    return T


def _confident(conf: np.ndarray) -> np.ndarray:
    return conf >= np.quantile(conf, 1 - CONF_KEEP)  # >=, so ties (e.g. uniform confidence) are kept


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising |s R src + t - dst|^2."""
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / A.var(0).sum())
    return s, R, md - s * R @ ms


def mean_rotation(Rs) -> np.ndarray:
    """Chordal mean of rotation matrices."""
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def yaw_part(R: np.ndarray) -> np.ndarray:
    """The rotation about +y closest to R."""
    theta = np.arctan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2])
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_to_y(up: np.ndarray) -> np.ndarray:
    """Rotation taking the unit vector `up` to +y."""
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, y)
    c = float(up @ y)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def frame_normals(depth: np.ndarray, conf: np.ndarray, K: np.ndarray, step: int = 3) -> np.ndarray:
    """Surface normals (camera frame) of confident pixels away from depth edges, every `step` pixels."""
    z = cv2.medianBlur(depth, 5)
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=-1)
    dx, dy = P[1:-1, 2:] - P[1:-1, :-2], P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    zc = z[1:-1, 1:-1]
    ok = _confident(conf)[1:-1, 1:-1] & (np.abs(dx[..., 2]) < 0.04 * zc) & (np.abs(dy[..., 2]) < 0.04 * zc)
    grid = np.zeros_like(ok)
    grid[::step, ::step] = True
    return n[ok & grid]


LEVEL_WINDOWS_DEG = (30, 15, 10, 6, 6)  # narrowing: the last passes ignore normals bent round corners


def level(normals: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, int]:
    """Up direction from surface normals: floor and ceiling normals parallel to it, wall normals
    perpendicular. Starts from a rough `up`; returns (up, number of floor/ceiling normals used)."""
    n_floor = 0
    for window_deg in LEVEL_WINDOWS_DEG:
        c = normals @ up
        window = np.radians(window_deg)
        horizontal = normals[np.abs(c) > np.cos(window)]
        vertical = normals[np.abs(c) < np.sin(window)]
        n_floor = len(horizontal)
        if n_floor < 50:
            return up, n_floor
        M = horizontal.T @ horizontal / len(horizontal) - vertical.T @ vertical / max(len(vertical), 1)
        cand = np.linalg.eigh(M)[1][:, -1]
        up = cand * np.sign(cand @ up)
    return up, n_floor


def solve_log_scales(n_runs: int, seams: list[tuple[int, float, float]], anchors: list[tuple[int, float, float]],
                     iterations: int = 5) -> np.ndarray:
    """Log scale per run from seams (x[r] - x[r-1] = value) and anchors (x[r] = value), each with a sigma;
    Huber-weighted least squares, so a bad frame cannot pull a run far."""
    rows, rhs, sig = [], [], []
    for r, value, sigma in seams:
        a = np.zeros(n_runs)
        a[r], a[r - 1] = 1.0, -1.0
        rows.append(a), rhs.append(value), sig.append(sigma)
    for r, value, sigma in anchors:
        a = np.zeros(n_runs)
        a[r] = 1.0
        rows.append(a), rhs.append(value), sig.append(sigma)
    A, b, s = np.array(rows), np.array(rhs), np.array(sig)
    w = np.ones(len(b))
    for _ in range(iterations):
        W = w / s
        x = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)[0]
        r = np.abs(A @ x - b) / s
        w = np.where(r > 2.0, 2.0 / np.maximum(r, 1e-9), 1.0)
    return x


# ---- joining runs -----------------------------------------------------------------------------

@dataclass
class Chain:
    c2w: np.ndarray              # (N, 4, 4) camera (OpenCV) to world, metres, y up
    depth: np.ndarray            # (N, h, w) metres
    conf: np.ndarray             # (N, h, w) DA3 confidence
    K: np.ndarray                # (3, 3) one camera for every frame, for the depth maps
    runs: list[tuple[int, int]]
    info: dict


def chain_runs(views: list[da3.ViewSet], runs: list[tuple[int, int]], K: np.ndarray,
               metric: dict[int, np.ndarray]) -> Chain:
    n = runs[-1][1]
    h, w = views[0].depth.shape[1:]
    local = [np.array([_c2w(e) for e in vs.world_to_cam]) for vs in views]

    # scale of every run
    seams, anchors, seam_spread = [], [], []
    for r in range(1, len(runs)):
        (pa, pb), (a, b) = runs[r - 1], runs[r]
        logs = []
        for f in range(a, pb):
            old, new = views[r - 1], views[r]
            m = _confident(old.conf[f - pa]) & _confident(new.conf[f - a])
            logs.append(float(np.median(np.log(old.depth[f - pa][m] / new.depth[f - a][m]))))
        logs = np.array(logs)
        spread = 1.4826 * float(np.median(np.abs(logs - np.median(logs))))
        seam_spread.append(spread)
        # s_new * d_new = s_old * d_old  ->  x[r] - x[r-1] = log(d_old / d_new)
        seams.append((r, float(np.median(logs)), max(spread / np.sqrt(len(logs)), SEAM_SIGMA_MIN)))
    for r, ((a, b), vs) in enumerate(zip(runs, views)):
        for f, m in metric.items():
            if a <= f < b:
                d = vs.depth[f - a]
                if m.shape != d.shape:
                    m = cv2.resize(m, d.shape[::-1], interpolation=cv2.INTER_AREA)
                good = _confident(vs.conf[f - a]) & (d > 1e-6) & (m > 0.3) & (m < 6.0)
                if good.sum() > 500:
                    anchors.append((r, float(np.median(np.log(m[good] / d[good]))), METRIC_SIGMA))
    if not anchors:
        raise CaptureError("could not set the metric scale: no frame had usable metric depth")
    log_s = solve_log_scales(len(runs), seams, anchors)
    scales = np.exp(log_s)

    # level every run on its own floor and walls
    ups, floor_counts = [], []
    for (a, b), vs, T in zip(runs, views, local):
        normals = np.concatenate([frame_normals(vs.depth[i], vs.conf[i], K) @ T[i, :3, :3].T for i in range(b - a)])
        rough = -T[:, :3, 1].mean(axis=0)
        up, count = level(normals, rough / np.linalg.norm(rough))
        ups.append(up)
        floor_counts.append(count)

    # join runs by the poses of the shared frames
    c2w = np.zeros((n, 4, 4))
    depth, conf = np.zeros((n, h, w), np.float32), np.zeros((n, h, w), np.float32)
    levelled, seam_turn, seam_shift, level_change = [], [], [], []
    A_prev = b_prev = None
    for r, ((a, b), vs, T) in enumerate(zip(runs, views, local)):
        s = scales[r]
        own_level = rotation_to_y(ups[r]) if floor_counts[r] >= MIN_FLOOR_NORMALS else None
        if r == 0:
            A = own_level if own_level is not None else rotation_to_y(ups[r])
            t = np.zeros(3)
        else:
            pa, pb = runs[r - 1]
            shared = range(a, pb)
            R_full = mean_rotation([c2w[f, :3, :3] @ T[f - a, :3, :3].T for f in shared])
            seam_turn.append(max(angle_deg(R_full.T @ c2w[f, :3, :3] @ T[f - a, :3, :3].T) for f in shared))
            if own_level is not None:
                A = yaw_part(R_full @ own_level.T) @ own_level
                level_change.append(angle_deg(A @ R_full.T))
            else:
                A = R_full
            offsets = np.array([c2w[f, :3, 3] - s * A @ T[f - a, :3, 3] for f in shared])
            t = np.median(offsets, axis=0)
            seam_shift.append(float(np.max(np.linalg.norm(offsets - t, axis=1))))
        levelled.append(own_level is not None)
        for k in range(a, b):
            if r > 0 and k < runs[r - 1][1]:
                continue  # shared frames keep the earlier run's version
            i = k - a
            c2w[k, :3, :3] = A @ T[i, :3, :3]
            c2w[k, :3, 3] = s * A @ T[i, :3, 3] + t
            c2w[k, 3, 3] = 1.0
            depth[k] = vs.depth[i] * s
            conf[k] = vs.conf[i]

    anchor_resid = [abs(log_s[r] - v) for r, v, _ in anchors]
    info = {"runs_levelled_on_floor": f"{sum(levelled)}/{len(runs)}",
            "level_change_deg_median": round(float(np.median(level_change)), 2) if level_change else None,
            "seam_rotation_spread_deg_median": round(float(np.median(seam_turn)), 2) if seam_turn else None,
            "seam_position_spread_m_median": round(float(np.median(seam_shift)), 3) if seam_shift else None,
            "seam_depth_ratio_spread_median": round(float(np.median(seam_spread)), 4) if seam_spread else None,
            "metric_anchors": len(anchors),
            "metric_anchor_scatter": round(float(1.4826 * np.median(anchor_resid)), 4)}
    return Chain(c2w, depth, conf, K, runs, info)


def load_video(video, work_dir, fx_over_width: float | None = None) -> VideoCapture:
    """`fx_over_width`: focal length in units of the upright frame width, when known (e.g. from
    metadata); otherwise estimated from the video."""
    video, work_dir = Path(video), Path(work_dir)
    frames = extract_keyframes(video, work_dir / "frames")
    n = len(frames)
    runs = runs_for(n)
    cache = pose_cache(work_dir)
    views = [_views(frames, a, b, cache) for a, b in runs]
    h, w = views[0].depth.shape[1:]
    run_fx = [float(np.median(v.K[:, 0, 0] / w)) for v in views]
    focal_given = fx_over_width is not None
    if not focal_given:
        fx_over_width = float(np.median(np.concatenate([v.K[:, 0, 0] / w for v in views])))
    K = camera_matrix(fx_over_width, w, h)

    sample = list(range(0, n, METRIC_SAMPLE_EVERY))
    metric = _metric(frames, sample, fx_over_width, work_dir / "da3" / f"{da3.METRIC_MODEL.split('/')[-1]}-{PROCESS_RES}")
    chain = chain_runs(views, runs, K, metric)

    focal_sigma = FOCAL_SIGMA_GIVEN if focal_given else FOCAL_SIGMA_ESTIMATED
    scale_sigma = float(np.linalg.norm([METRIC_SIGMA / np.sqrt(chain.info["metric_anchors"]), focal_sigma, METRIC_BIAS_SIGMA]))
    confidences = np.zeros(chain.depth.shape, np.uint8)
    for i in range(n):
        confidences[i][_confident(chain.conf[i])] = 2

    info = {"frames": n, "runs": len(runs), "keyframe_fps": KEYFRAME_FPS, "depth_size": [w, h],
            "camera": "DA3 ray output" if RAY_POSE else "DA3 camera head",
            "fx_over_width": round(fx_over_width, 4), "focal_source": "given" if focal_given else "DA3 median",
            "fx_over_width_per_run": [round(v, 4) for v in run_fx],
            "scale_sigma": round(scale_sigma, 4), **chain.info}
    (work_dir / "video_capture.json").write_text(json.dumps(info, indent=2))
    timestamps = np.arange(n) / KEYFRAME_FPS
    Ks = np.repeat(K[None], n, axis=0)
    return VideoCapture(video, frames, timestamps, chain.c2w[:, :3, 3].copy(), chain.c2w[:, :3, :3].copy(),
                        chain.depth, confidences, Ks, scale_sigma, info)
