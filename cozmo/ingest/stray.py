"""Load a Stray Scanner export (the LiDAR tier).

Folder layout (https://docs.strayrobots.io/apps/scanner/format.html):
    rgb.mp4                HEVC video, 1920x1440
    depth/NNNNNN.png       uint16 depth in millimetres, 256x192
    confidence/NNNNNN.png  uint8 confidence: 0 low, 1 medium, 2 high
    odometry.csv           per frame: timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy
    imu.csv                accelerometer and gyroscope
    camera_matrix.csv      intrinsics

Conventions, checked on our own captures (explore/probe_basic.py):
- The odometry pose maps OpenCV camera coordinates (x right, y down, z forward) to a
  gravity-aligned world frame with y up. Back-projecting with that pose puts 10-13% of all
  points into the two strongest 1 cm height bins (the floor); using the ARKit camera
  convention instead drops that to 2-3%.
- Intrinsics in odometry.csv are per frame and refer to the RGB image. fx varies by up to
  about 2% within one capture, so the per-frame values are always used.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ODOMETRY_COLUMNS = ["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy"]
DOCUMENTED_RGB_SIZE = (1920, 1440)
POSE_JUMP_M = 0.10  # a normal step at 60 fps is under 3 cm


class CaptureError(ValueError):
    """The folder is not a usable Stray Scanner export."""


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for a `width` x `height` image, OpenCV convention
    (pixel centres at integer coordinates)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5,
                          width, height)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


def upright_rotate_code(world_from_cam: np.ndarray, min_tilt: float = 0.3):
    """cv2.rotate code that turns a frame upright, or None when no rotation is needed or the
    camera points almost straight up or down (no reliable 'up' direction in the image)."""
    up = world_from_cam.T @ np.array([0.0, 1.0, 0.0])  # world up, in camera coordinates
    ux, uy = up[0], up[1]  # image right, image down
    if np.hypot(ux, uy) < min_tilt:
        return None
    if abs(uy) >= abs(ux):
        return None if uy < 0 else cv2.ROTATE_180
    return cv2.ROTATE_90_COUNTERCLOCKWISE if ux > 0 else cv2.ROTATE_90_CLOCKWISE


class StrayCapture:
    """One Stray Scanner export. Frame index i runs over the rows of odometry.csv."""

    def __init__(self, path):
        self.path = Path(path)
        self.warnings: list[str] = []

        missing = [n for n in ("odometry.csv", "camera_matrix.csv", "rgb.mp4") if not (self.path / n).is_file()]
        missing += [n + "/" for n in ("depth", "confidence") if not (self.path / n).is_dir()]
        if missing:
            raise CaptureError(f"{self.path}: not a complete Stray Scanner export, missing {', '.join(missing)}")

        odo = pd.read_csv(self.path / "odometry.csv", skipinitialspace=True)
        odo.columns = [c.strip() for c in odo.columns]
        lacking = [c for c in ODOMETRY_COLUMNS if c not in odo.columns]
        if lacking:
            raise CaptureError(f"odometry.csv is missing columns: {', '.join(lacking)}")
        if odo.empty:
            raise CaptureError("odometry.csv has no frames")
        if odo[ODOMETRY_COLUMNS].isna().any().any():
            raise CaptureError("odometry.csv has empty pose or intrinsics values")

        self.timestamps = odo["timestamp"].to_numpy(float)
        self.frame_ids = odo["frame"].to_numpy(int)
        self.positions = odo[["x", "y", "z"]].to_numpy(float)
        self.quaternions = odo[["qx", "qy", "qz", "qw"]].to_numpy(float)
        self._rgb_k = odo[["fx", "fy", "cx", "cy"]].to_numpy(float)
        self._rotations = Rotation.from_quat(self.quaternions)

        if np.any(np.diff(self.timestamps) <= 0):
            raise CaptureError("odometry.csv timestamps are not strictly increasing")
        for kind in ("depth", "confidence"):
            absent = [f for f in self.frame_ids if not self._frame_file(kind, f).is_file()]
            if absent:
                raise CaptureError(f"{len(absent)} {kind} frames listed in odometry.csv are missing "
                                   f"(first: {kind}/{absent[0]:06d}.png)")

        first = self.depth(0)
        self.depth_size = (first.shape[1], first.shape[0])
        self.rgb_size = self._probe_rgb_size()

    # ---- files -----------------------------------------------------------------------
    def _frame_file(self, kind: str, frame_id) -> Path:
        return self.path / kind / f"{int(frame_id):06d}.png"

    def _read_png(self, kind: str, i: int) -> np.ndarray:
        img = cv2.imread(str(self._frame_file(kind, self.frame_ids[i])), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise CaptureError(f"cannot read {kind} frame {self.frame_ids[i]:06d}")
        return img

    def _probe_rgb_size(self) -> tuple[int, int]:
        video = self.path / "rgb.mp4"
        cap = cv2.VideoCapture(str(video))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return w, h
        if shutil.which("ffprobe"):
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                  "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video)],
                                 capture_output=True, text=True)
            try:
                w, h = (int(v) for v in out.stdout.strip().splitlines()[0].split("x")[:2])
                return w, h
            except (ValueError, IndexError):
                pass
        self.warnings.append("could not read the rgb.mp4 frame size; assuming the documented "
                             f"{DOCUMENTED_RGB_SIZE[0]}x{DOCUMENTED_RGB_SIZE[1]}")
        return DOCUMENTED_RGB_SIZE

    # ---- per-frame data ----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.frame_ids)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def fps(self) -> float:
        return float(1.0 / np.median(np.diff(self.timestamps))) if len(self) > 1 else float("nan")

    def intrinsics(self, i: int, image: str = "rgb") -> Intrinsics:
        fx, fy, cx, cy = self._rgb_k[i]
        k = Intrinsics(fx, fy, cx, cy, *self.rgb_size)
        if image == "rgb":
            return k
        if image == "depth":
            return k.scaled_to(*self.depth_size)
        raise ValueError("image must be 'rgb' or 'depth'")

    def rotation(self, i: int) -> np.ndarray:
        """3x3 rotation, camera (OpenCV convention) to world."""
        return self._rotations[i].as_matrix()

    def pose(self, i: int) -> np.ndarray:
        """4x4 transform, camera (OpenCV convention) to world."""
        T = np.eye(4)
        T[:3, :3] = self.rotation(i)
        T[:3, 3] = self.positions[i]
        return T

    def depth(self, i: int) -> np.ndarray:
        """Depth in metres, float32; 0 where the sensor gave no value."""
        img = self._read_png("depth", i)
        if img.dtype != np.uint16:
            raise CaptureError(f"depth frame {self.frame_ids[i]:06d} is {img.dtype}, expected uint16 millimetres")
        return img.astype(np.float32) / 1000.0

    def confidence(self, i: int) -> np.ndarray:
        return self._read_png("confidence", i)

    def points(self, i: int, min_confidence: int = 2, min_depth: float = 0.2, max_depth: float = 5.0,
               stride: int = 1) -> np.ndarray:
        """World-frame 3D points (N, 3) from one depth frame."""
        z = self.depth(i)[::stride, ::stride]
        c = self.confidence(i)[::stride, ::stride]
        k = self.intrinsics(i, "depth")
        v, u = np.nonzero((c >= min_confidence) & (z > min_depth) & (z < max_depth))
        zz = z[v, u].astype(np.float64)
        cam = np.stack([(u * stride - k.cx) / k.fx * zz, (v * stride - k.cy) / k.fy * zz, zz], axis=1)
        return cam @ self.rotation(i).T + self.positions[i]

    def upright_rotate_code(self, i: int):
        return upright_rotate_code(self.rotation(i))

    def imu(self) -> pd.DataFrame:
        df = pd.read_csv(self.path / "imu.csv", skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        return df

    # ---- whole-capture facts -----------------------------------------------------------
    def step_lengths(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.positions, axis=0), axis=1)

    def path_length_m(self) -> float:
        return float(self.step_lengths().sum())

    def pose_jumps(self, threshold_m: float = POSE_JUMP_M) -> list[tuple[int, int, float]]:
        """(from_index, to_index, metres) for every frame-to-frame step above threshold_m."""
        steps = self.step_lengths()
        return [(int(i), int(i + 1), float(steps[i])) for i in np.nonzero(steps > threshold_m)[0]]

    def summary(self, confidence_samples: int = 20) -> dict:
        fx = self._rgb_k[:, 0]
        idx = np.linspace(0, len(self) - 1, min(confidence_samples, len(self))).astype(int)
        high = float(np.mean([np.mean(self.confidence(i) == 2) for i in idx]))
        extent = self.positions.max(0) - self.positions.min(0)
        return {
            "tier": "lidar",
            "path": str(self.path),
            "frames": len(self),
            "duration_s": round(self.duration_s, 2),
            "fps": round(self.fps, 2),
            "rgb_size": list(self.rgb_size),
            "depth_size": list(self.depth_size),
            "fx_range": [round(float(fx.min()), 2), round(float(fx.max()), 2)],
            "fx_variation_pct": round(float(100 * (fx.max() - fx.min()) / fx.mean()), 2),
            "path_length_m": round(self.path_length_m(), 2),
            "walk_extent_m": [round(float(e), 2) for e in extent],
            "start_to_end_m": round(float(np.linalg.norm(self.positions[-1] - self.positions[0])), 3),
            "pose_jumps": [{"from": a, "to": b, "m": round(d, 3)} for a, b, d in self.pose_jumps()],
            "high_confidence_fraction": round(high, 3),
            "warnings": list(self.warnings),
        }
