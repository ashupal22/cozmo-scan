"""Load one ARKitScenes raw walk: iPad Pro LiDAR in the same format as Stray Scanner, plus depth
rendered from Faro laser scans (ground truth) for a subset of frames.

Folder, as written by scripts/fetch_external.py:
    lowres_depth/<video>_<t>.png               uint16 mm, 256x192, 60 fps
    confidence/<video>_<t>.png                 uint8 0..2
    lowres_wide/<video>_<t>.png                RGB 256x192
    lowres_wide_intrinsics/<video>_<t>.pincam  "width height fx fy cx cy"
    lowres_wide.traj                           10 fps: "t  rx ry rz  tx ty tz"
    highres_depth/<video>_<t>.png              uint16 mm, 1920x1440, laser-rendered, 10 fps,
                                               same timestamps as lowres_depth

Conventions, from ARKitScenes' own threedod/benchmark_scripts/utils/tenFpsDataLoader.py:
- A trajectory line is world-to-camera (axis-angle rotation, translation). Its inverse maps OpenCV
  camera coordinates to the world, as the Stray Scanner odometry does.
- The trajectory is 10 fps and depth is 60 fps, so poses are interpolated (slerp for rotation,
  linear for position). Frames outside the trajectory's time range are dropped.
- Ground-truth depth uses the low-resolution intrinsics scaled to its own image size.
- The trajectory world is z-up. On our walks, 15% of surface normals point along z against about 2%
  along x or y, and floors sit below the cameras along +z. The rest of cozmo uses y-up (like Stray
  Scanner), so every pose is rotated by Z_UP_TO_Y_UP: (x, y, z) -> (x, z, -y).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from cozmo.ingest.stray import CaptureError, Intrinsics


Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


def _stamp(path: Path) -> str:
    """'41069048_5002.101.png' -> '5002.101'"""
    return path.stem.rsplit("_", 1)[1]


class ARKitScenesWalk:
    """One ARKitScenes walk with the same frame interface as StrayCapture, plus ground truth."""

    def __init__(self, path):
        self.path = Path(path)
        self.video_id = self.path.name
        self.warnings: list[str] = []
        for name in ("lowres_depth", "confidence", "lowres_wide_intrinsics"):
            if not (self.path / name).is_dir():
                raise CaptureError(f"{self.path}: missing {name}/")
        traj_file = self.path / "lowres_wide.traj"
        if not traj_file.is_file():
            raise CaptureError(f"{self.path}: missing lowres_wide.traj")

        traj = np.loadtxt(traj_file, ndmin=2)
        if traj.shape[1] != 7 or len(traj) < 2:
            raise CaptureError("lowres_wide.traj needs 7 columns and at least 2 rows")
        traj = traj[np.argsort(traj[:, 0])]
        cam_from_world = Rotation.from_rotvec(traj[:, 1:4])  # z-up world
        to_y_up = Rotation.from_matrix(Z_UP_TO_Y_UP)
        world_from_cam = to_y_up * cam_from_world.inv()
        self._traj_t = traj[:, 0]
        self._traj_rot = Slerp(self._traj_t, world_from_cam)
        self._traj_pos = to_y_up.apply(-cam_from_world.inv().apply(traj[:, 4:7]))

        confidence = {_stamp(p) for p in (self.path / "confidence").glob("*.png")}
        pincam = {_stamp(p) for p in (self.path / "lowres_wide_intrinsics").glob("*.pincam")}
        stamps = sorted((s for s in map(_stamp, (self.path / "lowres_depth").glob("*.png"))
                         if s in confidence and s in pincam), key=float)
        t = np.array([float(s) for s in stamps])
        inside = (t >= self._traj_t[0]) & (t <= self._traj_t[-1])
        self.dropped_outside_trajectory = int((~inside).sum())
        self._stamps = [s for s, keep in zip(stamps, inside) if keep]
        if not self._stamps:
            raise CaptureError(f"{self.path}: no depth frames inside the trajectory's time range")

        self.timestamps = t[inside]
        self._rotations = self._traj_rot(self.timestamps)
        self.positions = np.column_stack([np.interp(self.timestamps, self._traj_t, self._traj_pos[:, k])
                                          for k in range(3)])

        gt_dir = self.path / "highres_depth"
        gt = {_stamp(p) for p in gt_dir.glob("*.png")} if gt_dir.is_dir() else set()
        self.gt_indices = np.array([i for i, s in enumerate(self._stamps) if s in gt], dtype=int)

        first = self.depth(0)
        self.depth_size = (first.shape[1], first.shape[0])
        self.gt_size = None
        if len(self.gt_indices):
            g = self._read("highres_depth", int(self.gt_indices[0]))
            self.gt_size = (g.shape[1], g.shape[0])

    # ---- files -----------------------------------------------------------------------
    def _file(self, folder: str, i: int, ext: str = ".png") -> Path:
        return self.path / folder / f"{self.video_id}_{self._stamps[i]}{ext}"

    def _read(self, folder: str, i: int) -> np.ndarray:
        img = cv2.imread(str(self._file(folder, i)), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise CaptureError(f"cannot read {folder} frame {self._stamps[i]}")
        return img

    # ---- frame interface shared with StrayCapture ---------------------------------------
    def __len__(self) -> int:
        return len(self._stamps)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def fps(self) -> float:
        return float(1.0 / np.median(np.diff(self.timestamps))) if len(self) > 1 else float("nan")

    def rotation(self, i: int) -> np.ndarray:
        return self._rotations[i].as_matrix()

    def pose(self, i: int) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation(i)
        T[:3, 3] = self.positions[i]
        return T

    def intrinsics(self, i: int, image: str = "depth") -> Intrinsics:
        w, h, fx, fy, cx, cy = np.loadtxt(self._file("lowres_wide_intrinsics", i, ".pincam"))
        k = Intrinsics(float(fx), float(fy), float(cx), float(cy), int(w), int(h))
        if image in ("depth", "rgb"):
            return k if (k.width, k.height) == self.depth_size else k.scaled_to(*self.depth_size)
        if image == "gt":
            if self.gt_size is None:
                raise CaptureError(f"{self.path}: no ground-truth depth")
            return k.scaled_to(*self.gt_size)
        raise ValueError("image must be 'depth', 'rgb' or 'gt'")

    def depth(self, i: int) -> np.ndarray:
        return self._read("lowres_depth", i).astype(np.float32) / 1000.0

    def confidence(self, i: int) -> np.ndarray:
        return self._read("confidence", i)

    def rgb(self, i: int) -> np.ndarray:
        return self._read("lowres_wide", i)

    # ---- ground truth ------------------------------------------------------------------
    def has_gt(self, i: int) -> bool:
        return bool(np.isin(i, self.gt_indices))

    def gt_depth(self, i: int) -> np.ndarray | None:
        """Laser-rendered depth in metres (0 where unknown), or None if this frame has none."""
        if not self.has_gt(i):
            return None
        return self._read("highres_depth", i).astype(np.float32) / 1000.0
