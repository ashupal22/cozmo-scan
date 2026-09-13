"""World-frame points with surface normals from a LiDAR capture.

Works with any capture object that provides len(), timestamps, positions, depth(i),
confidence(i), intrinsics(i, "depth") and rotation(i). StrayCapture does.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PointSet:
    xyz: np.ndarray       # (N, 3) float32, world frame, y up
    normal: np.ndarray    # (N, 3) float32, unit length, facing the camera that saw the point
    frame: np.ndarray     # (N,) int32, index of the source frame
    camera_y: np.ndarray  # (N,) float32, height of that camera

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, mask) -> "PointSet":
        return PointSet(self.xyz[mask], self.normal[mask], self.frame[mask], self.camera_y[mask])


def select_keyframes(capture, min_translation_m: float = 0.05, min_rotation_deg: float = 3.0,
                     max_gap_s: float = 0.5) -> np.ndarray:
    """Keep a frame when the camera moved, turned, or enough time passed since the last kept frame."""
    keep, last = [0], 0
    R_last = capture.rotation(0)
    for i in range(1, len(capture)):
        R_i = capture.rotation(i)
        moved = np.linalg.norm(capture.positions[i] - capture.positions[last])
        cos = np.clip((np.trace(R_last.T @ R_i) - 1.0) / 2.0, -1.0, 1.0)
        turned = np.degrees(np.arccos(cos))
        waited = capture.timestamps[i] - capture.timestamps[last]
        if moved >= min_translation_m or turned >= min_rotation_deg or waited >= max_gap_s:
            keep.append(i)
            last, R_last = i, R_i
    return np.array(keep)


def frame_points(capture, i: int, min_confidence: int = 2, min_depth: float = 0.3, max_depth: float = 4.5,
                 stride: int = 2, max_depth_step: float = 0.04,
                 ground_truth: bool = False, depth_offset_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(xyz, normal) in the world frame for one depth frame. Pixels on depth edges are dropped,
    because their normals are meaningless. With ground_truth=True the capture's laser-rendered
    depth is used instead of the device depth (ARKitScenes walks only). `depth_offset_m` is added
    to every device depth value: a calibrated sensor bias, never applied to laser depth."""
    if ground_truth:
        raw = capture.gt_depth(i)
        if raw is None:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
        z = cv2.medianBlur(raw, 5)
        c = np.full(z.shape, 2, np.uint8)
        k = capture.intrinsics(i, "gt")
    else:
        z = cv2.medianBlur(capture.depth(i), 5)
        if depth_offset_m:
            z = np.where(z > 0, z + np.float32(depth_offset_m), z).astype(np.float32)
        c = capture.confidence(i)
        k = capture.intrinsics(i, "depth")
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    P = np.stack([(u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z], axis=-1)

    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    Pc, zc = P[1:-1, 1:-1], z[1:-1, 1:-1]

    ok = (c[1:-1, 1:-1] >= min_confidence) & (zc > min_depth) & (zc < max_depth)
    ok &= (np.abs(dx[..., 2]) < max_depth_step * zc) & (np.abs(dy[..., 2]) < max_depth_step * zc)
    if stride > 1:
        grid = np.zeros_like(ok)
        grid[::stride, ::stride] = True
        ok &= grid

    n[(n * Pc).sum(-1) > 0] *= -1  # face the camera
    R = capture.rotation(i)
    xyz = (Pc[ok] @ R.T + capture.positions[i]).astype(np.float32)
    return xyz, (n[ok] @ R.T).astype(np.float32)


def fuse(capture, frames=None, **frame_kwargs) -> PointSet:
    """Points with normals from the given frames (default: keyframes)."""
    frames = select_keyframes(capture) if frames is None else np.asarray(frames)
    xyz, nrm, fid, cam_y = [], [], [], []
    for i in frames:
        p, n = frame_points(capture, int(i), **frame_kwargs)
        xyz.append(p)
        nrm.append(n)
        fid.append(np.full(len(p), i, np.int32))
        cam_y.append(np.full(len(p), capture.positions[i][1], np.float32))
    return PointSet(np.concatenate(xyz), np.concatenate(nrm), np.concatenate(fid), np.concatenate(cam_y))
