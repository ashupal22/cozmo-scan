"""Fusion: a calibrated device depth offset moves device points, never laser points."""
import numpy as np
import pytest

from cozmo.geometry.fusion import frame_points
from cozmo.ingest.stray import Intrinsics

W, H = 64, 48


class FlatWallCapture:
    """A camera 2 m in front of a flat wall, looking straight at it (OpenCV axes = world axes)."""
    positions = np.zeros((1, 3))
    timestamps = np.zeros(1)

    def __len__(self):
        return 1

    def rotation(self, i):
        return np.eye(3)

    def depth(self, i):
        return np.full((H, W), 2.0, np.float32)

    def confidence(self, i):
        return np.full((H, W), 2, np.uint8)

    def intrinsics(self, i, image="depth"):
        return Intrinsics(50.0, 50.0, (W - 1) / 2, (H - 1) / 2, W, H)

    def gt_depth(self, i):
        return np.full((H, W), 2.0, np.float32)


def test_offset_moves_device_points_along_depth():
    cap = FlatWallCapture()
    plain, _ = frame_points(cap, 0, stride=1)
    shifted, _ = frame_points(cap, 0, stride=1, depth_offset_m=0.013)
    assert np.allclose(plain[:, 2], 2.0, atol=1e-6)
    assert np.allclose(shifted[:, 2], 2.013, atol=1e-6)
    # x and y scale with depth along each ray: the point stays on its pixel's ray
    assert np.allclose(shifted[:, :2], plain[:, :2] * 2.013 / 2.0, atol=1e-6)


def test_offset_is_never_applied_to_laser_depth():
    cap = FlatWallCapture()
    laser, _ = frame_points(cap, 0, stride=1, ground_truth=True, depth_offset_m=0.013)
    assert np.allclose(laser[:, 2], 2.0, atol=1e-6)
