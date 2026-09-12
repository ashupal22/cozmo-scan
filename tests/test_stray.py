import cv2
import numpy as np
import pytest

from cozmo.ingest.stray import CaptureError, Intrinsics, StrayCapture, upright_rotate_code
from tests.conftest import LOOK_DOWN


def test_loads_frames_and_sizes(stray_dir):
    cap = StrayCapture(stray_dir)
    assert len(cap) == 3
    assert cap.depth_size == (256, 192)
    assert cap.rgb_size == (1920, 1440)
    assert any("assuming the documented 1920x1440" in w for w in cap.warnings)
    assert cap.fps == pytest.approx(60.0, rel=1e-3)


def test_depth_intrinsics_are_scaled_per_frame(stray_dir):
    k = StrayCapture(stray_dir).intrinsics(0, "depth")
    assert (k.width, k.height) == (256, 192)
    assert k.fx == pytest.approx(1600 * 256 / 1920)
    assert k.cx == pytest.approx(127.5)  # centre of a 256-pixel-wide image
    assert k.cy == pytest.approx(95.5)


def test_intrinsics_scaling_round_trips():
    k = Intrinsics(1600.0, 1600.0, 955.4, 717.8, 1920, 1440)
    back = k.scaled_to(256, 192).scaled_to(1920, 1440)
    assert np.allclose([back.fx, back.fy, back.cx, back.cy], [k.fx, k.fy, k.cx, k.cy])


def test_points_from_a_downward_camera_lie_on_the_floor(stray_dir):
    pts = StrayCapture(stray_dir).points(0)
    assert pts.shape == (256 * 191, 3)  # the top row has confidence 0
    assert np.allclose(pts[:, 1], 0.0, atol=1e-9)
    assert pts[:, 0].max() == pytest.approx((255 - 127.5) / (1600 * 256 / 1920) * 1.5)
    assert pts[:, 0].mean() == pytest.approx(0.0, abs=1e-9)


def test_low_confidence_can_be_included(stray_dir):
    assert StrayCapture(stray_dir).points(0, min_confidence=0).shape[0] == 256 * 192


def test_pose_jumps_and_path_length(stray_dir):
    cap = StrayCapture(stray_dir)
    jumps = cap.pose_jumps()
    assert [(a, b) for a, b, _ in jumps] == [(1, 2)]
    assert jumps[0][2] == pytest.approx(0.39)
    assert cap.path_length_m() == pytest.approx(0.40)


def test_summary_reports_the_jump(stray_dir):
    s = StrayCapture(stray_dir).summary()
    assert s["tier"] == "lidar" and s["frames"] == 3
    assert s["pose_jumps"] == [{"from": 1, "to": 2, "m": 0.39}]
    assert s["high_confidence_fraction"] == pytest.approx(191 / 192, abs=1e-3)


def test_upright_rotation():
    assert upright_rotate_code(np.eye(3)) == cv2.ROTATE_180  # image down points at world up
    right_is_up = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    assert upright_rotate_code(right_is_up) == cv2.ROTATE_90_COUNTERCLOCKWISE
    assert upright_rotate_code(LOOK_DOWN) is None  # no 'up' in the image


def test_missing_odometry_is_a_clear_error(stray_dir):
    (stray_dir / "odometry.csv").unlink()
    with pytest.raises(CaptureError, match="odometry.csv"):
        StrayCapture(stray_dir)


def test_missing_depth_frame_is_a_clear_error(stray_dir):
    (stray_dir / "depth" / "000001.png").unlink()
    with pytest.raises(CaptureError, match="depth/000001.png"):
        StrayCapture(stray_dir)
