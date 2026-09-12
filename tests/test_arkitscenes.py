from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cozmo.geometry.fusion import frame_points, fuse
from cozmo.geometry.planes import find_floors
from cozmo.ingest.arkitscenes import ARKitScenesWalk
from cozmo.ingest.stray import CaptureError
from tests.conftest import LOOK_DOWN

VIDEO = "123"
# ARKitScenes stores poses in a z-up world; cozmo's world is y-up. Written out independently of the loader.
Y_UP_FROM_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def make_walk(root: Path) -> Path:
    """Synthetic walk: depth at 60 fps from t=10.000 s, trajectory at 10 fps from 10.1 to 10.4 s,
    camera looking straight down from 1.5 m while sliding along +x at 1 m/s. The trajectory file
    is written in ARKitScenes' z-up world, as the real data is."""
    for d in ("lowres_depth", "confidence", "lowres_wide", "lowres_wide_intrinsics", "highres_depth"):
        (root / d).mkdir(parents=True)
    stamps = [f"{10 + k / 60:.3f}" for k in range(31)]
    for k, s in enumerate(stamps):
        cv2.imwrite(str(root / "lowres_depth" / f"{VIDEO}_{s}.png"), np.full((192, 256), 1500, np.uint16))
        cv2.imwrite(str(root / "confidence" / f"{VIDEO}_{s}.png"), np.full((192, 256), 2, np.uint8))
        cv2.imwrite(str(root / "lowres_wide" / f"{VIDEO}_{s}.png"), np.zeros((192, 256, 3), np.uint8))
        (root / "lowres_wide_intrinsics" / f"{VIDEO}_{s}.pincam").write_text("256 192 213.333 213.333 127.5 95.5")
        if k % 6 == 0:
            cv2.imwrite(str(root / "highres_depth" / f"{VIDEO}_{s}.png"), np.full((1440, 1920), 1500, np.uint16))
    world_from_cam_z_up = Y_UP_FROM_Z_UP.T @ LOOK_DOWN
    cam_from_world = world_from_cam_z_up.T
    rotvec = Rotation.from_matrix(cam_from_world).as_rotvec()
    lines = []
    for t in (10.1, 10.2, 10.3, 10.4):
        position_z_up = Y_UP_FROM_Z_UP.T @ np.array([t - 10.1, 1.5, 0.0])
        translation = -cam_from_world @ position_z_up
        lines.append(" ".join(f"{v:.8f}" for v in [t, *rotvec, *translation]))
    (root / "lowres_wide.traj").write_text("\n".join(lines) + "\n")
    return root


@pytest.fixture
def walk_dir(tmp_path):
    return make_walk(tmp_path / VIDEO)


def test_frames_outside_the_trajectory_are_dropped(walk_dir):
    walk = ARKitScenesWalk(walk_dir)
    assert walk.timestamps[0] >= 10.1 and walk.timestamps[-1] <= 10.4
    assert walk.dropped_outside_trajectory + len(walk) == 31


def test_pose_is_inverted_and_interpolated(walk_dir):
    walk = ARKitScenesWalk(walk_dir)
    i = int(np.argmin(np.abs(walk.timestamps - 10.25)))
    assert np.allclose(walk.rotation(i), LOOK_DOWN, atol=1e-6)
    assert walk.positions[i] == pytest.approx([walk.timestamps[i] - 10.1, 1.5, 0.0], abs=1e-6)


def test_device_and_ground_truth_depth_both_land_on_the_floor(walk_dir):
    walk = ARKitScenesWalk(walk_dir)
    xyz, normals = frame_points(walk, 0, stride=1)
    assert np.allclose(xyz[:, 1], 0.0, atol=1e-5)
    assert len(walk.gt_indices) > 0 and walk.gt_size == (1920, 1440)
    assert walk.intrinsics(int(walk.gt_indices[0]), "gt").fx == pytest.approx(213.333 * 1920 / 256)
    gt = fuse(walk, walk.gt_indices, ground_truth=True, stride=8)
    assert find_floors(gt)[0].height == pytest.approx(0.0, abs=1e-3)


def test_frames_without_ground_truth_return_nothing(walk_dir):
    walk = ARKitScenesWalk(walk_dir)
    no_gt = next(i for i in range(len(walk)) if not walk.has_gt(i))
    assert walk.gt_depth(no_gt) is None
    assert frame_points(walk, no_gt, ground_truth=True)[0].shape == (0, 3)


def test_missing_trajectory_is_a_clear_error(walk_dir):
    (walk_dir / "lowres_wide.traj").unlink()
    with pytest.raises(CaptureError, match="lowres_wide.traj"):
        ARKitScenesWalk(walk_dir)
