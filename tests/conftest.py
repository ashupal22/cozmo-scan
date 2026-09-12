from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]

# Camera looking straight down: optical axis = world -y, image right = world +x, image down = world +z
LOOK_DOWN = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])


def make_stray_capture(root: Path, positions, rotation=LOOK_DOWN, depth_m=1.5) -> Path:
    """A tiny synthetic Stray Scanner export: constant depth, confidence 2 except the top row."""
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    (root / "rgb.mp4").write_bytes(b"")  # unreadable on purpose: size falls back to 1920x1440
    (root / "camera_matrix.csv").write_text("1600.0, 0.0, 959.5\n0.0, 1600.0, 719.5\n0.0, 0.0, 1.0")
    (root / "imu.csv").write_text("timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n"
                                  "100.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0\n")
    q = Rotation.from_matrix(rotation).as_quat()
    lines = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"]
    for i, p in enumerate(positions):
        lines.append(f"{100 + i / 60:.6f}, {i:06d}, {p[0]}, {p[1]}, {p[2]}, "
                     f"{q[0]}, {q[1]}, {q[2]}, {q[3]}, 1600.0, 1600.0, 959.5, 719.5, , ")
        depth = np.full((192, 256), int(round(depth_m * 1000)), np.uint16)
        conf = np.full((192, 256), 2, np.uint8)
        conf[0, :] = 0
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), depth)
        cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), conf)
    (root / "odometry.csv").write_text("\n".join(lines) + "\n")
    return root


@pytest.fixture
def stray_dir(tmp_path) -> Path:
    # camera 1.5 m above the floor (y = 0); third frame jumps 39 cm
    return make_stray_capture(tmp_path / "cap", positions=[(0.0, 1.5, 0.0), (0.01, 1.5, 0.0), (0.40, 1.5, 0.0)])
