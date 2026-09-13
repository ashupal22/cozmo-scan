"""iPhone videos store portrait clips as landscape pixels plus a rotation flag. Key frames must come out upright."""
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from cozmo.ingest.detect import detect_tier
from cozmo.video.capture import extract_keyframes

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_rotation_flag_is_applied(tmp_path):
    upright = np.zeros((160, 120, 3), np.uint8)
    upright[:40, :30] = 255                                   # marker at the top left of the upright picture
    src = tmp_path / "frames"
    src.mkdir()
    for k in range(48):                                       # 8 s at 6 fps: enough key frames
        cv2.imwrite(str(src / f"{k:03d}.png"), cv2.rotate(upright, cv2.ROTATE_90_COUNTERCLOCKWISE))  # stored landscape
    raw = tmp_path / "landscape.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "6", "-i", str(src / "%03d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(raw)], check=True)
    clip = tmp_path / "IMG_0001.MOV"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "-90", "-i", str(raw), "-c", "copy", str(clip)],
                   check=True)
    assert detect_tier(clip) == "video"
    frames = extract_keyframes(clip, tmp_path / "keys")
    img = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    assert h > w                                               # portrait, as the phone was held
    assert img[: h // 4, : w // 4].mean() > 200 and img[-h // 4:, -w // 4:].mean() < 50
