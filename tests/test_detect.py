import shutil

import pytest

from cozmo.ingest.detect import TierError, detect_tier
from cozmo.ingest.photos import photo_rooms, photo_warnings


def test_lidar_export(stray_dir):
    assert detect_tier(stray_dir) == "lidar"


def test_video_file(tmp_path):
    clip = tmp_path / "walk.MOV"
    clip.write_bytes(b"x")
    assert detect_tier(clip) == "video"


def test_folder_with_one_video(tmp_path):
    (tmp_path / "walk.mp4").write_bytes(b"x")
    assert detect_tier(tmp_path) == "video"


def test_photo_room_folders(tmp_path):
    for room, n in (("kitchen", 3), ("hall", 1)):
        (tmp_path / room).mkdir()
        for i in range(n):
            (tmp_path / room / f"{i}.jpg").write_bytes(b"x")
    assert detect_tier(tmp_path) == "photo"
    rooms = photo_rooms(tmp_path)
    assert {k: len(v) for k, v in rooms.items()} == {"hall": 1, "kitchen": 3}
    assert photo_warnings(rooms) == ["room 'hall' has 1 photos; the capture guide asks for 2 to 8"]


def test_incomplete_stray_export_names_what_is_missing(stray_dir):
    shutil.rmtree(stray_dir / "depth")
    with pytest.raises(TierError, match="depth"):
        detect_tier(stray_dir)


def test_several_videos_is_ambiguous(tmp_path):
    (tmp_path / "a.mov").write_bytes(b"x")
    (tmp_path / "b.mov").write_bytes(b"x")
    with pytest.raises(TierError, match="2 video"):
        detect_tier(tmp_path)


def test_empty_folder(tmp_path):
    with pytest.raises(TierError, match="no LiDAR export"):
        detect_tier(tmp_path)


def test_missing_path(tmp_path):
    with pytest.raises(TierError, match="does not exist"):
        detect_tier(tmp_path / "nope")
