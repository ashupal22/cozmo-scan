"""Decide which input tier a path is: a LiDAR export, a video clip, or photo folders."""
from __future__ import annotations

from pathlib import Path

VIDEO_EXT = {".mov", ".mp4", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".heic", ".heif", ".png"}
STRAY_FILES = ("odometry.csv", "camera_matrix.csv", "rgb.mp4")
STRAY_DIRS = ("depth", "confidence")


class TierError(ValueError):
    """The path does not look like any supported capture."""


def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def detect_tier(path) -> str:
    """Return "lidar", "video" or "photo" for a capture path, or raise TierError."""
    p = Path(path)
    if not p.exists():
        raise TierError(f"{p} does not exist")
    if p.is_file():
        if p.suffix.lower() in VIDEO_EXT:
            return "video"
        raise TierError(f"{p.name} is not a video ({', '.join(sorted(VIDEO_EXT))}); "
                        "pass a video file or a capture folder")

    has = {n: (p / n).is_file() for n in STRAY_FILES} | {n: (p / n).is_dir() for n in STRAY_DIRS}
    if all(has.values()):
        return "lidar"
    if has["odometry.csv"] or has["depth"] or has["confidence"]:
        missing = [n for n, ok in has.items() if not ok]
        raise TierError(f"{p} looks like a Stray Scanner export but is missing: {', '.join(missing)}")

    videos = sorted(c for c in p.iterdir() if c.is_file() and c.suffix.lower() in VIDEO_EXT)
    if len(videos) > 1:
        raise TierError(f"{p} holds {len(videos)} video files; pass one video file directly")
    photos_here = images_in(p)
    room_dirs = [c for c in p.iterdir() if c.is_dir() and not c.name.startswith(".") and images_in(c)]
    if len(videos) == 1 and not photos_here and not room_dirs:
        return "video"
    if photos_here or room_dirs:
        return "photo"
    raise TierError(f"{p}: no LiDAR export, video or photos found")
