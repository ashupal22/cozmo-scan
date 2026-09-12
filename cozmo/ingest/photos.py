"""Photo tier input: one folder per room, 2 to 8 stills each."""
from __future__ import annotations

from pathlib import Path

from cozmo.ingest.detect import TierError, images_in

MIN_PHOTOS, MAX_PHOTOS = 2, 8


def photo_rooms(path) -> dict[str, list[Path]]:
    """Map room name to its photos. A folder of photos with no sub-folders counts as one room."""
    p = Path(path)
    direct = images_in(p)
    rooms = {c.name: images_in(c) for c in sorted(p.iterdir()) if c.is_dir() and not c.name.startswith(".")}
    rooms = {name: photos for name, photos in rooms.items() if photos}
    if direct and rooms:
        raise TierError(f"{p} mixes loose photos with room folders; put every photo inside a room folder")
    if direct:
        return {p.name: direct}
    if not rooms:
        raise TierError(f"{p}: no photos found")
    return rooms


def photo_warnings(rooms: dict[str, list[Path]]) -> list[str]:
    return [f"room '{name}' has {len(photos)} photos; the capture guide asks for {MIN_PHOTOS} to {MAX_PHOTOS}"
            for name, photos in rooms.items() if not MIN_PHOTOS <= len(photos) <= MAX_PHOTOS]
