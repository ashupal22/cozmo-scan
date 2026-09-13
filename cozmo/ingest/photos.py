"""Photo tier input: one folder per room, 2 to 8 stills each."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cozmo.ingest.detect import TierError, images_in

MIN_PHOTOS, MAX_PHOTOS = 2, 8
EXIF_IFD = 0x8769
FOCAL_35MM_TAG = 0xA405               # FocalLengthIn35mmFilm, written by iPhones
FULL_FRAME_DIAGONAL_MM = float(np.hypot(36.0, 24.0))


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


def load_photo(path) -> tuple[np.ndarray, float | None]:
    """RGB uint8 image turned upright by its EXIF orientation, and its focal length / image width from EXIF,
    or None when the photo does not say.

    iPhones write the focal length as a 35 mm equivalent, which by the usual definition matches the diagonal
    field of view of a 36 x 24 mm frame, so in pixels f = f35 x image diagonal / 43.27 mm. HEIC photos (the
    iPhone default) are read through pillow-heif."""
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as im:
        f35 = im.getexif().get_ifd(EXIF_IFD).get(FOCAL_35MM_TAG)
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    h, w = rgb.shape[:2]
    fx_over_width = float(f35) * float(np.hypot(w, h)) / (FULL_FRAME_DIAGONAL_MM * w) if f35 else None
    return rgb, fx_over_width
