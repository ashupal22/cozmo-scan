"""Photo loading: EXIF orientation and the 35 mm-equivalent focal length iPhones write."""
import numpy as np
import pytest
from PIL import Image

from cozmo.ingest.photos import EXIF_IFD, FOCAL_35MM_TAG, load_photo


def write_photo(path, fmt, orientation=6, f35=26):
    img = np.zeros((300, 400, 3), np.uint8)       # landscape as stored by the sensor
    img[:50, :100] = (255, 0, 0)                  # red block top-left, to check the turn
    exif = Image.Exif()
    exif[0x0112] = orientation                    # 6: turn 90 degrees clockwise to view
    if f35 is not None:
        exif.get_ifd(EXIF_IFD)[FOCAL_35MM_TAG] = f35
    Image.fromarray(img).save(path, format=fmt, exif=exif.tobytes())


@pytest.mark.parametrize("fmt, suffix", [("JPEG", ".jpg"), ("HEIF", ".heic")])
def test_upright_and_focal_from_exif(tmp_path, fmt, suffix):
    if fmt == "HEIF":
        pillow_heif = pytest.importorskip("pillow_heif")
        pillow_heif.register_heif_opener()
    path = tmp_path / f"room{suffix}"
    write_photo(path, fmt)
    rgb, fx = load_photo(path)
    assert rgb.shape[:2] == (400, 300)            # portrait once turned
    assert rgb[:50, -100:].mean(axis=(0, 1))[0] > 150 or rgb[:100, -50:].mean(axis=(0, 1))[0] > 150
    # 26 mm equivalent on a 3:4 portrait frame: 26 x 500 / (43.27 x 300)
    assert fx == pytest.approx(26 * 500 / (np.hypot(36, 24) * 300), rel=1e-6)


def test_no_focal_in_exif(tmp_path):
    path = tmp_path / "plain.jpg"
    write_photo(path, "JPEG", orientation=1, f35=None)
    rgb, fx = load_photo(path)
    assert rgb.shape[:2] == (300, 400) and fx is None
