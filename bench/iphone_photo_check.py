"""The photo tier's scale input on real iPhone photos: HEIC decoding, EXIF orientation and the focal length.

Our photo sets are stills cut from video, with no EXIF, so the EXIF path had only synthetic tests. This downloads two
public sample photos taken on an iPhone 12 Pro (heic.digital samples; used locally, never redistributed) and
checks what cozmo.ingest.photos.load_photo reads.

Reference: the iPhone 12 Pro wide camera has a 4.2 mm lens (in the EXIF) and 1.4 µm pixels, so its focal length is
4.2 mm / 1.4 µm = 3000 px on the 4032 x 3024 frame. The focal length sets the photo tier's metric scale (metres =
model output x focal / 300), so its error is the scale error.

    python bench/iphone_photo_check.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cozmo.ingest.photos import EXIF_IFD, FOCAL_35MM_TAG, load_photo  # noqa: E402

OUT = ROOT / "bench" / "results" / "iphone_photo_check.json"
SAMPLES = ["https://heic.digital/download-sample/old-safe-wall.heic", "https://heic.digital/download-sample/classic-car.heic"]
PIXEL_PITCH_MM = 0.0014          # iPhone 12 Pro wide camera


def main():
    from PIL import Image
    import pillow_heif
    pillow_heif.register_heif_opener()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for url in SAMPLES:
            path = Path(tmp) / url.rsplit("/", 1)[-1]
            subprocess.run(["curl", "-s", "-L", "-m", "120", "-o", str(path), url], check=True)
            with Image.open(path) as im:
                exif = im.getexif()
                ifd = exif.get_ifd(EXIF_IFD)
                model, focal_mm, f35 = exif.get(0x0110), float(ifd.get(0x920A)), ifd.get(FOCAL_35MM_TAG)
            rgb, fx_over_width = load_photo(path)
            fx = fx_over_width * rgb.shape[1]
            reference = focal_mm / PIXEL_PITCH_MM
            rows.append({"sample": path.name, "camera": model, "upright_size": [rgb.shape[1], rgb.shape[0]],
                         "exif_focal_mm": focal_mm, "exif_35mm_equivalent": f35, "focal_px_from_exif": round(fx, 1),
                         "focal_px_reference": round(reference, 1), "scale_error_pct": round(100 * (fx / reference - 1), 2)})
            print(json.dumps(rows[-1]))
    OUT.write_text(json.dumps({"check": "iphone_photo_check", "pixel_pitch_mm": PIXEL_PITCH_MM, "photos": rows}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
