"""Checksums of the raw captures, so anyone can confirm that their copies from Google Drive are the ones every result used.

    python scripts/capture_manifest.py /path/to/captures                  # writes data/captures_manifest.json
    python scripts/capture_manifest.py /path/to/captures --check          # compares your copies with it

Per capture: SHA-256 and size of each top-level file (rgb.mp4, odometry.csv, ...), and for the depth/ and confidence/
folders the file count, total size and one SHA-256 over all file names and file hashes in sorted order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "captures_manifest.json"
CAPTURES = ("c00a170fe1", "1a8384c3f6", "c7d28f72c6")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def folder_digest(folder: Path) -> dict:
    h, files, size = hashlib.sha256(), 0, 0
    for p in sorted(q for q in folder.rglob("*") if q.is_file()):
        h.update(p.relative_to(folder).as_posix().encode())
        h.update(bytes.fromhex(file_sha256(p)))
        files += 1
        size += p.stat().st_size
    return {"files": files, "bytes": size, "sha256": h.hexdigest()}


def capture_entry(folder: Path) -> dict:
    entry = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            entry[p.name] = {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
        elif p.is_dir() and not p.name.startswith("."):
            entry[p.name + "/"] = folder_digest(p)
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    manifest = {cid: capture_entry(args.captures / cid) for cid in CAPTURES if (args.captures / cid).is_dir()}
    if not args.check:
        OUT.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {OUT} ({len(manifest)} captures)")
        return 0
    expected = json.loads(OUT.read_text())
    bad = [f"{cid}/{name}" for cid, entries in expected.items() for name, e in entries.items()
           if manifest.get(cid, {}).get(name) != e]
    print("all captures match" if not bad else "differ: " + ", ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
