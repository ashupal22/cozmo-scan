"""Download the external datasets listed in docs/datasets.md into data/external/ (ignored by git).

    python scripts/fetch_external.py arkitscenes      # 2 venues x 3 walks, LiDAR + laser-rendered depth (~1 GB)
    python scripts/fetch_external.py houselayout3d    # multi-room layout annotations (~32 MB, MIT)

Re-running skips anything already downloaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARKIT_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
# Validation venues where every walk has laser-rendered ground-truth depth (highres_depth);
# chosen as the smallest downloads among venues with 3 walks.
ARKIT_VISITS = [381644, 384651]
ARKIT_ZIP_ASSETS = ["lowres_depth", "confidence", "lowres_wide", "lowres_wide_intrinsics", "highres_depth"]
ARKIT_FILE_ASSETS = ["lowres_wide.traj"]


def download(url: str, dst: Path) -> Path:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
        total, done, t0 = int(response.headers.get("Content-Length", 0)), 0, time.time()
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            rate = done / max(time.time() - t0, 1e-6) / 1e6
            print(f"\r  {dst.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB ({rate:.1f} MB/s)", end="", flush=True)
    print()
    tmp.rename(dst)
    return dst


def fetch_arkitscenes(out: Path, visits: list[int]) -> None:
    meta = download(f"{ARKIT_URL}/raw/metadata.csv", out / "metadata.csv")
    with open(meta, newline="") as f:
        rows = list(csv.DictReader(f))
    walks = [r for r in rows
             if r["visit_id"] not in ("", "NA") and int(float(r["visit_id"])) in visits
             and r["fold"] == "Validation" and r["is_in_upsampling"] == "True"]
    if not walks:
        sys.exit("no matching ARKitScenes walks found in metadata.csv")

    manifest = []
    for r in walks:
        video, visit = r["video_id"], int(float(r["visit_id"]))
        walk_dir = out / "Validation" / video
        print(f"visit {visit}, walk {video}")
        for asset in ARKIT_ZIP_ASSETS:
            if (walk_dir / asset).is_dir():
                continue
            archive = download(f"{ARKIT_URL}/raw/Validation/{video}/{asset}.zip", walk_dir / f"{asset}.zip")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(walk_dir)
            archive.unlink()
        for asset in ARKIT_FILE_ASSETS:
            download(f"{ARKIT_URL}/raw/Validation/{video}/{asset}", walk_dir / asset)
        manifest.append({"visit_id": visit, "video_id": video, "sky_direction": r["sky_direction"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"done: {len(manifest)} walks in {out}")


def fetch_houselayout3d(out: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub, or: git clone https://huggingface.co/datasets/houselayout3d/HouseLayout3D")
    snapshot_download(repo_id="houselayout3d/HouseLayout3D", repo_type="dataset", local_dir=str(out))
    print(f"done: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["arkitscenes", "houselayout3d"])
    parser.add_argument("--out", type=Path, help="default: data/external/<dataset>")
    parser.add_argument("--visits", type=int, nargs="+", default=ARKIT_VISITS, help="ARKitScenes visit ids")
    args = parser.parse_args()
    out = args.out or ROOT / "data" / "external" / args.dataset
    if args.dataset == "arkitscenes":
        fetch_arkitscenes(out, args.visits)
    else:
        fetch_houselayout3d(out)


if __name__ == "__main__":
    main()
