"""Photo-tier benchmark sets cut from our own walks: a disclosed stand-in, since no separate photo capture exists.

For each walk, every room of its LiDAR plan gets a folder of stills from the phone's video, taken where the
camera stood inside that room: the directions the camera looked are split into sectors, and each sector gives
its sharpest roughly level frame. That yields up to MAX_PER_ROOM per room (the brief allows 2 to 8). Frames are
full resolution and upright, with no poses, no depth and no metadata. The photo tier's plan is judged against
the LiDAR plan of the same walk.

Honest limits: video frames are softer than photos (motion blur, compression), and they come from wherever
the walker stood, often close to a wall, not from a corner as a person taking room photos would. Rooms of
the LiDAR plan that the camera never entered get no folder, and are reported in the manifest.

    COZMO_DATA=/path/to/captures python bench/make_photo_sets.py [walk ...]
Writes data/derived/<walk>_photos/<room>/*.jpg and data/derived/<walk>_photos/manifest.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from video_vs_lidar import DATA, DERIVED, VIDEO_POSE_LAG_S, WALKS, code_commit, upright_video  # noqa: E402

from cozmo.ingest.stray import StrayCapture  # noqa: E402
from cozmo.pipeline import run_lidar  # noqa: E402

MAX_PER_ROOM = 6
MIN_PER_ROOM = 2
SECTORS = 6
CANDIDATE_EVERY = 10          # odometry rows between candidate frames (6 per second)
MAX_PITCH_DEG = 30            # a room photo looks roughly level
MIN_INSIDE_M = 0.3            # camera at least this far inside the room outline
SHARPNESS_SIZE = (360, 480)


def forward_heading_pitch(cap: StrayCapture, i: int, theta: float) -> tuple[float, float]:
    """Heading (degrees, in the drift-corrected plan) and pitch of the camera's viewing direction."""
    f = cap.rotation(i)[:, 2]
    c, s = np.cos(theta), np.sin(theta)
    x, z = c * f[0] + s * f[2], -s * f[0] + c * f[2]      # same convention as FrameCorrection.transform_xz
    return float(np.degrees(np.arctan2(z, x))), float(np.degrees(np.arcsin(np.clip(f[1], -1, 1))))


def video_frame_for_rows(video: Path, cap: StrayCapture) -> np.ndarray:
    """Index of the video frame shown at each odometry row (video frames lag poses by VIDEO_POSE_LAG_S)."""
    pts = np.array([float(v.strip(",")) for v in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0",
         str(video)], capture_output=True, text=True).stdout.split() if v.strip(",")])
    t = cap.timestamps - cap.timestamps[0]
    return np.clip(np.searchsorted(pts + VIDEO_POSE_LAG_S, t), 0, len(pts) - 1)


def decode(video: Path, wanted: set[int], size=None) -> dict[int, np.ndarray]:
    """Frames by index, decoded in order (seeking is not exact for these files)."""
    out, reader, j = {}, cv2.VideoCapture(str(video)), 0
    last = max(wanted) if wanted else -1
    while j <= last:
        ok, frame = reader.read()
        if not ok:
            break
        if j in wanted:
            out[j] = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if size else frame
        j += 1
    reader.release()
    return out


def make_sets(cid: str) -> dict:
    cap = StrayCapture(DATA / cid)
    plan = run_lidar(DATA / cid)
    corr = plan.correction
    video = upright_video(cid)
    frame_of_row = video_frame_for_rows(video, cap)
    rooms = {rid: Polygon(o.vertices) for rid, o in plan.outlines.items()}
    candidates: dict[int, list[tuple[int, float]]] = {rid: [] for rid in rooms}
    for i in range(0, len(cap), CANDIDATE_EVERY):
        if not corr.valid[i]:
            continue
        here = Point(corr.positions[i, 0], corr.positions[i, 2])
        heading, pitch = forward_heading_pitch(cap, i, corr.theta[i])
        if abs(pitch) > MAX_PITCH_DEG:
            continue
        for rid, poly in rooms.items():
            if poly.contains(here) and poly.exterior.distance(here) >= MIN_INSIDE_M:
                candidates[rid].append((i, heading))
    wanted = {int(frame_of_row[i]) for c in candidates.values() for i, _ in c}
    small = decode(video, wanted, SHARPNESS_SIZE)
    sharp = {j: float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) for j, im in small.items()}
    del small
    chosen: dict[int, list[int]] = {}
    for rid, cands in candidates.items():
        best: dict[int, tuple[float, int]] = {}
        for i, heading in cands:
            j = int(frame_of_row[i])
            sector = int(((heading % 360.0) / 360.0) * SECTORS) % SECTORS
            if j in sharp and (sector not in best or sharp[j] > best[sector][0]):
                best[sector] = (sharp[j], j)
        frames = sorted((v for v in best.values()), reverse=True)[:MAX_PER_ROOM]
        if len(frames) >= MIN_PER_ROOM:
            chosen[rid] = sorted(j for _, j in frames)
    full = decode(video, {j for js in chosen.values() for j in js})
    out_dir = DERIVED / f"{cid}_photos"
    for old in out_dir.glob("*/*.jpg"):
        old.unlink()
    manifest = {"walk": cid, "code_commit": code_commit(), "source": str(video.name),
                "rooms": {}, "lidar_rooms_without_photos": []}
    for rid in sorted(rooms):
        if rid not in chosen:
            manifest["lidar_rooms_without_photos"].append({"room": f"R{rid}", "area_m2": round(rooms[rid].area, 2),
                                                           "candidate_frames": len(candidates[rid])})
            continue
        folder = out_dir / f"room_{rid}"
        folder.mkdir(parents=True, exist_ok=True)
        for j in chosen[rid]:
            cv2.imwrite(str(folder / f"{j:05d}.jpg"), full[j], [cv2.IMWRITE_JPEG_QUALITY, 92])
        manifest["rooms"][f"room_{rid}"] = {"lidar_room": f"R{rid}", "area_m2": round(rooms[rid].area, 2),
                                            "photos": len(chosen[rid]), "video_frames": chosen[rid]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    for cid in (sys.argv[1:] or WALKS):
        m = make_sets(cid)
        print(cid, {k: v["photos"] for k, v in m["rooms"].items()}, "without photos:", m["lidar_rooms_without_photos"], flush=True)


if __name__ == "__main__":
    main()
