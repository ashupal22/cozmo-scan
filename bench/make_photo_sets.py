"""Photo-tier benchmark sets cut from our own walks: a disclosed stand-in, since no separate photo capture exists.

For each walk, every room of its LiDAR plan gets a folder of stills from the phone's video, taken where the
camera stood inside that room. Frames are full resolution and upright, with no poses, no depth and no metadata.
The photo tier's plan is judged against the LiDAR plan of the same walk. Two ways to pick the stills:

- sectors (default): the directions the camera looked are split into sectors, and each sector gives its
  sharpest roughly level frame, up to MAX_PER_ROOM. These are arbitrary views, often through doorways, and
  unrelated views break Depth Anything 3's camera estimates (bench/README.md).
- --sweep: what the capture protocol asks for. Find where the camera turned the most while staying within
  SWEEP_RADIUS_M of one spot, and take a frame every SWEEP_STEP_DEG of that turn, so neighbouring photos
  overlap by about a third (portrait view about 48 degrees wide). Rooms where the walk never turned at least
  SWEEP_MIN_DEG in one spot fall back to sectors; the manifest says which.

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
MAX_SWEEP_PHOTOS = 8
SWEEP_EVERY = 5               # odometry rows between sweep candidates (12 per second)
SWEEP_RADIUS_M = 0.6
SWEEP_MAX_S = 20.0
SWEEP_STEP_DEG = 30.0
SWEEP_MIN_DEG = 90.0
SWEEP_ROW_GAP = 15            # candidates further apart than this break a sweep (level, inside the room)
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


def best_sweep(cands: list[tuple[int, float]], positions: np.ndarray, times: np.ndarray):
    """The stretch of candidates (row, heading), in time order, where the camera turned the most while staying
    in one spot. Returns (turned degrees, candidate indices, unwrapped headings) or None."""
    rows = np.array([c[0] for c in cands])
    heading = np.array([c[1] for c in cands])
    best = None
    for s in range(len(rows)):
        unwrapped, lo, hi = [0.0], 0.0, 0.0
        for e in range(s + 1, len(rows)):
            if rows[e] - rows[e - 1] > SWEEP_ROW_GAP or times[rows[e]] - times[rows[s]] > SWEEP_MAX_S:
                break
            if np.linalg.norm(positions[rows[e], [0, 2]] - positions[rows[s], [0, 2]]) > SWEEP_RADIUS_M:
                break
            unwrapped.append(unwrapped[-1] + (heading[e] - heading[e - 1] + 180.0) % 360.0 - 180.0)
            lo, hi = min(lo, unwrapped[-1]), max(hi, unwrapped[-1])
        if best is None or hi - lo > best[0]:
            best = (hi - lo, np.arange(s, s + len(unwrapped)), np.array(unwrapped) - lo)
    return best


def sweep_targets(turned: np.ndarray, idx: np.ndarray) -> list[np.ndarray]:
    """Candidate indices near each target heading (every SWEEP_STEP_DEG from the start of the turn)."""
    span = float(turned.max())
    count = min(MAX_SWEEP_PHOTOS, int(span // SWEEP_STEP_DEG) + 1)
    return [idx[np.abs(turned - k * SWEEP_STEP_DEG) <= SWEEP_STEP_DEG / 3] for k in range(count)]


def make_sets(cid: str, sweep: bool = False) -> dict:
    cap = StrayCapture(DATA / cid)
    plan = run_lidar(DATA / cid)
    corr = plan.correction
    video = upright_video(cid)
    frame_of_row = video_frame_for_rows(video, cap)
    rooms = {rid: Polygon(o.vertices) for rid, o in plan.outlines.items()}
    candidates: dict[int, list[tuple[int, float]]] = {rid: [] for rid in rooms}
    for i in range(0, len(cap), SWEEP_EVERY if sweep else CANDIDATE_EVERY):
        if not corr.valid[i]:
            continue
        here = Point(corr.positions[i, 0], corr.positions[i, 2])
        heading, pitch = forward_heading_pitch(cap, i, corr.theta[i])
        if abs(pitch) > MAX_PITCH_DEG:
            continue
        for rid, poly in rooms.items():
            if poly.contains(here) and poly.exterior.distance(here) >= MIN_INSIDE_M:
                candidates[rid].append((i, heading))
    sweeps, modes = {}, {}
    if sweep:
        for rid, cands in candidates.items():
            found = best_sweep(cands, corr.positions, cap.timestamps) if cands else None
            if found is not None and found[0] >= SWEEP_MIN_DEG:
                sweeps[rid] = (found[0], sweep_targets(found[2], found[1]))
    wanted = set()
    for rid, cands in candidates.items():
        if rid in sweeps:
            wanted |= {int(frame_of_row[cands[k][0]]) for group in sweeps[rid][1] for k in group}
        else:
            wanted |= {int(frame_of_row[i]) for i, _ in cands[::CANDIDATE_EVERY // SWEEP_EVERY if sweep else 1]}
    small = decode(video, wanted, SHARPNESS_SIZE)
    sharp = {j: float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) for j, im in small.items()}
    del small
    chosen: dict[int, list[int]] = {}
    for rid, (turned, groups) in sweeps.items():
        picks = [max((int(frame_of_row[candidates[rid][k][0]]) for k in g), key=lambda j: sharp.get(j, -1.0))
                 for g in groups if len(g)]
        if len(picks) >= MIN_PER_ROOM:
            chosen[rid] = sorted(set(picks))
            modes[rid] = {"mode": "sweep", "turned_deg": round(float(turned))}
    for rid, cands in candidates.items():
        if rid in chosen:
            continue
        best: dict[int, tuple[float, int]] = {}
        for i, heading in cands:
            j = int(frame_of_row[i])
            sector = int(((heading % 360.0) / 360.0) * SECTORS) % SECTORS
            if j in sharp and (sector not in best or sharp[j] > best[sector][0]):
                best[sector] = (sharp[j], j)
        frames = sorted((v for v in best.values()), reverse=True)[:MAX_PER_ROOM]
        if len(frames) >= MIN_PER_ROOM:
            chosen[rid] = sorted(j for _, j in frames)
            modes[rid] = {"mode": "sectors"}
    full = decode(video, {j for js in chosen.values() for j in js})
    out_dir = DERIVED / f"{cid}_{'photo_sweeps' if sweep else 'photos'}"
    for old in out_dir.glob("*/*.jpg"):
        old.unlink()
    manifest = {"walk": cid, "code_commit": code_commit(), "source": str(video.name), "selection": "sweep" if sweep else "sectors",
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
                                            "photos": len(chosen[rid]), "video_frames": chosen[rid], **modes[rid]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    sweep = "--sweep" in sys.argv
    for cid in ([a for a in sys.argv[1:] if not a.startswith("--")] or WALKS):
        m = make_sets(cid, sweep=sweep)
        print(cid, {k: (v["photos"], v["mode"], v.get("turned_deg")) for k, v in m["rooms"].items()},
              "without photos:", m["lidar_rooms_without_photos"], flush=True)


if __name__ == "__main__":
    main()
