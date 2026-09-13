# Submission guide: cozmo-scan

Cozmo AI Applied AI case study. This page says what is submitted, where each requirement of the brief is answered, what
is not done, and how to install and run the pipeline.

## 1. What is submitted

| Item | Where |
|---|---|
| Code, documents, benchmark results | https://github.com/ashupal22/cozmo-scan (branch `main`; give the final commit hash when submitting) |
| Raw captures: three Stray Scanner LiDAR walks (video, depth, confidence, poses, IMU) | Google Drive: https://drive.google.com/drive/folders/1rvcx0uEIwU6mIlEi8m5SF88jHK6ubAOu |
| Checksums of the raw captures | `data/captures_manifest.json` (`python scripts/capture_manifest.py <captures> --check`) |
| Public data used | ARKitScenes (6 walks, laser truth), HouseLayout3D, BD3 test photos: fetched by script, never redistributed |

Suggested message:

> Repository: https://github.com/ashupal22/cozmo-scan (commit <final commit hash>). Raw captures: <Drive link>. Start with
> SUBMISSION.md, then docs/compliance_matrix.md and docs/technical_report.pdf. Install with `bash scripts/install.sh`; one command per
> capture: `cozmo run <capture>`. Gaps are stated in the compliance matrix: no tape truth, no head-to-head, no real
> staged-damage room.

## 2. Reading order

1. `docs/compliance_matrix.md`: every requirement → file → artifact → status.
2. `docs/technical_report.pdf` (4 pages): architecture, tiers and device matrix, drift, error budget, calibration, fix loop, failure modes.
3. `docs/benchmark_report.md`: gates at all three tiers, repeatability, head-to-head (empty), timing.
4. `docs/fix_loop_declaration.pdf` (the one-page declaration), then `docs/fix_loop.md` and `bench/results/fix_loop/`: the result, the before and after runs, the diff and the post-mortem.
5. `docs/capture_protocol.pdf` (one page) and `docs/device_matrix.md`.
6. `docs/reproduction_check.md`: every benchmark re-run from a clean clone.
7. `README.md`: install and run. `docs/walk_in.md`: the defence-day runbook.
8. `docs/data_formats.md`: input, test-data and output formats, and why.

## 3. How the pipeline works, in plain words

**One capture, one tier, one command.**
- The brief's three tiers are separate inputs:
  - a Stray Scanner LiDAR folder
  - a Camera-app video
  - folders of photos
- `cozmo run` detects which one it received, and each tier alone produces the full output.
- On the walk-in day the examiners choose one tier.
- A normal iPhone video or photo has no depth. Only a LiDAR recording app saves depth for each frame.

**The steps of each tier.**
- **LiDAR:**
  - phone depth, corrected by +11.9 mm
  - 3D points with surface directions
  - drift correction (pose graph, loop closures)
  - floor and ceiling
  - walls first, then rooms and doors
  - measurements with 90% intervals
  - damage, hidden-damage rules, repair scope
  - `result.json`, `plan.svg`, `summary.md`, and the one-page visual report `report.pdf` (`report.png`)
- **Video:**
  - key frames at 3 per second, up to 360
  - depth and camera path from Depth Anything 3, in overlapping runs of 12 frames
  - focal length from the room's straight lines
  - metres from the metric depth model, times a gain of 1.07 calibrated against LiDAR
  - then the same steps as LiDAR
- **Photos, per room folder:**
  - depth and cameras from Depth Anything 3
  - focal length from EXIF, giving metres
  - a right-angled room box, and the doors seen through its walls
  - rooms joined into one plan through matching doors
  - the same outputs

**What is measurement and maths, and what is a trained model.**

| Part | How it is made |
|---|---|
| LiDAR depth | Measured by the phone's LiDAR (Apple fills the gaps between laser points); within about 1 cm of a laser scanner after our correction |
| Camera path at the LiDAR tier | Apple's ARKit tracking (camera and motion sensors) |
| Drift correction, floor, ceiling, walls, rooms, doors, stitching, sizes, intervals | Our maths: least squares, plane fits, graph cuts, geometry, error propagation |
| Hidden-damage flags and repair scope | Written rules and arithmetic |
| Depth and camera path at the video and photo tiers | **Model**: Depth Anything 3, because a plain camera cannot measure distance |
| Damage type, and the mirror, glass and wet-floor warnings | **Model**: CLIP. It never changes a measurement; `--no-damage` switches it off |
| +11.9 mm, ×1.07, and the interval factors ×4.5 and ×5.1 | Fitted from measured errors, not predicted |

Input, test-data and output formats, and why each was chosen: `docs/data_formats.md`.

## 4. The brief's deliverables

| # | Deliverable | Where | Status |
|---|---|---|---|
| 1 | Compliance matrix | `docs/compliance_matrix.md` | Done |
| 2 | Capture route and device matrix | `docs/capture_protocol.md` + `.pdf` (Route 2: stock apps), `docs/device_matrix.md` | Done |
| 3 | README to a first run in under 15 minutes on a clean machine, one command per capture | `README.md`, `scripts/install.sh`, `scripts/fetch_models.py` | Done: install tested in a fresh environment (macOS, Apple silicon) |
| 4 | Reproduction bundle | `bench/reproduce.sh`, `scripts/fetch_external.py`, `bench/compare_results.py`, `docs/reproduction_check.md` | Done: 8 of 9 benchmarks give the same numbers; video gives the same measurements with wider interval bounds |
| 5 | Benchmark report | `docs/benchmark_report.md`, `bench/README.md` | Done, except the head-to-head table (empty) |
| 6 | Fix loop bundle | `docs/fix_loop_declaration.md` + `.pdf` (one page), `docs/fix_loop.md`, `bench/results/fix_loop/` | Done |
| 7 | Technical report, at most 6 pages | `docs/technical_report.md` + `.pdf` | Done (4 pages) |
| 8 | Raw benchmark data | Google Drive, `data/captures_manifest.json` | Partial: sensor logs yes; no tape truth, no app exports |

## 5. Coverage of the brief

**Part 1, capture and tiers**
- Stock capture route with a one-page protocol, followable by a non-engineer: done.
- Photo tier (2–8 photos per room, one folder per room, stitched plan): runs. Stitching often leaves separate groups.
- Video tier (walkthrough clip): runs.
- LiDAR tier (depth, poses, intrinsics): runs.
- The same output from each tier, with intervals widening as data thins: done (LiDAR, then video ×4.5, then photo ×5.1).
- Device matrix with honest accuracy: done.

**Part 2, output contract**
- Per-room plan with walls, ceiling height, floor area and openings: done. Windows are not detected; openings are doors and gaps.
- Stitched plan with adjacency: done. Photo tier partial.
- Damage regions with class and size: done; detection quality is limited (next section).
- Hidden-damage flags with the rule that fired, and scope keyed to surfaces: done.
- An interval on every measurement, one command per capture, validated JSON, rendered plan: done (plus `summary.md`).

**Part 2, benchmark set**
- Multi-room capture with a connector: done (whole flat with hallway).
- Same room captured twice at one tier: done (flat walked twice).
- Same rooms at all three tiers: partial (video and photos cut from the LiDAR walks).
- Staged-damage room: not done (public defect photos and synthetic staging instead).
- Laser or tape truth on everything: not done (public laser data only).

**Part 2, gates**

| Gate | Result | Status |
|---|---|---|
| Ceiling height ≤ 1.5 cm | 6 of 6 public laser-truth walks | Met (public data) |
| Ceiling spread ≤ 1 cm | 10.1 and 43.9 mm; report says "biased, then corrected" | Not met |
| Repeatability ≤ 1 cm or 0.5% | 3 of 14 room dimensions | Not met |
| Drift accountability | method and on/off ablation: 4.4% → 1.6% | Met |
| Opening widths ≤ 2 cm on 85% | no tape truth; detection not repeatable | Not met |
| Photo stitch, footprint ±8% | −31% and −41%, separate groups | Not met |
| Photo walls ±8% | 1 of 18; intervals hold 17 of 18 | Not met (calibrated) |
| Video walls ±3% | 0 of 22; intervals hold 6 of 7 | Not met (calibrated) |
| Calibration at every tier | LiDAR 6/6 ceilings and 4/4 walls; video 6/7; photo 17/18 | Done |

**Part 3, head-to-head:** not done. It needs magicplan (or Polycam) and a tape measure on two of the rooms;
`bench/tape_truth.py` is ready.

**Part 4, fix loop:** declaration committed before the fix, fix shipped, regenerable before and after, readable diff,
post-mortem. The prediction was badly wrong, and the fix is switched off because it measured worse.

**Part 5, process:** 94 commits, each with its evidence; predictions committed before results.

**Walk-in test:** all three tiers run cold from a fresh install (`docs/walk_in.md`).

**Constraints:**
- Models and datasets disclosed; everything runs locally.
- Weights fetched by script.
- Mirrors, glass, wet-look floors and low light are covered by warnings and the protocol.

## 6. What was built

- **LiDAR tier:**
  - fusion of depth into oriented points
  - floor and ceiling chosen by covered area
  - +11.9 mm depth-bias correction (ceiling 1/6 → 6/6)
  - drift correction (pose graph, loop closures)
  - walls-first room layout, rooms split at doorways
  - openings, and an error model with intervals
- **Video tier:**
  - Depth Anything 3 on Apple silicon
  - chained runs with a common scale
  - focal length from the room's lines
  - metric scale calibrated to about 1%
  - tracking-break detection
  - intervals calibrated (×4.5)
  - key-frame cap set from a measurement
- **Photo tier:**
  - one room box per folder, EXIF focal length (checked within 1% on real iPhone photos)
  - door detection
  - a stitch solver benchmarked on 23 real buildings
  - intervals calibrated (×5.1)
- **Contract:**
  - JSON schema and validation
  - plan drawing and `summary.md`
  - CLIP damage detection
  - 6 concealed-damage rules and repair scope
  - scene-condition warnings
- **Evidence:** 15 benchmark scripts with committed results, a clean-clone reproduction check, and checksums for the raw data.

## 7. Not done, and what would improve it

**Needs a phone and a tape measure in real rooms:**
1. Tape truth on two or more rooms: real errors for every gate.
2. Head-to-head against magicplan: 10% of the score.
3. A furnished room with staged damage (two classes).
4. The same rooms captured with an iPhone 15 as photo folders and as a Camera-app video.

**Engineering:**
- Video camera path at blank-wall turns (G-WALL-VIDEO).
- Photo room shape and stitching (G-WALL-PHOTO, G-PHOTO-STITCH).
- Window detection, and repeatable door detection and widths (G-OPEN).
- Repeatable room splitting (G-REPEAT).
- Damage seen from across a room (0 of 2 staged marks found).
- Correcting geometry near mirrors and glass (only flagged today).
- Linux testing.

## 8. User guide: install and run

**Install (macOS on Apple silicon, Python 3.10–3.12, about 3 GB of model weights):**

```bash
brew install ffmpeg
git clone https://github.com/ashupal22/cozmo-scan.git && cd cozmo-scan
python3 -m venv .venv && source .venv/bin/activate
bash scripts/install.sh        # packages, Depth Anything 3 (pinned), model weights
pytest -q                      # 121 tests, about 30 s
```

**Capture:** follow `docs/capture_protocol.pdf` (one page).
- LiDAR: Stray Scanner on an iPhone Pro, walking every room along the walls and tilting up to the ceiling.
- Video: Camera app, 1080p at 30 fps, HDR off, under 2 minutes.
- Photos: from each room's doorway, 5–6 overlapping photos plus one per other door, one folder per room.

**Run (one command per capture; the tier is detected from the input):**

```bash
cozmo run path/to/StrayScannerExport      # LiDAR: about 30 s to 2 min
cozmo run path/to/walkthrough.mov         # video: about 13 min for a 2-minute clip the first time
cozmo run path/to/home                    # photos: home/kitchen/*.heic, home/bedroom 1/*.heic, ...
```

**Output:** `out/<name>/`
- `result.json`: everything, with a 90% interval on every number, validated against `schema/output.schema.json`.
- `plan.svg`: the floor plan, dimensioned.
- `summary.md`: each room's width × length, ceiling height and door widths, with ranges.
- `report.png` and `report.pdf`: everything above on one page, with the plan drawing. The easiest way to see a result.

**Other commands:**
- `cozmo inspect <capture>`: what a capture holds.
- `cozmo validate <result.json>`: checks a file against the schema.
- `cozmo report <result.json>`: redraws the one-page visual report for any result.
- `--no-drift`: the drift ablation.
- `--no-damage`: skips damage detection (saves 20–60 s).

**Reproduce every reported number:**

```bash
export COZMO_DATA=/path/to/captures      # the three folders from Google Drive
python scripts/fetch_external.py arkitscenes && python scripts/fetch_external.py houselayout3d
bash bench/reproduce.sh                  # about an hour with cached model outputs
python bench/compare_results.py bench/results --committed <a clean checkout>/bench/results
```

**Troubleshooting:**
- "ffmpeg is needed": run `brew install ffmpeg`.
- No internet on the day: run `python scripts/fetch_models.py` beforehand, then set `HF_HUB_OFFLINE=1`.
- Disk: keep 5 GB free, since a video run writes key frames and model outputs.
- Apple GPU error: the model reruns on the CPU automatically, just slower.
