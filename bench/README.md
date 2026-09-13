# Benchmarks

Every number here can be regenerated from raw data with the command shown. Results are stored in `bench/results/`, and git history keeps the before/after versions.

## Floor and ceiling vs laser ground truth (ARKitScenes)

```bash
python scripts/fetch_external.py arkitscenes     # once, ~3 GB
python bench/arkitscenes_planes.py               # ~10 min on an Apple M4
```

This compares device LiDAR against depth rendered from Faro laser scans on 6 walks in 2 venues (3 walks each). Targets from `docs/gates.md`: **G-CEIL**, ceiling error ≤ 15 mm; **G-CEIL-SPREAD**, spread across walks of the same room ≤ 10 mm.

### Before and after choosing planes by area (commit `1749867`)

Negative ceiling error means the ceiling reads too low; positive floor error means the floor reads too high.

| Walk | Venue | Ceiling error, before | Ceiling error, after | Floor error, before | Floor error, after |
|---|---|---|---|---|---|
| 41069048 | 381644 | −28.1 mm | −28.1 mm | +16.1 mm | +16.1 mm |
| 41069050 | 381644 | −26.9 mm | −26.9 mm | +14.3 mm | +14.3 mm |
| 41069051 | 381644 | −23.6 mm | −23.6 mm | +10.7 mm | +10.7 mm |
| 41142278 | 384651 | **−221.2 mm** | −20.9 mm | **+207.7 mm** | +7.4 mm |
| 41142280 | 384651 | −11.2 mm | −11.2 mm | +12.1 mm | +12.1 mm |
| 41142281 | 384651 | −19.8 mm | −19.8 mm | +11.0 mm | +11.0 mm |
| **Within 15 mm** | | **1/6** | **1/6** | | |
| **Mean** | | −55.1 mm | −21.8 mm | | |

### What the numbers say

- **The fix worked for the problem it targeted.** On walk 41142278 a small raised surface seen up close had more points than the floor. Choosing by covered area picks the real floor, so the pipeline's choice now matches the laser on all 6 walks.
- **The remaining error is a consistent bias, not noise.** Every walk reads the ceiling 11–28 mm low and the floor 7–16 mm high. Per pixel, device depth is 9–16 mm shorter than laser depth between 0.3 and 2 m: the sensor sees surfaces slightly closer than they are. In the brief's words this is **repeatable but biased**, and it fails G-CEIL.
- **Spread:** venue 381644 spreads 9.2 mm across its walks (passes G-CEIL-SPREAD). Venue 384651 spreads 42.6 mm, but its own laser ceiling heights also differ by 34 mm between walks, because different walks see different parts of a ceiling with more than one level. A fair spread test must compare the same ceiling plane in the same room. That's the next revision of this benchmark.

### Caveats

- ARKitScenes was captured on a 2020 iPad Pro, not an iPhone 15 or newer.
- Laser-rendered frames were pre-filtered to agree with the device depth, which may flatter the device slightly.

### Next candidate fix

Calibrate the depth bias (as a constant offset or proportional to range, whichever fits the residuals) on held-out walks, then re-run. This is a candidate for the brief's fix loop, pending the full benchmark on all gates.

## Drift correction on and off (our three walks)

```bash
COZMO_DATA=/path/to/captures python bench/drift_ablation.py     # ~2 min
```

The brief's G-DRIFT gate asks what we do about drift, and for an ablation showing the stitched footprint with the correction on and off. The method is in `cozmo/slam/drift.py`:

1. Cut the walk into 4-second submaps.
2. Each submap measures its dominant wall direction and its floor height.
3. The pose graph holds the heading straight against the walls. It models heading drift as a steady creep, removes vertical drift using the floor, and closes loops where the walk revisits a place.
4. An ARKit snap is used as a loop closure back to the start of the walk.

**What can and can't be measured.** Our own walks have no ground truth. ARKitScenes can't measure drift either, because its laser depth is rendered from the same phone poses, so pose errors cancel. The evidence is therefore:
- **Within a walk:** walls seen twice should land on top of each other once drift is gone, so the area covered by wall points shrinks. Wall directions should agree.
- **Between walks:** `c7d28f72c6` and `1a8384c3f6` are the same flat, so their wall maps should agree after one rigid fit.
- **Synthetic walks with known drift** in `tests/test_drift.py`.

### Per walk (commit `412a6dc`)

| Walk | Walked | Wall-map area, off → on | Wall direction p90, off → on | Loops accepted | Largest correction | Rooms, off → on | Footprint, off → on |
|---|---|---|---|---|---|---|---|
| c00a170fe1 | 14 m | 3.8 → 3.0 m² | 1.58° → 1.20° | 2 of 3 | 1.1°, 0.11 m | 2 → 2 | 17.07 → 17.85 m² |
| 1a8384c3f6 | 54 m | 14.2 → 11.0 m² | **5.10° → 1.59°** | 6 of 27, plus the ARKit snap | 4.6°, 0.61 m | 5 → 5 | 38.93 → 40.01 m² |
| c7d28f72c6 | 100 m | 18.1 → 14.0 m² | 1.84° → 1.60° | 54 of 68 | 0.7°, 0.21 m | 6 → 5 | 43.46 → 44.53 m² |

Plan drawings, both ways: `bench/results/drift_<walk>_off.svg` and `_on.svg`.

- **Every walk gets sharper:** 21–23% less wall-map area.
- **`1a8384c3f6` had a real 6° heading creep.** Its wall directions went from 83° at the start to 89° at the end, while the other walk of the same flat stays within about 1°. The correction straightens it.
- **Its ARKit snap is confirmed.** Near the end, ARKit jumped 59 cm and landed 10 cm from where the walk began. After the snap, 76% of its wall points sit within 5 cm of the walls seen at the start, so it's used as a loop closure to the start.

### Same flat, two walks (`c7d28f72c6` vs `1a8384c3f6`)

Walls of each walk compared with the other after one rigid fit, in both directions:

| | Drift off | Drift on |
|---|---|---|
| Wall points within 2 cm of the other walk | 51.6% | **68.2%** |
| Within 5 cm | 69.5% | **86.8%** |
| Per-metre-tile median distance, p50 | 1.52 cm | 1.17 cm |
| Per-metre-tile median distance, p90 | 5.99 cm | **2.78 cm** |
| Worst tile | 12.5 cm | 12.8 cm |

### What this does not fix yet

The wall maps agree much better, but **the two plans still differ**: footprints of 44.5 vs 40.0 m². One reason is room splitting: in `c7d28f72c6` the hallway merged into the largest room once walls sharpened. The other is outlines traced from seen floor, which depend on how much of each room was walked. The next fix is wall-first outlines and splitting rooms at doors.

## Same flat, same plan: rooms from walls first (`bench/same_flat_plans.py`)

```bash
COZMO_DATA=/path/to/captures python bench/same_flat_plans.py     # ~3 min
```

Both walks of the flat go through the LiDAR pipeline with drift correction. The second walk's plan is placed on the first by one rigid fit of their wall maps. Rooms are paired when they overlap by IoU ≥ 0.5.

| | Rooms traced from seen floor (old) | Rooms from walls first (new, `cozmo/geometry/layout.py`) |
|---|---|---|
| Rooms, walk A / walk B | 5 / 5 | 9 / 7 |
| Footprint, walk A / walk B | 44.5 / 40.0 m² | 55.8 / 54.7 m² |
| **Footprint difference** | 10.2% | **1.9%** |
| Rooms paired (IoU ≥ 0.5) | 5 | 7 |
| Median IoU of paired rooms | 0.82 | 0.78 |
| Median difference of room main dimensions | 24.9 cm | 17.0 cm |
| Dimensions within 1 cm or 0.5% (the gate) | 2 of 10 | 3 of 14 |

**What this shows**
- **Walls first makes the outline of the flat repeatable.** Rooms reach their walls behind furniture instead of stopping where the floor was hidden, so both walks agree on the footprint to 1.9%.
- **When both walks see a room's walls well, the room repeats to the gate's level.** Room A-R4 / B-R4 measures 5.56 vs 5.65 m², IoU 0.93, with both main dimensions within 0.6 cm.
- **The remaining difference is room splitting.** Walk `1a8384c3f6` saw almost no wall above 1.5 m because the phone pointed down, so some inner walls were never seen as walls. It merged the corridor with the living area into one 19 m² room, where the other walk gives separate rooms.
  - This is why the capture guide must ask for a ceiling sweep.
  - The pipeline softens it: a room is split again where the seen floor narrows at a doorway.

### History of this benchmark

- **First run (`ea62760`): the correction barely engaged.** No wall directions were used on two walks, no loops were accepted on two walks, and a table top used as floor moved one walk 44 cm.
- **Fixes (`2720656`, `1807cd0`, `b6b144c`):** looser wall-direction threshold, floor gating, a submap merge bug, loops matched around the map centre, heading drift modelled as a steady creep, ARKit snaps tied to the walk start, and loops matched on first-pass-straightened maps.
- **Metric fix (`5fdba7f`):** the two-walk comparison became symmetric. The one-way version rewarded blurred maps.

## Video tier: focal length and real-world scale (`bench/video_scale.py`)

```bash
COZMO_DATA=/path/to/captures python bench/video_scale.py     # ~6 min with cached model outputs
```

The video tier has no depth sensor and no camera calibration, so its real-world size rests on two estimates: the focal length, and a monocular metric-depth model (DA3METRIC-LARGE) whose metres are proportional to that focal length. The reference is each capture's own data. The Stray video, turned upright the way the Camera app stores it, goes through the video tier, and ARKit's calibration and the LiDAR depth of the same frames are the truth.

Matching video frames to poses needed one correction. Stray's video frames lag its poses by about 75 ms (4–5 frames at 60 fps). Rotations measured from feature matches with LiDAR depth disagree with ARKit by a median 1.94° at no lag and 0.72° at 4 frames.

### Focal length (focal / image width)

| Walk | ARKit calibration | Room lines (`cozmo/video/focal.py`) | DA3's own estimate |
|---|---|---|---|
| c00a170fe1 | 1.110 | +1.1% | +10.9% |
| 1a8384c3f6 | 1.108 | +1.6% | +11.4% |
| c7d28f72c6 | 1.098 | +2.4% | +11.2% |

### Scale: LiDAR depth ÷ video depth, median over key frames (gain and range correction off)

| Walk | Ratio | Gain from the other two walks | Held-out walk off by |
|---|---|---|---|
| c00a170fe1 | 1.062 | 1.074 | −1.09% |
| 1a8384c3f6 | 1.075 | 1.067 | +0.75% |
| c7d28f72c6 | 1.072 | 1.068 | +0.35% |

### Range bias left after the gain, binned by video depth

"+" means the video reads short. Each cell is before → after the correction fitted on the other two walks.

| Video depth | c00a170fe1 | 1a8384c3f6 | c7d28f72c6 |
|---|---|---|---|
| 0.6 m | −5.2 → −3.4% | −2.4 → −0.1% | −2.5 → +0.8% |
| 1.1 m | −3.6 → −3.0% | −0.2 → +0.9% | −0.6 → +0.6% |
| 1.5 m | −1.0 → −1.0% | +0.4 → +0.6% | −0.2 → 0.0% |
| 1.9 m | +1.0 → +0.7% | +0.5 → +0.2% | −0.3 → −0.8% |
| 2.3 m | +1.1 → +0.2% | +1.2 → +0.3% | +0.4 → −0.8% |
| 2.9 m | — | +0.4 → −0.9% | +3.6 → +1.3% |
| 3.6 m | — | −1.2 → −3.2% | +5.6 → +3.7% |

### What the numbers say

- **DA3's own focal length is 11% too long on every walk.** Metric depth grows with the focal length, so every depth would read 11% long, while sideways positions (depth ÷ focal) stay put: the map would be stretched along each viewing direction, not scaled evenly. The room-line estimate measures the focal length from straight edges instead: only the right value lets three perpendicular directions explain most of them. It lands within 2.4%.
- **With the right focal length, the metric model still reads about 7% short, and consistently so** (1.062–1.075). A fixed gain of 1.07 leaves at most 1.1% on the walk it was not fitted on.
- **A small bias by range remains.** Video depth reads a few percent long up close and short far away. The correction log(LiDAR/video) = a + b·log(depth), with a = −0.0119 and b = 0.0253, helps most at 2–3 m, where room walls are seen. It makes the nearest bin worse on two walks, and c00a170fe1 keeps a 3% offset below 1.2 m. Whether it helps the plans is judged on the plans themselves (next section).
- **Caveats.** The three walks cover two flats and one phone, so the scale interval keeps a 2% allowance for the calibration itself. The Stray video is ARKit's unstabilised 4:3 stream, not a Camera-app clip.

## Video tier: plans against LiDAR (`bench/video_vs_lidar.py`, `bench/calibrate_intervals.py`)

```bash
COZMO_DATA=/path/to/captures python bench/video_vs_lidar.py --variants oracle     # ~20 min with cached model outputs
python bench/calibrate_intervals.py
```

Each walk's video goes through the video tier, and its plan is compared with the LiDAR plan of the same walk. Rooms are paired after a local alignment. Walls are paired corner to corner. The 3% gate (G-WALL-VIDEO) is scored on LiDAR walls of at least 1 m whose two neighbouring faces were fitted to wall points. The `oracle` variant keeps the video tier's depth but uses ARKit's camera path, to separate the two sources of error.

### Results (code `1d00698`)

These are also the shipped video tier's numbers. The fix loop's change made them worse and is switched off, and the re-run with it off reproduces these video plans exactly ([`docs/fix_loop.md`](../docs/fix_loop.md)).


| Walk | Variant | Rooms (video / LiDAR) | Rooms paired | Footprint | Gate walls within 3% | Paired walls, typical error | Interval held the LiDAR length | Wall-map agreement |
|---|---|---|---|---|---|---|---|---|
| c00a170fe1 | video | 3 / 3 | 1 | −18.6% | 0 of 6 (2 paired) | 8.5% | 1 of 2 | 0.47 |
| c00a170fe1 | ARKit path | 2 / 3 | 0 | −10.0% | — | — | — | 0.88 |
| 1a8384c3f6 | video | 7 / 7 | 5 | −9.4% | 0 of 17 (3 paired) | 15.5% | 0 of 8 | 0.55 |
| 1a8384c3f6 | ARKit path | 6 / 7 | 2 | +6.2% | 3 of 8 (6 paired) | 2.9% | 6 of 6 | 0.82 |
| c7d28f72c6 | video | 12 / 9 | 0 | −4.9% | — | — | — | 0.39 |
| c7d28f72c6 | ARKit path | 11 / 9 | 7 | −22.2% | 1 of 23 (11 paired) | 14.6% | 4 of 11 | 0.77 |

### What the numbers say

- **The video tier does not meet G-WALL-VIDEO.** Paired walls are typically 8–16% off, mostly short. c7d28f72c6's −4.9% footprint is a coincidence: its plan is scrambled, and no room pairs with LiDAR.
- **The camera path is one cause.** With ARKit's path, wall maps agree far better with LiDAR (0.77–0.88 against 0.39–0.55). On 1a8384c3f6, wall lengths become unbiased: typical error 2.9%, every interval holds. Tracking is lost where the phone turns while facing a blank wall up close.
- **Building rooms from noisy depth is the other cause.** Even with ARKit's path, c7d28f72c6's walls read about 15% short, and rooms split or merge differently from LiDAR. Plans from video depth also flip with small input changes: c00a170fe1's ARKit-path footprint moved from −21.6% to −35.4% after a correction of about 1% in depth.

### Interval calibration

Across the 9 paired video walls (each video wall counted once), the error divided by the wall's own sigma runs from 1.5 to 7.4. With 9 walls, the 90% split-conformal factor is the largest ratio: 4.49. Every video interval is therefore widened by `VIDEO_INTERVAL_SCALE = 4.5` (`cozmo/export/document.py`), so a 3 m wall reads about ±0.7 m. The evidence is thin: two flats, and no leave-one-walk-out check was possible because c7d28f72c6 has no paired walls. The factor must be re-measured after any change to the video tier.

### Options tried and not used

| Option | What it did | Decision |
|---|---|---|
| Layout tolerances scaled to wall scatter (video walls scatter 4.8 cm, LiDAR 2.0 cm) | c00a170fe1 ARKit-path footprint −35.4% → −10.0% | **Used** |
| Counting a wall seen anywhere within a scatter-wide band | Found a 3.7 m wall that was missed, but also turned furniture into walls: c00a170fe1 ARKit-path footprint −10.0% → −34.4% | Not used |
| Cutting the walk at tracking breaks and dropping the frames inside long ones | Lost 21% of c00a170fe1's frames: footprint −18.6% → −65.2% | Not used |
| Cutting at tracking breaks, keeping frames, ignoring the depth of very unsure frames | Wall-map agreement 1a8384c3f6 0.55 → 0.63, c7d28f72c6 0.39 → 0.49, but footprints no better (c7d28f72c6 −12.4% → −24.6%) | Not used; breaks are reported as a warning |
| Ignoring the depth of very unsure frames alone | c00a170fe1 −18.6% → −15.6%, 1a8384c3f6 −9.4% → −13.6% | Not used |

Tracking breaks are detected from DA3's confidence (median below 2.0 in either frame of a pair). This flags every pair of frames whose rotation error exceeds 20 degrees, with 25 false alarms in 111 pairs on c00a170fe1 and 40 in 343 on 1a8384c3f6. The drift step can hold such breaks loosely and search wider for loop closures across them. On a synthetic walk turned 25 degrees and shifted 25 cm at a break, median distortion was 41.7 cm undeclared, 24.2 cm declared, and 0.3 cm declared with the wider search (`tests/test_drift.py`). It is not used for video because the plans did not improve.

## Depth bias correction for LiDAR ceilings (a separate improvement, not the scored fix loop)

```bash
python bench/arkitscenes_planes.py                                          # before: depth as recorded
python bench/arkitscenes_planes.py --bias-correction leave-one-venue-out    # after
```

Device depth reads 9–16 mm shorter than laser depth on every walk (first section). Each walk's device depth now gets the offset measured on the **other** venue's walks, so no walk is corrected with its own ground truth. The prediction was recorded in the commit message before the run (`c159802`): 6 of 6 walks within 15 mm, mean about −6 mm.

| Walk | Venue | Offset added | Ceiling height error, before | After | Floor error, before | After |
|---|---|---|---|---|---|---|
| 41069048 | 381644 | +11.3 mm | -28.1 mm | -11.3 mm | +16.1 mm | +5.7 mm |
| 41069050 | 381644 | +11.3 mm | -26.9 mm | -9.1 mm | +14.3 mm | +3.4 mm |
| 41069051 | 381644 | +11.3 mm | -23.6 mm | -5.4 mm | +10.7 mm | -0.0 mm |
| 41142278 | 384651 | +12.5 mm | -20.9 mm | -2.7 mm | +7.4 mm | -3.1 mm |
| 41142280 | 384651 | +12.5 mm | -11.2 mm | +9.5 mm | +12.1 mm | -0.2 mm |
| 41142281 | 384651 | +12.5 mm | -19.8 mm | -0.4 mm | +11.0 mm | -1.1 mm |
| **Within 15 mm (G-CEIL)** | | | **1/6** | **6/6** | | |
| **Mean** | | | -21.8 mm | -3.2 mm | | |

Before: code `33e470f`. After: code `c159802`. Device bias on all six walks: -11.9 mm.

- **G-CEIL now passes on all six walks**, as predicted. The mean error is −3.2 mm against the predicted −6 mm: the correction did slightly more than the viewing-angle argument suggested.
- **The spread gate did not improve.** Spread across walks: before 381644: 9.2 mm, 384651: 42.6 mm; after 381644: 10.1 mm, 384651: 43.9 mm. Venue 381644 moved from just inside the 10 mm gate to 0.1 mm outside it. One offset per venue cannot change the spread by much; the small movement comes from plane fits over slightly moved points. Venue 384651 stays confounded by a ceiling with more than one level (first section).
- **The calibration comes from a 2020 iPad Pro.** It is not yet verified on an iPhone 15 Pro. Until a tape-measured room confirms it, the pipeline keeps the full 13 mm bias in its ceiling interval.

## Stitching rooms into one plan (`bench/stitch_benchmark.py`)

```bash
python bench/houselayout_properties.py      # once: 23 test cases, 130 rooms, from HouseLayout3D
python bench/stitch_benchmark.py            # about a minute
```

This tests the photo tier's stitch (G-PHOTO-STITCH, A-ADJ) separately from room reconstruction. Real buildings are cut into rooms that doors join. Each room is handed to the solver in its own frame, the way the photo tier reports it: a random quarter turn plus a small turn, a random shift, scale error, a jittered outline, noisy door widths and positions, missed doors and false doors. Levels are **exact** (no noise), **photo** (3% scale, 5 cm door noise, 10% missed and 5% false doors) and **hard** (twice that). Two random framings per case.

A door pair counts as right only when both sides are the same real door. Rooms can be joined through the wrong doors and still count as adjacent, and that misplaces them.

### The solver as committed (code `ac83422`): beam 128, no shared-wall or compactness score

| Level | Runs | All doors right | Door precision / recall | Adjacency exact | One connected plan | Room placement error (median) | Time per case |
|---|---|---|---|---|---|---|---|
| exact | 46 | 40/46 | 0.939 / 0.939 | 40/46 | 46/46 | 0.026 m | 0.03 s |
| photo | 46 | 14/46 | 0.556 / 0.607 | 15/46 | 22/46 | 1.633 m | 0.01 s |
| hard | 46 | 8/46 | 0.404 / 0.485 | 9/46 | 17/46 | 1.318 m | 0.01 s |

### How the settings were chosen

Beam width (shared-wall weight 0.5):

| Setting | Exact: all doors right | Exact: door precision | Photo: all doors right | Photo: door precision | Photo: placement (m) | Hard: door precision |
|---|---|---|---|---|---|---|
| beam 8 | 22/46 | 0.718 | 11/46 | 0.508 | 1.586 | 0.437 |
| beam 32 (exact only) | 22/46 | 0.752 | — | — | — | — |
| beam 128 | 26/46 | 0.78 | 13/46 | 0.55 | 1.425 | 0.427 |
| beam 512 | 26/46 | 0.784 | 13/46 | 0.551 | 1.425 | 0.427 |

Shared-wall weight (beam 128):

| Setting | Exact: all doors right | Exact: door precision | Photo: all doors right | Photo: door precision | Photo: placement (m) | Hard: door precision |
|---|---|---|---|---|---|---|
| shared wall 0.5 | 26/46 | 0.78 | 13/46 | 0.55 | 1.425 | 0.427 |
| shared wall 0.25 | 26/46 | 0.788 | 13/46 | 0.528 | 1.425 | 0.451 |
| shared wall 0.1 | 30/46 | 0.827 | 14/46 | 0.566 | 1.633 | 0.428 |
| shared wall 0 (committed) | 40/46 | 0.939 | 14/46 | 0.556 | 1.633 | 0.404 |

Compactness weight (beam 8, shared wall 0.5):

| Setting | Exact: all doors right | Exact: door precision | Photo: all doors right | Photo: door precision | Photo: placement (m) | Hard: door precision |
|---|---|---|---|---|---|---|
| compactness 0 | 22/46 | 0.718 | 11/46 | 0.508 | 1.586 | 0.437 |
| compactness 0.5 | 16/46 | 0.525 | 9/46 | 0.431 | 1.53 | 0.367 |
| compactness 1.0 | 10/46 | 0.388 | 9/46 | 0.369 | 1.784 | 0.308 |
| compactness 0.5, shared wall 0.2 | 14/46 | 0.528 | 8/46 | 0.424 | 1.53 | 0.371 |

- **A wider beam** recovers from early wrong choices; beyond 128 nothing changes.
- **The shared-wall bonus misled door choices.** A room attached next to two rooms through a wrong or imagined door out-scored the true door. Without the bonus, zero-noise runs get every door right 40 times in 46, against 26. With noise the bonus helped placement slightly (1.43 against 1.63 m), but not door choices. It is removed, and the unit test for an imagined door now passes.
- **Compactness** fixed one case (JmbYfDe2QKZ) and hurt overall.

### What the numbers say

- **Without noise, the solver rejoins buildings through the right doors in 40 of 46 runs**, with rooms placed within a few centimetres.
- **G-PHOTO-STITCH is not met.** With photo-like noise it builds one connected plan in 22/46 runs and gets every door right in 14/46, with rooms placed about 1.633 m off. Missed doors split the plan into islands, and noisy widths make doors interchangeable.

## Photo tier: room boxes against LiDAR (`bench/photo_vs_lidar.py`)

```bash
python bench/make_photo_sets.py --sweep c00a170fe1 1a8384c3f6     # stand-in photo sets, once
COZMO_DATA=/path/to/captures python bench/photo_vs_lidar.py       # about 5 min
```

We have no real iPhone photo sets with tape truth. The stand-in sets are stills cut from our walks, one folder per LiDAR room, taken as overlapping views while turning where possible (the protocol's sweep). Each photo room is compared with the LiDAR room it came from: the box's two dimensions against the room's extents along its own axes. The stills carry no EXIF, so the true focal length stands in for the EXIF value. Code `71cfbac-dirty`, interval factor 5.1.

| Room | Photos | Walls seen | Box dimensions, photo vs LiDAR | Area error | Doors found |
|---|---|---|---|---|---|
| c00a170fe1 room_1 | 4 | 3/4 | 2.47 vs 3.45 m (-29%), 3.68 vs 4.71 m (-22%) | -22% | 1 |
| c00a170fe1 room_2 | 6 | 3/4 | 1.87 vs 2.43 m (-23%), 2.14 vs 3.04 m (-30%) | -46% | 1 |
| 1a8384c3f6 room_1 | 4 | 3/4 | 1.23 vs 5.49 m (-78%, interval missed), 1.84 vs 6.15 m (-70%) | -88% | 1 |
| 1a8384c3f6 room_2 | 3 | 3/4 | 1.21 vs 3.35 m (-64%), 2.34 vs 3.64 m (-36%) | -70% | 1 |
| 1a8384c3f6 room_3 | 4 | 3/4 | 2.35 vs 2.64 m (-11%), 4.76 vs 3.14 m (+52%) | +56% | 1 |
| 1a8384c3f6 room_4 | 3 | 2/4 | 1.60 vs 1.72 m (-7%), 2.33 vs 3.10 m (-25%) | -30% | 1 |
| 1a8384c3f6 room_5 | 6 | 3/4 | 2.16 vs 1.72 m (+26%), 2.61 vs 3.10 m (-16%) | +6% | 1 |
| 1a8384c3f6 room_6 | 4 | 2/4 | 0.97 vs 1.17 m (-17%), 2.02 vs 3.51 m (-43%) | -49% | 1 |
| 1a8384c3f6 room_7 | 4 | 3/4 | 1.67 vs 1.52 m (+10%), 2.00 vs 1.72 m (+16%) | +28% | 1 |

- **G-WALL-PHOTO is not met:** 1 of 18 dimensions within 8%, and boxes are mostly too small. A box from a few views stops at the farthest wall it saw well. The stand-in sets were not shot from doorways, so the side closed at the camera is often too close.
- **Footprint:** c00a170fe1: 13.07 m² vs 19.03 m² (-31%), interval [3.55, 48.21]; 1a8384c3f6: 30.93 m² vs 52.7 m² (-41%), interval [10.9, 87.79].
- **Stitch:** c00a170fe1: 1 door pair(s) join 2 rooms into 1 group(s); 1a8384c3f6: 3 door pair(s) join 7 rooms into 4 group(s). G-PHOTO-STITCH is not met. The stitch benchmark below shows the solver needs doors on both sides to be seen.
- **Intervals:** 18 dimensions, z = error / model sigma, sorted: [0.22, 0.29, 0.38, 0.54, 0.84, 1.06, 1.45, 1.58, 1.88, 2.54, 2.83, 4.1, 5.24, 5.48, 5.91, 7.1, 8.19, 27.58]. PHOTO_INTERVAL_SCALE = 5.1 is the smallest factor under which the nominal 90% interval holds on at least 90% of them (in-sample, the rule used for video). With it the shipped intervals hold on 17 of 18 dimensions and on both footprints. The finite-sample split-conformal rule would take the single worst dimension, a 1 m corridor boxed as 3 m: 16.77x.

## Damage detection checks (`bench/damage_sanity.py`)

```bash
COZMO_DATA=/path/to/captures python bench/damage_sanity.py        # about 5 min
```

No damaged room was available, so A-DMG-DETECT (staged damage, two classes) cannot be scored yet. Two checks we can run:

| Walk (undamaged) | Images | Tiles | Highest damage score | Tiles over 0.6 | Painted stains found |
|---|---|---|---|---|---|
| c00a170fe1 | 37 | 444 | 0.546 | 0 | 2 of 10 |
| 1a8384c3f6 | 115 | 1380 | 0.545 | 0 | 2 of 29 |
| c7d28f72c6 | 215 | 2580 | 0.763 | 8 | 12 of 54 |

- **False alarms are rare but not zero:** 8 of 4404 tiles pass the threshold, all on c7d28f72c6. A region also needs two views on one surface: `cozmo run` on all three walks, at the LiDAR and video tiers, gives 0 damage regions (`bench/results/timing.json`).
- **Recall is low:** painted stains were found in 16 of 93 images. Painted stains are not real damage, so this only shows the detector can fire. The threshold was chosen for few false alarms; real recall needs staged damage.

## Opening widths, walk against walk (`bench/same_flat_openings.py`)

```bash
COZMO_DATA=/path/to/captures python bench/same_flat_openings.py     # about 2 min
```

No tape truth, so this checks repeatability. Both walks of the same flat go through `cozmo run` (LiDAR). The second plan is placed on the first by one rigid fit, and openings are paired when their centres are within 0.3 m (code `4c49191`).

- **Detection does not repeat:** 23 openings in `c7d28f72c6`, 12 in `1a8384c3f6`, 7 pairs.
- **Most widths are not measured:** only 1 pair was measured jamb to jamb in both walks, and those widths differ by 13.0 cm. The others were walked through with a jamb out of view, so they carry the typical 0.80 m with a ±0.25 m interval. All 7 pairs' intervals overlap, so the uncertainty is reported honestly.
- **G-OPEN (≤ 2 cm on 85%) would fail.** The next steps are seeing both jambs (the protocol's walk-through step) and edge refinement on the images.

## Damage detector on real defect photos (`bench/damage_bd3.py`)

```bash
python bench/damage_bd3.py        # downloads the 157 MB test split once; about 3 min
```

BD3 (Kottari and Arjunan, 2024) has phone photos of building walls taken about 1 m away, one defect label each. We use the 793-image test split of its CC-BY-4.0 re-release on Hugging Face (`chandrabhuma/building_defect_vqa`). It is used locally and never redistributed. The detector runs exactly as shipped (12 tiles, threshold 0.6). BD3 classes map to ours: stain → water stain, algae → mold, cracks → crack, peeling → peeling paint, spalling → hole.

| BD3 class | Images | Flagged as damage | Right class |
|---|---|---|---|
| algae | 123 | 48% | 52 (42%) |
| major_crack | 113 | 87% | 93 (82%) |
| minor_crack | 117 | 89% | 100 (85%) |
| peeling | 105 | 59% | 3 (3%) |
| plain | 123 | 6% | — (false alarms) |
| spalling | 100 | 45% | 10 (10%) |
| stain | 112 | 44% | 22 (20%) |

- **Overall at the shipped threshold:** 62% of damaged photos flagged, 42% with the right class, and 6.5% of plain walls flagged.
- **Cracks are found well** (about 88%, class right about 83%).
- **Peeling paint is almost never named as peeling.** It is flagged, but as another class. Stains and spalling are found less than half the time.
- **Threshold trade-off:** 0.3: 93% found, 26.8% false alarms / 0.4: 87% found, 18.7% false alarms / 0.5: 76% found, 11.4% false alarms / 0.6: 62% found, 6.5% false alarms / 0.7: 48% found, 3.3% false alarms. We keep 0.6: our undamaged walks gave 0 false regions with it.
- **Caveat:** this checks classes on close-up photos. It does not check metric extent in a room; that still needs a staged-damage capture (A-DMG-DETECT).

## LiDAR wall-to-wall distances against laser truth (`bench/arkitscenes_wall_distances.py`)

```bash
python bench/arkitscenes_wall_distances.py        # about 10 min
```

Our layout cannot run on ARKitScenes scans: the phone stays within about 1 m, and rooms are grown from the walk. So this measures what the layout is built on, where the sensor puts the walls. On each of the six walks, wall planes are found in the laser-rendered depth and in the device depth of the same frames and poses, so pose errors cancel. Two planes facing each other across the room give a wall-to-wall distance, the reading a laser measurer takes (code `5016e5c`).

| Walk | Laser distance | Device error, as recorded | Device error, corrected (+11.9 mm) |
|---|---|---|---|
| 41069048 | 1.029 m | -0.28 cm | +1.42 cm |
| 41069051 | 3.337 m | -1.26 cm | +0.95 cm |
| 41142278 | 1.809 m | -2.78 cm | -0.97 cm |
| 41142280 | 1.828 m | -3.33 cm | -1.22 cm |

- **As recorded:** 2/4 within max(2 cm, 1%), median -2.0 cm. The sensor reads walls 10.6 mm too close (median over walks).
- **With the shipped correction:** **4/4** within the gate, median -0.0 cm, |error| p90 1.36 cm; walls sit 3.8 mm close. This matches the ceiling result: the sensor reads short, and the correction removes it.
- **Caveats:**
  - Only 4 distances: these small scans rarely see two opposite walls well.
  - This tests the sensor and fusion, not our wall snapping and corners. Those still need tape on our own rooms.

