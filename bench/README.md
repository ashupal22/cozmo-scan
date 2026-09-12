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

### History of this benchmark

- **First run (`ea62760`): the correction barely engaged.** No wall directions were used on two walks, no loops were accepted on two walks, and a table top used as floor moved one walk 44 cm.
- **Fixes (`2720656`, `1807cd0`, `b6b144c`):** looser wall-direction threshold, floor gating, a submap merge bug, loops matched around the map centre, heading drift modelled as a steady creep, ARKit snaps tied to the walk start, and loops matched on first-pass-straightened maps.
- **Metric fix (`5fdba7f`):** the two-walk comparison became symmetric. The one-way version rewarded blurred maps.
