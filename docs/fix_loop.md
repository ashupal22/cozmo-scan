# Fix loop declaration

Written and committed before the fix is built or benchmarked, so the prediction below is on record ahead of the result.

## 1. The worst-performing gate

**G-WALL-VIDEO: video wall lengths within ±3%.** Measured against the LiDAR plans of the same walks (`bench/results/video_vs_lidar.json`, code `1d00698`):

| | c00a170fe1 | 1a8384c3f6 | c7d28f72c6 | All |
|---|---|---|---|---|
| Gate walls within 3% | 0 of 6 | 0 of 17 | — (no room paired) | **0 of 23** |
| Paired walls, typical error | 8.5% (2 walls) | 15.5% (8 walls) | — | 15.3% |
| Rooms paired with LiDAR | 1 of 3 | 5 of 7 | 0 of 9 | 6 of 19 |
| Wall-map agreement with LiDAR | 0.47 | 0.55 | 0.39 | |

The other failing gates are further from a fix this round. The photo tier's gates are not built yet. The LiDAR ceiling gate (1 of 6 walks within 15 mm) has a proven cause and is fixed separately, outside this loop.

## 2. Root cause and evidence

**Hypothesis.** The video tier's camera path breaks where Depth Anything 3 loses track: the phone turns while facing a blank wall up close. At some breaks the heading comes out 60–90° wrong, and drift correction cannot see or repair that:
- wall directions only measure heading modulo 90°
- loop closures search only ±8° of turn
- the shipped pipeline does not declare breaks, so the steady-creep heading model resists any sudden change

Whole stretches of the walk then sit turned by up to a quarter turn. Walls seen before and after a break do not meet, and rooms are built from mismatched walls: split, merged and short.

**Evidence.**
1. **Same depth, true camera path.** With ARKit's path and the video tier's depth, wall-map agreement rises from 0.39–0.55 to 0.77–0.88. On 1a8384c3f6, wall lengths become unbiased: typical error 2.9%, 3 of 8 gate walls within 3%.
2. **At the tracking breaks themselves** (13 breaks on c00a170fe1 and 1a8384c3f6), the chain's rotation error has median 7.2°, but 3 breaks are off by 60–90° (90.4°, 61.9°, 86.6°).
3. **Inside a run of 12 key frames, DA3 is accurate**: 1–2 cm, about 1° per frame pair. The damage is done at the joins.
4. **Synthetic walk with a declared break** (`tests/test_drift.py` helpers). A 25° break is repaired (median distortion 43.7 → 0.3 cm). A 62° break is snapped to the wrong quarter turn (+89.9°; 103.5 → 140.6 cm). A 90° break is left as it is (142.4 → 140.6 cm).
5. **Ruled out.** Joining the confident frames on either side of a break with one DA3 run is much worse than the chain (median 71.4° against 7.2°), because the two sides of a blank-wall turn do not overlap. Denser key frames through a turn did not keep DA3 on track either.

**Not addressed by this fix.** Room building on noisy depth: with ARKit's path, c7d28f72c6's walls still read about 15% short.

## 3. The fix

1. **Declare tracking breaks.** The video tier passes its tracking breaks (DA3 confidence below 2.0) to drift correction. Drift correction holds them loosely and without creep continuity. The machinery exists and is tested; it is switched off in the shipped pipeline.
2. **Search every quarter turn across a break.** Loop closures between places on different sides of a break search all four quarter turns, not only ±8°, and the pose graph is started from the quarter turn the loops agree on. Wall directions then fix the remaining small heading error, and loops the position.
3. **Drop unsure depth.** Depth from frames DA3 is very unsure about (median confidence below 1.2) is not fused. Their ghost walls turn a loosely held stretch the wrong way (c00a170fe1: cutting at breaks without this, footprint −18.6% → −69.6%).

## 4. Prediction

Measured by `bench/video_vs_lidar.py` on the same three walks:

| | Before | Predicted after |
|---|---|---|
| **Gate walls within 3%** | 0 of 23 | **2** (range 1–4). **G-WALL-VIDEO still fails** |
| Paired walls, typical error | 15.3% | 10% (7–12%) |
| Rooms paired with LiDAR | 6 of 19 | 9 (8–12), mostly on c7d28f72c6 (0 → 3–5) |
| Wall-map agreement, c7d28f72c6 | 0.39 | 0.60 (0.50–0.70) |
| Wall-map agreement, 1a8384c3f6 | 0.55 | 0.63 (0.58–0.70) |
| Wall-map agreement, c00a170fe1 | 0.47 | unchanged ±0.05: a 14 m walk with few revisits, so few loops to close |

Caveat on the metric: gate walls are counted in paired rooms only, so pairing more rooms also adds walls to the count.

**Why it will fall short of the gate.** Room building on video depth remains (evidence 1, c7d28f72c6). Breaks that the walk never revisits have no loop to close. Position errors across breaks (0.2–1.5 m) are corrected only where loops exist.

## How to regenerate

```bash
git checkout 1d00698 && COZMO_DATA=/path/to/captures python bench/video_vs_lidar.py      # before
git checkout <fix commit> && COZMO_DATA=/path/to/captures python bench/video_vs_lidar.py  # after
git diff 1d00698 <fix commit> -- cozmo/                                                   # the change
```

Model outputs are cached under `data/derived/<walk>_video_work/cache/`, and both runs replay them identically.

---

# Fix loop outcome and post-mortem

Written after the runs. Everything below can be regenerated (see the end).

## 5. Result: the prediction was badly wrong

The fix shipped in `7bbd7c5`, and run #1 collapsed: footprints −65/−59/−59%, 1 room paired. The cause was an implementation error. `tracking_breaks` still dropped the frames inside long breaks (24, 37 and 73 frames), which the earlier ablation had already shown to be harmful. `63e77ed` corrected only that (frames are never dropped), with no tuning, and run #2 is the after run.

| | Before (`1d00698`) | Predicted | After #2 (`63e77ed`) |
|---|---|---|---|
| **Gate walls within 3%** | 0 of 23 | 2 (1–4) | **0 of 6** (only one room left to count walls in) |
| Rooms paired with LiDAR | 6 of 19 | 9 (8–12) | **1 of 19** |
| Footprint error c00a / 1a83 / c7d2 | −18.6 / −9.4 / −4.9% | better | **−18.8 / −35.3 / −48.2%** |
| Wall-map agreement c00a / 1a83 / c7d2 | 0.47 / 0.55 / 0.39 | 0.47 / 0.63 / 0.60 | 0.47 / 0.49 / 0.48 |

G-WALL-VIDEO still fails, and every other number moved the wrong way. The prediction was badly wrong.

## 6. Post-mortem: why it went wrong

**Which part did the damage.** The same code was re-run on the same walks with each part switched on or off (`bench/results/fix_loop/ablation_*`, code `321d5d5` plus the switches). With the fix off, the video plans come out identical to the before run's (footprints 19.82, 49.57 and 53.02 m², same room counts), so the cached model outputs replay exactly. Only the LiDAR reference moved, because it now includes the +11.9 mm depth correction:

| Setting | Footprint c00a / 1a83 / c7d2 | Rooms paired | Gate walls within 3% |
|---|---|---|---|
| Fix off (the before behaviour) | −19.5 / −13.0 / −5.4% | 6 of 20 | 0 of 22 |
| Part 3 only: ignore depth of unsure frames | −16.6 / −17.0 / −12.9% | 5 of 20 | 1 of 20 |
| Parts 1–3: the shipped fix (after #2) | −18.8 / −35.3 / −48.2% | 1 of 19 | 0 of 6 |

Declaring the breaks (parts 1 and 2) did the damage. Part 3 on its own is mixed: the first gate wall ever within 3%, but two footprints got worse.

**Why declaring breaks hurt.** `bench/fix_loop_postmortem.py` compares each walk's heading with ARKit's, with breaks declared and without:
1. **The quarter-turn search never engaged.** Loops across breaks were proposed 1, 109 and 178 times on the three walks, and 0, 5 and 27 were accepted. They never agreed on a turn for any stretch, so no quarter turn was applied on any walk. The synthetic test had a clean loop at every break. Our real walks rarely look at the same place from both sides of a break.
2. **Declaring a break also throws away what held the walk together there.** Before the fix, most breaks were nearly right: the median error over 13 breaks was 7.2°, and only 3 were 60–90° off. Held loosely, the stretches were free to turn, and wall directions (blind modulo 90°) let them settle a quarter or half turn off. On c7d2 the second stretch went from 1° off to 178° off, and on 1a83 the fifth from 96° to −170°. We gave up correct information at ten breaks to fix three, and fixed none.

**Was the root cause right?** Partly. Camera-path errors are the main loss: with ARKit's path, wall-map agreement is 0.77–0.88 against 0.39–0.55. But the mechanism we predicted, loops across breaks, needs revisits our walks do not have.

**What we should have done.** Count the loops available across each real break before predicting. Evidence 4 was synthetic, and its walks revisited everything.

## 7. What ships

The fix is switched off (`DECLARE_TRACKING_BREAKS = False`, `IGNORE_UNSURE_DEPTH = False`), which restores the before behaviour, the best we measured. The code and its tests stay, and the after run regenerates at any commit with the switches turned on. G-WALL-VIDEO stays failed. The video intervals stay widened 4.5×, and they hold on 6 of 7 paired gate walls in the fix-off run.

The next attempt should re-attach a stretch only where evidence says it is turned: a loop closure that disagrees with the chain, or a big turn in the phone's gyroscope. Held stretches should never be loosened on DA3 confidence alone.

## 8. The ceiling fix (separate from the scored loop)

For G-CEIL the order was the same: prediction committed first (`c159802`, 14:04: "6 of 6 walks within 15 mm, mean ceiling height error about −6 mm"), then the run (`7dac386`, 14:15). Result: **6 of 6 walks within 15 mm, mean −3.2 mm** (before: 1 of 6, −21.8 mm). There the root cause was measured directly: device depth reads 12 mm short of laser depth on every walk. That is why the prediction held. It has shipped since `ac83422` (`LIDAR_DEPTH_OFFSET_M`).

## How to regenerate every run

```bash
export COZMO_DATA=/path/to/captures        # the three Stray Scanner folders (Google Drive, data/README.md)
git checkout 1d00698 && python bench/video_vs_lidar.py --variants oracle --out before.json   # before
git checkout 63e77ed && python bench/video_vs_lidar.py --variants oracle --out after.json    # after #2
git checkout main   # ablation: set cozmo.pipeline.DECLARE_TRACKING_BREAKS / cozmo.video.capture.IGNORE_UNSURE_DEPTH
```

The readable diff of the fix and its correction is `bench/results/fix_loop/fix.diff` (`git diff 7dac386 63e77ed -- cozmo/video cozmo/slam cozmo/pipeline.py tests`). Code commits in between touch only the stitch solver.
The committed results are `bench/results/fix_loop/{before,after,after2,ablation_none,ablation_unsure_only}_video_vs_lidar.json`.
