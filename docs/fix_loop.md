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
