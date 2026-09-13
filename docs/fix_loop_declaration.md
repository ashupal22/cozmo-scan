# Fix declaration (one page)

Condensed, with no number or claim changed, from the declaration committed before the fix was built (commit `3d54e0d`, `docs/fix_loop.md` sections 1–4). The outcome and post-mortem are in sections 5–8.

## 1. The worst-performing gate

**G-WALL-VIDEO, video wall lengths within ±3%: 0 of 23 gate walls pass** (`bench/results/video_vs_lidar.json`, code `1d00698`, against the LiDAR plans of the same three walks).

| | c00a170fe1 | 1a8384c3f6 | c7d28f72c6 | All |
|---|---|---|---|---|
| Gate walls within 3% | 0 of 6 | 0 of 17 | no room paired | **0 of 23** |
| Rooms paired with LiDAR | 1 of 3 | 5 of 7 | 0 of 9 | 6 of 19 |
| Wall-map agreement with LiDAR | 0.47 | 0.55 | 0.39 | |

## 2. Root cause hypothesis and evidence

**Hypothesis.** Depth Anything 3 loses track where the phone turns facing a blank wall up close. At some tracking breaks the heading comes out 60–90° wrong, and drift correction can't repair it:
- wall directions only measure heading modulo 90°;
- loop closures search only ±8°;
- breaks are not declared, so the creep model resists a sudden change.

**Evidence.**
1. With ARKit's camera path and the same video depth, wall-map agreement rises from 0.39–0.55 to 0.77–0.88.
2. At 13 breaks the median rotation error is 7.2°, but 3 breaks are off by 90.4°, 61.9° and 86.6°.
3. Inside a 12-frame run DA3 is accurate, about 1–2 cm and 1° per frame pair; the damage is done at the joins.
4. On a synthetic walk, a declared 25° break is repaired (43.7 → 0.3 cm), a 62° break snaps to the wrong quarter turn, and a 90° break is left as it was.
5. Ruled out: joining the frames either side of a break in one DA3 run is much worse (71.4° against 7.2°), and denser key frames did not help.

Not addressed by this fix: room building on noisy depth (c7d28f72c6's walls read about 15% short even with ARKit's path).

## 3. The fix

1. Declare tracking breaks (DA3 confidence below 2.0) to drift correction, and hold them loosely.
2. Search every quarter turn for loop closures across a break, and start the pose graph from the turn the loops agree on.
3. Ignore depth from very unsure frames (median confidence below 1.2).

## 4. Predicted numbers after the fix

| | Before | Predicted |
|---|---|---|
| **Gate walls within 3%** | 0 of 23 | **2** (1–4); G-WALL-VIDEO still fails |
| Paired walls, typical error | 15.3% | 10% (7–12%) |
| Rooms paired | 6 of 19 | 9 (8–12) |
| Agreement c7d28f72c6 / 1a8384c3f6 / c00a170fe1 | 0.39 / 0.55 / 0.47 | 0.60 / 0.63 / unchanged ±0.05 |

**Why it will fall short of the gate:** room building on video depth remains, breaks the walk never revisits have no loop to close, and position errors across breaks are corrected only where loops exist.
