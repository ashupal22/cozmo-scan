# Walk-in test runbook

What we do at the defense, on the demo laptop (Apple silicon Mac). The examiners capture a space we have
never seen, choose the tier on the day, and measure it with a laser while the pipeline runs.

## The evening before

```bash
cd cozmo-scan && bash scripts/install.sh      # creates .venv and installs pinned dependencies
source .venv/bin/activate
pytest -q                                     # 121 tests, about 30 s
python scripts/fetch_models.py                # DA3 weights into the local cache; instant when already there
df -h .                                       # need 5 GB free: a video run writes about 1.2 GB
```

Then rehearse: run one capture of each tier end to end, offline (`HF_HUB_OFFLINE=1`), and time it. The first
invocation of the day must never be the examiners'.

## Opening, before anything runs

Say the failures first. They are in `docs/benchmark_report.md` and they will be found anyway:

> LiDAR is our measuring tier and it holds: ceiling within 15 mm on 6 of 6 public walks with laser truth,
> and two walks of the same flat agree to 1.6% on footprint. Video and photo miss their wall gates, and our
> fix-loop prediction was wrong — the fix is shipped but switched off, and the post-mortem says why.
> Every number carries an interval widened from measured error, not guessed. Please check the intervals,
> not only the values.

This is the ground we are actually strong on. The brief caps the score for confident garbage on thin input;
being the team whose ranges mean what they say is the thing we can win today.

## When the capture arrives

1. `cozmo inspect <path>` — one second. Shows the tier it detected and the capture summary.
2. **Say the prediction out loud before the laser touches a wall.** LiDAR: walls seen well within about 2 cm (laser-truth
   scans: 4 of 4 wall-to-wall distances within 1.4 cm), ceiling within 1.5 cm if the ceiling was seen; the main risk is
   a room split differently from how they measure it. Video: expect several percent out, and the interval will be wide enough to say so. Photo: wider still.
3. Run it:

| Tier | What arrives | Command | Cold time on an M4 |
|---|---|---|---|
| LiDAR | Stray Scanner export folder (AirDrop) | `cozmo run <folder>` | 22–30 s for a 3-room flat, about 100 s for a whole flat |
| Video | one `.mov` (AirDrop) | `cozmo run <file>.mov` | about 6 s per second of clip; about 13 min for a 2 minute clip |
| Photo | photos sorted into one folder per room | `cozmo run <folder>` | about 15 s per room |

Video keeps every key frame up to 2 minutes of clip (360 frames). A 120-frame cap was measured and dropped: our 115 s walk collapsed to 1 room (-86% footprint).
`--no-damage` saves 20–60 s if the clock is against us.

4. Narrate while it runs: fuse depth into points → drift correction → floor → walls-first layout → rooms split
   at doorways → openings → damage → JSON and SVG.
5. Open `report.pdf`: the plan, every room's width × length, ceiling height and door widths with ranges, damage and
   the warnings, on one page. Read each value **with its interval** before they measure.
6. Show `result.json`: one wall, one opening, one damage region, one concealed flag with the rule that fired,
   one scope line. That is the output contract, item by item.
7. Show `bench/results/drift_*_on.svg` against `_off.svg`. Pre-computed — never run the ablation live.

## If it fails

Say "the fallback is in the pipeline", and run the same capture one tier down. A Stray export contains the
video, so a LiDAR failure re-runs as `cozmo run <folder>/rgb.mp4`. Rehearse the sentence and the command. That
`rgb.mp4` is stored sideways, and this fallback is not benchmarked (our video results use an upright copy), so
rehearse it the evening before.

## Reading the output with them

- **Rooms**: walls (`R1.W1`, ...) with a length and a 90% interval. Compare the laser reading with the interval.
- **Warnings**: `quality.warnings` names what was not seen — an unobserved ceiling (`observed: false`), a wall
  behind the photographer, a tracking break, glass or a wet-look floor in view. These are where to expect error,
  and we said so before they measured.
- **Tier trust**: LiDAR is the measuring tier. Video and photo intervals are widened 4.5x and 5.1x because those
  tiers miss their gates (`docs/benchmark_report.md`).

## Questions to have answers ready for, tools closed

- **Why does video miss by so much?** Mostly the camera path: with ARKit's path and the same video depth, wall-map
  agreement rises from 0.39–0.55 to 0.76–0.88. Room building on video depth still loses up to about 23% of footprint
  (−11.1%, +2.0%, −22.6%), so the depth matters too.
- **Your fix-loop prediction was wrong. What happened?** `docs/fix_loop.md` — root cause, the shipped fix, the
  measured result, and why it is switched off.
- **Why walls first?** Repeatability. Rooms reach their walls behind furniture, so two walks of the same flat
  went from 10.2% to 1.9% apart on footprint.
- **What do you do about drift?** Pose graph over wall directions and loop closures, with the on/off ablation.
