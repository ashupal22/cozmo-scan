# Walk-in test runbook

What we do at the defense, on the demo laptop (Apple silicon Mac, set up with `bash scripts/install.sh`).

## Before the examiners arrive

1. `cd cozmo-scan && source .venv/bin/activate && pytest -q`: 112 tests, about 30 s.
2. Warm the models so the first run does not pay for loading: `python scripts/fetch_models.py`. The weights are already cached, so this is instant.
3. Check free disk space (a video run writes key frames and model outputs, about 1 GB for 3 minutes of video) and plug in power.

## When the capture arrives

| Tier they choose | What we receive | Command | Expected time on an M4 |
|---|---|---|---|
| LiDAR | Stray Scanner export folder (AirDrop or Files) | `cozmo run <folder>` | 30 s for a 1-room walk, about 2 min for a whole flat |
| Video | One `.mov` (AirDrop) | `cozmo run <file>.mov` | about 10 s per second of video on first run |
| Photo | Photos (AirDrop), sorted into one folder per room | `cozmo run <folder>` | about 15–20 s per room |

`--no-damage` saves 20–60 s if time is short. Every run writes `out/<name>/result.json` and `plan.svg`. Open the SVG in a browser to show the plan.

## Reading the output with the examiners

- **Rooms:** each room lists its walls (`R1.W1`, ...) with a length and a 90% interval. Compare each laser reading with the interval, not only with the value.
- **Warnings:** `quality.warnings` says what was not seen: a ceiling (its height is then a prior, `observed: false`), a wall behind the photographer, a tracking break in a video. These are the places to expect larger errors.
- **Tier trust:** LiDAR is the measuring tier. Video and photo intervals are widened 4.5× and 5.1×, because those tiers miss their gates (`docs/benchmark_report.md`).
