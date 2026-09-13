# Reproduction check

Every benchmark we report was re-run from a clean clone of the repository at commit `a2a1604`, with the same raw
captures (`COZMO_DATA`, checked against `data/captures_manifest.json`) and the same public data and model weights.
The regenerated result files were compared with the committed ones by `bench/compare_results.py`. It ignores run
times and commit ids, and numbers must agree to a relative 1e-6.

**8 of 9 benchmarks regenerated the same numbers.** The ninth, video, regenerated the same measurements, errors and counts, with wider interval bounds. Commits after `a2a1604` change documents and add `summary.md` to each run, not the benchmark code.

| Command | Committed result | Time (Apple M4) | Result |
|---|---|---|---|
| `python bench/stitch_benchmark.py` | `stitch_benchmark.json` | 31 s | **same numbers** |
| `python bench/same_flat_openings.py` | `same_flat_openings.json` | 95 s | **same numbers** |
| `python bench/arkitscenes_planes.py --bias-correction leave-one-venue-out` | `arkitscenes_planes_bias_corrected.json` | 445 s | **same numbers** |
| `python bench/drift_footprint.py` | `drift_footprint.json` | 130 s | **same numbers** |
| `python bench/photo_vs_lidar.py` | `photo_vs_lidar.json` | 37 s | **same numbers** |
| `python bench/video_vs_lidar.py --variants oracle` | `fix_loop/ablation_none_video_vs_lidar.json` | 461 s | same measurements, errors and counts; 20 interval bounds wider (see below) |
| `python bench/damage_bd3.py` | `damage_bd3.json` | 98 s | **same numbers** |
| `python bench/arkitscenes_wall_distances.py` | `arkitscenes_wall_distances.json` | 437 s | **same numbers** |
| `python bench/staged_damage.py` | `staged_damage.json` | 159 s | **same numbers** |

## Differences

- `video_vs_lidar_head.json` against `fix_loop/ablation_none_video_vs_lidar.json`: 20 values differ, 20 of them interval bounds. Every measured value, error and count is the same. Whether each interval holds the LiDAR value is unchanged too. The committed file was produced before commit `4f5aa4e`. That commit
  takes an interval on a log scale when its sigma exceeds 25% of the value, so lengths and areas never go below zero,
  which makes wide video intervals asymmetric: for example, a footprint interval moved from 21.18–77.97 to 30.32–81.04 m².
  The regenerated file is committed as `bench/results/video_vs_lidar_head.json`, and `docs/benchmark_report.md` now
  reads it.

Model outputs (Depth Anything 3, CLIP) are cached by input content under `data/derived/`, and these runs replay them.
A fresh run recomputes them, and DA3 on Apple's GPU is not bit-identical (`bench/results/timing_cold_video.json`).

## Not re-run here

- `bench/arkitscenes_planes.py` without bias correction, `bench/drift_ablation.py`, `bench/same_flat_plans.py`: they
  record earlier stages of the LiDAR tier and use the same code paths as the rows above.
- `bench/video_scale.py`, `bench/calibrate_intervals.py`, `bench/houselayout_properties.py`: these produce inputs
  (calibration constants, stitch test cases) whose effects are covered by the rows above.
- The fix-loop before and after runs: they regenerate at their own commits (`docs/fix_loop.md`, "How to regenerate").
- `bench/damage_sanity.py`: not run, to save time.

## Repeat it

```bash
git clone https://github.com/ashupal22/cozmo-scan.git repro && cd repro
export COZMO_DATA=/path/to/captures      # then place data/derived and data/external as in README.md
bash bench/reproduce.sh                  # regenerates bench/results/
python bench/compare_results.py bench/results --committed /path/to/original/bench/results
```
