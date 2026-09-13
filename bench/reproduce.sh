#!/usr/bin/env bash
# Regenerate every number in bench/README.md, docs/benchmark_report.md and docs/fix_loop.md (except the fix-loop runs
# at older commits: see docs/fix_loop.md). Needs COZMO_DATA (our captures, Google Drive) and the public data
# (python scripts/fetch_external.py arkitscenes; python scripts/fetch_external.py houselayout3d).
# About 1 hour on an Apple M4 with cached model outputs; longer the first time.
set -euo pipefail
: "${COZMO_DATA:?set COZMO_DATA to the folder holding c00a170fe1, 1a8384c3f6 and c7d28f72c6}"
cd "$(dirname "$0")/.."
python bench/arkitscenes_planes.py                                          # LiDAR floor/ceiling vs laser, raw depth
python bench/arkitscenes_planes.py --bias-correction leave-one-venue-out    # ... with the depth-bias correction
python bench/drift_ablation.py                                              # G-DRIFT: footprint with drift on and off
python bench/same_flat_plans.py                                             # repeatability: same flat, two walks
python bench/video_scale.py                                                 # video focal and scale calibration
python bench/video_vs_lidar.py --variants oracle                            # video plans vs LiDAR (G-WALL-VIDEO)
python bench/calibrate_intervals.py                                         # video interval factor
python bench/make_photo_sets.py --sweep c00a170fe1 1a8384c3f6               # stand-in photo sets (if missing)
python bench/photo_vs_lidar.py                                              # photo plans vs LiDAR (G-WALL-PHOTO)
python bench/houselayout_properties.py                                      # stitch test cases from HouseLayout3D
python bench/stitch_benchmark.py                                            # stitch solver (G-PHOTO-STITCH, A-ADJ)
python bench/damage_sanity.py                                               # damage detector checks
