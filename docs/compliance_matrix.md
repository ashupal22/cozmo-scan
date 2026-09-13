# Compliance matrix

Every requirement in the brief → where it lives → what it is → status.

**Status key**
- **Met:** the requirement is met, with evidence.
- **Built, gate not met:** the capability exists and is benchmarked, but misses its target. The number is given.
- **Partial:** some of the requirement is done; the rest is said.
- **Not done:** missing, with the reason.

Paths are relative to the repo root. Results are in `bench/results/`. Our gate IDs are defined in [`docs/gates.md`](gates.md).

## Part 1: capture and tiers

| # | Requirement (brief) | File path | Artifact | Status |
|---|---|---|---|---|
| 1.1 | Capture route: a stock capture protocol, one page, followable by a non-engineer | `docs/capture_protocol.md`, `docs/capture_protocol.pdf` (one A4 page) | Route 2. Stray Scanner (LiDAR) and the iPhone Camera app (video, photos), with install, walk, duration, what to avoid and hand-off | **Met** |
| 1.2 | Three input tiers, all mandatory, same output contract | `cozmo/pipeline.py` (`run_lidar`, `run_video`, `run_photos`), `cozmo/ingest/detect.py` | `cozmo run <capture>` detects the tier. All three write the same schema, validated on every run | **Met** (runs), accuracy per tier below |
| 1.3 | Photos: 2–8 stills per room, per-room folders, stitched whole-property plan, wider intervals | `cozmo/photo/room.py`, `cozmo/stitch/solver.py`, `bench/results/timing.json` | Room box per folder from Depth Anything 3, joined through doors; intervals widened 5.1× | **Partial**: runs end to end, but stitching often leaves several groups (G-PHOTO-STITCH below) |
| 1.4 | Video: handheld walkthrough, iPhone 15 or newer | `cozmo/video/`, `bench/results/timing.json` | Key frames → DA3 poses and depth → metric scale → drift correction → layout | **Met** (runs); accuracy fails G-WALL-VIDEO |
| 1.5 | LiDAR: depth, poses, intrinsics on Pro devices | `cozmo/ingest/stray.py`, `cozmo/geometry/`, `cozmo/slam/`, `bench/results/timing.json` | Fusion, depth-bias correction, drift correction, walls-first layout | **Met** (runs) |
| 1.6 | Intervals widen honestly as sensor data thins | `cozmo/export/document.py`, `bench/calibrate_intervals.py`, `bench/photo_vs_lidar.py` | LiDAR: measured error model. Video ×4.5 (holds on 6 of 7 walls). Photo ×5.1 (holds on 17 of 18). Unseen values are marked `observed: false` with a wide range | **Met** on our benchmark; thin evidence (two flats) |
| 1.7 | Device matrix: tier by hardware, and the accuracy each tier honestly delivers | `docs/device_matrix.md`, `README.md` | Table with measured accuracy and gate status per tier | **Met** |

## Part 2: output contract

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 2.1 | Dimensioned per-room plan: walls, ceiling height, floor area, openings | `cozmo/export/document.py`, `schema/output.schema.json` | `rooms[]`: polygon, walls with lengths, floor area, perimeter, ceiling height, openings with widths | **Met**, every tier. Unseen ceilings are flagged `observed: false`. Photo door widths are typical, not measured |
| 2.2 | Stitched multi-room plan with correct adjacency | `cozmo/geometry/layout.py` (LiDAR, video: one map), `cozmo/stitch/solver.py` (photo) | `plan.adjacency`, `plan.stitch` | **Partial**: LiDAR and video give one connected map. Photo joins 2 of 2 rooms on one flat but leaves 7 rooms in 4 groups on the other |
| 2.3 | Per-surface damage regions with class and metric extent | `cozmo/damage/detect.py` | `damage[]`: surface, class (5), area, width, height, height above floor, outline, confidence, views | **Partial**: built for every tier. On 793 real defect photos (BD3, `bench/results/damage_bd3.json`): 62% flagged, 42% with the right class, 6.5% false alarms on plain walls; cracks about 88%, peeling rarely named right. 0 false regions on our 3 undamaged walks. Damage staged synthetically on our real walk (`bench/staged_damage.py`): 0/2 found; a visible stain and a 1.2 m crack raised their tiles' scores (0.23 → 0.68 (mold); 0.36 → 0.63 (crack)) but in one frame each, and a region needs two. Metric extent in a real room untested |
| 2.4 | Concealed-damage flags with the rule that fired | `cozmo/damage/rules.py`, `docs/damage_rules.md` | `concealed_flags[]` with `rule_id`, description and evidence; 6 rules | **Met** as logic (unit tested). Fires only when damage is detected |
| 2.5 | Scope line items keyed to surfaces | `cozmo/damage/scope.py` | `scope[]` with `surface_id`, unit, quantity with interval, and `because` (damage or rule) | **Met** as logic (unit tested) |
| 2.6 | A confidence interval on every measurement | `schema/output.schema.json` (`measurement` requires `ci_low`, `ci_high`, `confidence`) | Nominal 90% intervals; log-scale when very uncertain, so they never go below zero | **Met** |
| 2.7 | One command per capture | `cozmo/cli.py` | `cozmo run <capture>` | **Met** |
| 2.8 | JSON to the published schema | `schema/output.schema.json`, `cozmo/export/validate.py` | Every run is validated before it is written; `cozmo validate` checks any file | **Met** (schema is ours: Cozmo did not publish one) |
| 2.9 | Rendered plan | `cozmo/export/render.py` | `plan.svg` written next to `result.json` by every run | **Met** |
| 2.10 | Stitched plan from every tier, including photos | as 2.2 | | **Partial** (see 2.2) |

## Part 2: benchmark set composition

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| B.1 | One multi-room capture: 3 or more rooms plus a connector | `data/README.md` (Google Drive) | `c7d28f72c6`, `1a8384c3f6`: a whole flat with a T-shaped hallway, 7–9 rooms | **Met** |
| B.2 | One furnished room with staged damage, two damage classes | `bench/damage_bd3.py`, `bench/staged_damage.py` (substitutes) | Classes on 793 real defect photos (BD3); a water stain and a crack staged synthetically at known size on our real walk: 0/2 found | **Not done** as specified: no damaged room was available. The substitutes test classes and the in-room path, not a real staged room |
| B.3 | The same rooms at all three tiers, photo tier as per-room folders | `bench/make_photo_sets.py`, `data/derived/` | Video and photo inputs come from the same walks: the Stray Scanner RGB stream, and stills cut from it into per-room folders | **Partial**: same rooms, but not separate iPhone 15 video and photo captures |
| B.4 | At least one room captured twice at the same tier | `bench/same_flat_plans.py` | Same flat walked twice (LiDAR) | **Met** |
| B.5 | Laser or tape ground truth on everything; raw sensor data and measurements submitted | `docs/datasets.md`, Google Drive | Raw Stray Scanner data (depth, confidence, poses, IMU, video) on Drive. Laser truth from ARKitScenes for LiDAR floor and ceiling | **Partial**: no tape measurements of our own rooms. Accuracy on them is measured against LiDAR, not tape |

## Part 2: gates

| # | Gate | File path | Result | Status |
|---|---|---|---|---|
| G-OPEN | Opening widths ≤ 2 cm on ≥ 85%, missed and phantom openings count | `cozmo/geometry/layout.py`, `bench/same_flat_openings.py` | No tape truth. Walk against walk on the same flat: 23 vs 12 openings found, 7 paired, only 1 measured jamb to jamb in both (widths 13.0 cm apart); the rest are a typical 0.80 m ±0.25 m. Intervals of all 7 pairs overlap | **Not met** (detection not repeatable; most widths not measured) |
| G-CEIL | Ceiling height ≤ 1.5 cm per room | `bench/arkitscenes_planes.py`, `arkitscenes_planes_bias_corrected.json` | 6 of 6 ARKitScenes walks within 15 mm after depth-bias correction (mean −3.2 mm) | **Met** on public laser data. iPhone transfer unverified |
| G-CEIL-SPREAD | Spread across captures ≤ 1 cm; say whether biased or unrepeatable | same | Before correction: repeatable but biased (−21.8 mm). After: venue 381644 spread 10.1 mm, venue 384651 43.9 mm (a two-level ceiling) | **Not met**: 0.1 mm over on one venue |
| A-WALL-LIDAR | Wall lengths ≤ max(2 cm, 1%) (our gate) | `bench/arkitscenes_wall_distances.py` | Laser truth on ARKitScenes, sensor and fusion only: 4/4 wall-to-wall distances within the gate after depth correction (2/4 as recorded), median error -0.0 cm. Only 4 distances; our layout's wall lengths are not tested (no tape of our rooms) | **Partial** (thin evidence, layout untested) |
| G-REPEAT | Same room twice: every wall within 1 cm or 0.5% | `bench/same_flat_plans.py` | Footprints agree within 1.6% (shipped pipeline, `drift_footprint.json`). 3 of 14 room dimensions within the gate (`same_flat_plans.json`). The best room is within 0.6 cm on both dimensions | **Not met** |
| G-DRIFT | Method stated, plus a footprint ablation with correction on and off | `cozmo/slam/`, `bench/drift_ablation.py`, `bench/drift_footprint.py`, `bench/README.md` | Loop closures, a 4-DoF pose graph, wall-direction and floor priors. Footprint ablation with the shipped pipeline: the two walks of the same flat differ 4.4% off → 1.6% on; sharper maps; same-flat wall agreement p90 5.99 → 2.78 cm | **Met** |
| G-PHOTO-STITCH | One stitched plan, correct adjacency, no overlaps, footprint ±8% | `bench/photo_vs_lidar.py`, `bench/stitch_benchmark.py` | Photo footprints −31% and −41%. On HouseLayout3D the solver gets all doors right in 40 of 46 noise-free runs and 14 of 46 with photo-like noise | **Not met** |
| G-WALL-PHOTO | Wall lengths ±8%, calibrated intervals | `bench/photo_vs_lidar.py` | 1 of 18 box dimensions within 8%; intervals hold on 17 of 18 | **Not met** (calibrated) |
| G-WALL-VIDEO | Wall lengths ±3%, calibrated intervals | `bench/video_vs_lidar.py` | 0 of 22 gate walls within 3%; paired walls 8–16% off; intervals hold on 6 of 7 | **Not met** (calibrated); this is the fix-loop gate |
| Calibration | Scored at every tier; no confident garbage | `bench/calibrate_intervals.py`, `bench/photo_vs_lidar.py`, `bench/results/arkitscenes_planes_bias_corrected.json`, `bench/results/arkitscenes_wall_distances.json` | Video ×4.5 holds 6 of 7 walls; photo ×5.1 holds 17 of 18 dimensions. LiDAR: every wall-length and ceiling interval is at least ±3.0 cm (two faces × the 13 mm bias allowance). On laser truth it holds 6 of 6 ceilings (largest error 1.1 cm) and 4 of 4 wall-to-wall distances (largest 1.4 cm), so it is conservative on purpose until an iPhone room is taped | **Partial**: LiDAR checked on 10 public laser-truth values, not our own rooms |

## Part 3: head-to-head

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 3.1 | 2 benchmark rooms, LiDAR tier vs one consumer app (named, versioned, export submitted), error by dimension, beat or tie on ≥ 70% | — | — | **Not done**: needs the rooms re-captured with an app such as Polycam or magicplan plus tape truth. We had no device at the end. Protocol ready: `docs/benchmark_report.md` |

## Part 4: fix loop

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 4.1 | One-page declaration: worst gate with the failing number, root-cause hypothesis and evidence, fix and predicted number | `docs/fix_loop.md` sections 1–4 | Committed before the fix (`3d54e0d`). G-WALL-VIDEO: 0 of 23 gate walls; predicted 2 | **Met** |
| 4.2 | Ship the fix | `7bbd7c5`, `63e77ed` | Tracking breaks declared, quarter-turn loop search, unsure depth ignored | **Met** (shipped, then switched off after it measured worse) |
| 4.3 | Before run, after run, both regenerable, readable diff | `bench/results/fix_loop/`, `docs/fix_loop.md` "How to regenerate", `fix.diff` | Before, after #1, after #2, two ablations, post-mortem log | **Met** |
| 4.4 | Say why it fell short | `docs/fix_loop.md` sections 5–8 | Prediction badly wrong. Post-mortem: the loops never engaged, and loosening the breaks freed stretches that had been nearly right | **Met** (gate not moved) |

## Part 5 and deliverables

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 5.1 | Commit as you work | git history | 70+ commits, each with its own evidence; predictions committed before results | **Met** |
| D.1 | Compliance matrix | `docs/compliance_matrix.md` | This file | **Met** |
| D.2 | Capture route and device matrix | `docs/capture_protocol.md` (+ `.pdf`, one A4 page), `docs/device_matrix.md` | One-page protocol, checked by rendering it (`scripts/report_pdf.py docs/capture_protocol.md 1`), plus the matrix | **Met** |
| D.3 | README to running on a fresh capture in < 15 min on a clean machine, one command per capture | `README.md`, `scripts/install.sh`, `scripts/fetch_models.py` | Tested in a fresh venv: install 87 s with warm caches; at `caefd8c` all 114 tests pass there, and LiDAR and photo captures run. Cold download about 3.3 GB | **Met** on macOS/Apple silicon; Linux untested |
| D.4 | Reproduction bundle: everything needed to regenerate every number from raw inputs; caches must replay and the live path must run | `bench/reproduce.sh`, `scripts/fetch_external.py`, `data/README.md`, `data/captures_manifest.json`, `bench/compare_results.py`, `docs/reproduction_check.md` | One script for every benchmark, and model caches keyed by input content. Re-run from a clean clone: 8 of 9 benchmarks give the same numbers, and video gives the same measurements with 20 interval bounds wider (log-scale intervals since `4f5aa4e`). The live path regenerates the caches | **Met** (caches regenerate; they are not shipped) |
| D.5 | Benchmark report: gates at all three tiers, repeatability table, head-to-head table, timing | `docs/benchmark_report.md`, `bench/README.md` | Tables with source files | **Partial**: the head-to-head table is empty (3.1) |
| D.6 | Fix loop bundle | `docs/fix_loop.md`, `bench/results/fix_loop/`, `bench/fix_loop_postmortem.py` | Declaration, runs, diff, post-mortem | **Met** |
| D.7 | Technical report, at most 6 pages | `docs/technical_report.md` (and `.pdf`) | Architecture, tiers, device matrix, drift, error budget, calibration, fix loop, failure modes | **Met** |
| D.8 | Raw benchmark data: sensor logs, ground truth, app exports | Google Drive (`data/README.md`), `data/captures_manifest.json`, `docs/datasets.md` | Our raw Stray Scanner captures (depth, confidence, poses, IMU, video), with SHA-256 checksums anyone can check their Drive copies against (`scripts/capture_manifest.py --check`); public laser truth fetched by script | **Partial**: no tape truth, no app exports |

## Constraints

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| C.1 | Handheld consumer capture only; any pretrained model or dataset with disclosure; runs without our infrastructure | `README.md` "Models and data used" | DA3 (Apache 2.0), CLIP (MIT), ARKitScenes, HouseLayout3D; runs locally | **Met** |
| C.2 | Weights and large binaries fetched by script | `scripts/fetch_models.py`, `scripts/fetch_external.py` | Hugging Face cache; nothing large in git | **Met** |
| C.3 | Mirrors, glass, wet-look surfaces, low light: covered in the submission | `cozmo/damage/detect.py` (`scene_conditions`, `low_light`), `docs/capture_protocol.md`, `docs/technical_report.md` | Every run warns when images show a mirror, glass or a wet-look floor (CLIP), or when 30% or more are dark. Protocol steps cover lights and mirrors, and the report lists the failure modes. On our walks the flags are real (a glass partition, a TV screen). Geometry near them is flagged, not corrected | **Met** (warnings), correction not built |
| W.1 | Walk-in test: all three tiers ready to run cold on a new capture | `README.md`, `docs/walk_in.md`, `bench/results/timing.json` | All three tiers run from a fresh install, 23 s to 2.3 min per capture; a cold 37 s video takes 6.3 min, and long clips are capped at 120 key frames. Each run writes `summary.md` (room width × length, ceiling, door widths, with ranges) to read against the laser | **Met** (readiness); accuracy as above |
