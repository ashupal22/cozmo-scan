# Gates: pass/fail targets

The brief says *"Round 1 gates apply, with five additions"* and *"JSON to the published schema"*. Cozmo has not provided the Round 1 gates or schema and will not share more material, so we define both ourselves. This file lists every target we test against, and marks each one as coming **from the brief** or as **our decision**. Every decision has a reason, and the technical report repeats them.

Output format: [`schema/output.schema.json`](../schema/output.schema.json) (v0.1.0, our own). Example: [`schema/example_output.json`](../schema/example_output.json).

## How errors are measured

- **Error** = |our value − laser or tape measurement|, for the same element and measured the same way.
- **Opening width** = clear width between the inner jamb faces, trim excluded.
- **Wall length** = inside face, corner to corner.
- **Footprint** = sum of room floor areas (inside faces).
- **Intervals** are nominal 90% (`confidence: 0.9` in the JSON).
- **Repeatability tolerance** for a wall of length L = max(1 cm, 0.5% × L). This reads the brief's "1 cm or 0.5%" as "whichever is looser"; see decision 2 below.

## From the brief

| ID | Gate | Tier | Target | Brief |
|---|---|---|---|---|
| G-OPEN | Opening widths | LiDAR | Error ≤ 2 cm on ≥ 85% of openings. A missed or phantom opening counts as a miss | p. 2 |
| G-CEIL | Ceiling height | LiDAR | Error ≤ 1.5 cm per room | p. 2 |
| G-CEIL-SPREAD | Ceiling repeatability | LiDAR | Spread across captures of the same room ≤ 1 cm. Report says whether we are biased or unrepeatable | p. 2 |
| G-REPEAT | Wall repeatability | Same tier, 2 captures | Every wall agrees within max(1 cm, 0.5%) | p. 2 |
| G-DRIFT | Drift accountability | Video, LiDAR | Method stated, plus a footprint ablation with correction on and off. Poses used as-is = fail | p. 2 |
| G-PHOTO-STITCH | Whole-property stitch | Photo | One plan from per-room folders, correct adjacency, no room overlaps, footprint within ±8% with calibrated intervals | p. 2 |
| G-WALL-PHOTO | Wall lengths | Photo | Within ±8%, calibrated intervals | p. 2 |
| G-WALL-VIDEO | Wall lengths | Video | Within ±3%, calibrated intervals | p. 2 |
| G-H2H | Head-to-head | LiDAR, 2 rooms | Beat or tie a consumer app on ≥ 70% of shared dimensions | p. 3 |
| G-CONTRACT | Output contract | All | Per-room plan, stitched plan, damage regions, concealed flags with rule, scope keyed to surfaces, interval on every measurement, one command, JSON, rendered plan | p. 1 |
| G-INSTALL | Fresh setup | All | README to a first result on a clean machine in < 15 min | p. 4 |

## Our decisions

Round 1 gates were not provided, so these targets are ours.

| ID | Gate | Tier | Target | Why this value |
|---|---|---|---|---|
| A-WALL-LIDAR | Wall lengths | LiDAR | Error ≤ max(2 cm, 1%) | Photo and video gates are "looser", so LiDAR must be tighter than 3%. Matches reported incumbent accuracy of 1–3 cm |
| A-AREA | Room floor area | LiDAR / video / photo | ≤ 2% / 6% / 8% | Area error is about twice the linear error; photo is capped at the brief's 8% footprint gate |
| A-ADJ | Adjacency | All | Every true connection found, no false ones, no overlaps (> 0.05 m²) | Brief requires "correct adjacency" at every tier |
| A-DMG-DETECT | Damage detection | All | Every staged damage region found with the correct class; overlap with the ground-truth region (IoU on the surface plane) ≥ 0.5; at most 1 false region per room | Brief requires class and metric extent, with no threshold given |
| A-DMG-AREA | Damage area | LiDAR | Within ±25% | Small regions with soft edges; tighten once measured |
| A-FLAG-RULE | Concealed flags | All | 100% of flags name a rule that exists in the rules file | Brief: "with the rule that fired" |
| A-SCOPE-KEY | Scope items | All | 100% of items reference an existing surface and a damage region or rule | Brief: "keyed to surfaces" |
| A-CALIB | Calibration | Each tier | Nominal 90% intervals contain the truth 85–95% of the time; intervals widen from LiDAR to video to photo | Brief: calibration scored at every tier; confident garbage caps the score |
| A-SCHEMA | JSON validity | All | 100% of outputs validate against the schema, and ci_low ≤ value ≤ ci_high | Brief: "JSON to the published schema" |
| A-DETERMINISM | Same input, same output | All | Running twice on the same capture gives identical JSON (except runtime) | Needed for G-REPEAT and a regenerable fix loop |
| A-RUNTIME | Runtime | All | ≤ 10 min per capture on an Apple M4 laptop; video about 13 min for the protocol's 2 minute clip | The walk-in test runs live while the examiners measure. Revised for video: a 120-frame cap met 10 min but collapsed the plan (bench/README.md), and the brief scores accuracy, not runtime |

## How we read unclear parts of the brief

1. **Opening and ceiling gates at the photo and video tiers.** The brief only loosens wall lengths for those tiers. We apply G-OPEN and G-CEIL strictly at the LiDAR tier. At the photo and video tiers we still report the same errors, and those intervals must be calibrated (A-CALIB). We never hide a failure: the benchmark shows the numbers against the LiDAR thresholds too.
2. **"Within 1 cm or 0.5% per wall"** means whichever is looser: max(1 cm, 0.5% × L). We also report how many walls pass the stricter reading, min(1 cm, 0.5% × L).
3. **Footprint** means the sum of room floor areas (inside faces). We also report the outer outline area including walls, so either reading can be checked.
4. **Round 1 contract items** are taken as the list in the brief's Part 2 (G-CONTRACT). The schema covers each item, and the compliance matrix maps each one to a file.
