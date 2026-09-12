# Gates: pass/fail targets

The brief says *"Round 1 gates apply, with five additions"* and *"JSON to the published schema"*. Neither the Round 1 gates nor the schema were provided. This file lists every target we test against, and marks each one as coming **from the brief** or as **our assumption**. If Cozmo sends the real Round 1 gates or schema, update this file and `schema/output.schema.json`. Nothing else should need to change.

Output format: [`schema/output.schema.json`](../schema/output.schema.json) (v0.1.0, stand-in). Example: [`schema/example_output.json`](../schema/example_output.json).

## How errors are measured

- **Error** = |our value − laser or tape measurement|, for the same element and measured the same way.
- **Opening width** = clear width between the inner jamb faces, trim excluded.
- **Wall length** = inside face, corner to corner.
- **Footprint** = sum of room floor areas (inside faces).
- **Intervals** are nominal 90% (`confidence: 0.9` in the JSON).
- **Repeatability tolerance** for a wall of length L = max(1 cm, 0.5% × L). This reads the brief's "1 cm or 0.5%" as "whichever is looser"; see the open questions below.

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

## Our assumptions

Replace these if Cozmo sends the Round 1 gates.

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
| A-RUNTIME | Runtime | All | ≤ 10 min per capture on an Apple M4 laptop | The walk-in test runs live while the examiners measure |

## Open questions for Cozmo

1. Please share the Round 1 JSON schema and the Round 1 gates.
2. Do the opening (G-OPEN) and ceiling (G-CEIL) gates also apply at the photo and video tiers? The brief only loosens wall lengths for those tiers.
3. Does "within 1 cm or 0.5% per wall" mean whichever is looser (our reading) or whichever is tighter?
4. Is the footprint the sum of room floor areas (our reading) or the outer outline including walls?
