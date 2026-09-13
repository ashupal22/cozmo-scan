# cozmo-scan: technical report

Cozmo AI Applied AI case study, Round 2. Repo: `ashupal22/cozmo-scan`. Every number below comes from a script in `bench/` and a result file in `bench/results/`; the [compliance matrix](compliance_matrix.md) maps each requirement to its evidence.

## 1. Summary

One command (`cozmo run <capture>`) turns a Stray Scanner LiDAR export, an iPhone video or per-room photo folders into the same JSON (our schema, validated on every run) and a plan drawing. The JSON holds rooms, walls, openings, ceiling heights, the stitched plan, damage regions, concealed-damage flags and a repair scope, with a 90% interval on every number.

| | LiDAR | Video | Photo |
|---|---|---|---|
| Runs cold from a fresh install | yes | yes | yes |
| Accuracy we can show | Ceiling within 15 mm on 6/6 laser-truth walks; wall-to-wall distances within 1.36 cm of laser on 4 of 4 pairs; same flat walked twice agrees within 1.6% footprint | Walls 8–16% off LiDAR; 0 of 22 gate walls within 3% | Room boxes 26% off (median); 1 of 18 within 8% |
| Intervals hold on our benchmark | ceiling interval carries the 13 mm bias | 6 of 7 walls (×4.5) | 17 of 18 dimensions (×5.1) |
| Gates met | G-CEIL (public data), G-DRIFT | — | — |

The honest position: the LiDAR tier is the product today; the video and photo tiers run end to end and say how wrong they may be, but miss their gates. The fix loop targeted the worst gate (G-WALL-VIDEO), shipped a fix, measured it worse, and switched it off with a post-mortem (section 7). What we could not do without a device and a tape measure: tape ground truth, staged damage, and the head-to-head against a consumer app.

## 2. Architecture

```
capture ─► ingest ─► per-frame depth + camera poses ─► fusion ─► drift correction ─► floor / ceiling planes
            │          LiDAR: sensor depth, ARKit poses        (points with normals)    (pose graph)
            │          video: Depth Anything 3 runs, chained
            │          photo: Depth Anything 3 per room folder
            ▼
   walls-first room layout ─► openings ─► document + error model ─► damage ─► rules ─► scope ─► JSON + SVG
   (photo: one box per room, then the stitch solver joins rooms through their doors)
```

- **Ingest** (`cozmo/ingest`) detects the tier. LiDAR: Stray Scanner's 256×192 depth, confidence, ARKit poses and intrinsics. Video: key frames at 3 fps. Photo: EXIF orientation and the 35 mm-equivalent focal length (fx/width = f35 × diagonal / (43.27 mm × width)).
- **Fusion** (`cozmo/geometry/fusion.py`) back-projects confident depth into world points with normals, dropping depth edges.
- **Drift correction** (`cozmo/slam`) is described in section 4.
- **Planes**: the floor and ceiling are the horizontal planes with the largest covered area, not the most points, which rejects table tops (a table top once moved a floor by 221 mm).
- **Walls-first layout** (`cozmo/geometry/layout.py`): wall faces are fitted from points seen above an adaptive height, the plan is split into inside and outside by a min cut over line cells, faces are paired into walls, corners snapped, rooms split where the floor narrows at a doorway, and openings kept only with proof (the walk passed through, or floor was seen at the line). If the walls are not at right angles it falls back to rooms traced from the seen floor.
- **Document** (`cozmo/export`): every measurement is a value with `ci_low`, `ci_high`, `confidence` and, when the sensor never saw it, `observed: false`.

## 3. Tier design and device matrix

**LiDAR.** The phone's depth is trusted and ARKit's poses are corrected (section 4). On ARKitScenes, whose laser scans are registered to every frame, device depth reads 9–16 mm short on every walk (median −11.9 mm), so 11.9 mm is added to every depth pixel along its ray (`LIDAR_DEPTH_OFFSET_M`). Measured leave-one-venue-out, this took ceiling height from 1/6 to 6/6 walks within 15 mm. On the same scans the distance between opposite walls went from -2.0 cm to -0.0 cm (median), all 4 measurable pairs within max(2 cm, 1%) (`bench/arkitscenes_wall_distances.py`; sensor and fusion only, the layout cannot run on these scans). The calibration comes from a 2020 iPad Pro; until a tape-measured iPhone room confirms it, ceiling intervals keep the full 13 mm bias term.

**Video.** No sensor depth and no poses, so both come from Depth Anything 3 (DA3), Apache-licensed models only.
- **Runs of 12 key frames**, sharing 4 frames with the next run, go through DA3-BASE, with cameras solved from its ray output. Inside a run DA3 is accurate: 1–2 cm and about 1° per frame pair. Longer runs fold opposite white walls together.
- **Focal length** comes from the room's straight lines (a Manhattan vanishing-point fit), because DA3's own focal is 11% too long on all three walks. The room-line focal is within +1.1% to +2.4%.
- **Scale** comes from DA3METRIC (metres = output × focal / 300) with a calibrated gain of 1.07 and a small range correction. Measured leave-one-walk-out against LiDAR, the residual is −1.1% to +0.8%.
- **Chaining**: runs are chained through their shared frames with per-run log scales solved jointly (Huber), levelled on floor and wall normals, and passed to the same fusion, drift correction and layout as LiDAR. The layout tolerances scale with wall scatter: 4.8 cm here against 2.0 cm for LiDAR.
- **Found along the way**: Stray's video lags its poses by about 75 ms. This mattered for evaluation, not for the product.

**Photo.** Each room folder is one DA3 run (2–8 views), scaled by DA3METRIC with the EXIF focal, and levelled.
- **Room box**: the room is modelled as a right-angled box. On each side the wall is the farthest face with at least half the best coverage (nearer faces are furniture).
- **Unseen side**: a side no photo saw is closed at the camera, because the protocol has the photos taken from the doorway. It gets a 0.5 m face uncertainty.
- **Doors**: wall gaps through which the photos see floor-level surfaces beyond (windows have sills, so they do not qualify), plus the doorway the photos were taken from, at a typical width.
- **Stitching**: rooms are joined by a beam search over door pairings (`cozmo/stitch/solver.py`). Paired doors must face opposite ways a wall's thickness apart, with widths that agree; overlaps are penalised or rejected, and so are doors that open into another room's wall. Rooms no door joins are set beside the plan and flagged.

**Damage, flags, scope** (all tiers, `cozmo/damage`).
- **Damage regions**: CLIP ViT-B/32 scores 12 tiles per image against prompts for 5 damage classes and 22 undamaged things. A winning tile's depth pixels are placed on the plan with the corrected poses, assigned to the nearest wall, floor or ceiling, and measured on that surface. Tiles on one surface are merged, and two views are required.
- **Concealed-damage flags**: six rules (ceiling water, low wall water, mold, peeling paint, long crack, hole), each naming its evidence.
- **Scope**: repair line items keyed to surfaces, with quantities from the plan's intervals.

**Device matrix.**

| Tier | Hardware | Capture tool | Accuracy it honestly delivers today |
|---|---|---|---|
| LiDAR | iPhone 12 Pro and newer Pro models, iPad Pro 2020+ | Stray Scanner (free) | Ceiling ±15 mm and wall-to-wall distances ±1.36 cm (public laser data, 6 walks and 4 distances); footprint repeat 1.6%; our layout's walls not yet taped |
| Video | Any iPhone 15 or newer (1× lens, 30 fps) | Camera app | Walls typically 8–16% off; intervals about ±20% on a 3 m wall |
| Photo | Any iPhone 15 or newer | Camera app | Rooms about 26% small (median); intervals about ×/÷ 1.5 to 2 |

## 4. Drift handling

The walk is cut into 4-second submaps. Each submap measures its dominant wall direction and floor height.
- **Pose graph**: a 4-DoF pose graph (x, z, height, heading) holds the heading straight against the walls (heading drift modelled as a steady creep), removes vertical drift with the floor, and closes loops where the walk revisits a place.
- **Loop matching**: an FFT correlative search plus point-to-line ICP on first-pass-straightened maps.
- **ARKit relocalisation jumps** are used as loop closures back to the walk's start.
- **Video**: runs through the same graph. Its tracking breaks are detected (DA3 confidence) and reported, but not declared (section 7).

Our walks have no pose truth, and ARKitScenes cannot measure drift because its laser depth is rendered with the same poses. So the evidence is consistency:
- **Sharper maps**: wall-map area is 21–23% smaller with correction on every walk.
- **Real heading creep fixed**: 1a8384c3f6's 6° creep is straightened (wall-direction p90 5.10° → 1.59°).
- **Same flat agrees better**: the two walks agree within 5 cm on 86.8% of wall points, against 69.5% without correction (tile p90 5.99 → 2.78 cm).
- **Footprint ablation** (G-DRIFT) with the shipped pipeline (`bench/drift_footprint.py`): the two walks of the same flat differ by **4.4% in footprint with correction off and 1.6% with it on**. c7d28f72c6 goes from 12 rooms (walls doubled by drift) to 9.

## 5. Error budget

Typical 1-sigma sizes, and how each enters the output.

| Source | LiDAR | Video | Photo | Where it goes |
|---|---|---|---|---|
| Depth bias | 11.9 mm, corrected; 13 mm kept in ceiling intervals | — | — | ceiling, room size |
| Wall face (fit spread / √samples, plus a floor) | 2 cm scatter | 3 cm + 4.8 cm scatter | 10 cm | wall lengths from two neighbouring faces |
| Global scale | 0 | 1.5–2.5% (metric model, focal, bias) | 3% per room | every length, ×2 for areas |
| Camera path | small after drift correction | tracking breaks: 3 of 13 breaks off by 60–90° | per room only | room placement, merged or split rooms |
| Room building on noisy depth | small | c7d2 walls about 15% short even on the true path | unseen walls, few views: boxes 26% small | wall lengths, areas |
| Openings | jamb to jamb on wall points (5 cm modelled); nominal ±15 cm when a jamb is unseen | 8 cm | typical width ±15 cm | widths |

The first three rows are modelled in `cozmo/export/document.py`. The last three are not modelled, which is why the video and photo tiers need the calibration factors in section 6.

## 6. Calibration analysis

Each tier's model sigma is multiplied by a factor fitted on benchmark errors, so the nominal 90% interval holds on at least 90% of benchmark measurements. With 9 video walls this is also the split-conformal rule (the largest ratio): 4.5, and the video intervals then hold on 6 of 7 gate walls in the shipped run. For photos, 18 box dimensions give 5.1 (holds on 17 of 18). The finite-sample split-conformal rule would take the single worst dimension (a 1 m corridor boxed as 3 m) and 16.8×; we report both numbers. Where sigma exceeds 25% of a value, the interval is taken on a log scale (value ÷ f to value × f), so lengths and areas never go below zero. The asymmetry also suits our errors, which are mostly underestimates. Photo rooms have independent scales, so their areas add in quadrature in the footprint.

Caveats we state plainly:
- The evidence is thin: two flats, one phone, and LiDAR plans as the reference instead of tape.
- The factors must be re-measured after any change to a tier.
- LiDAR wall intervals are not yet checked against tape.

"Confident garbage" is avoided by construction: every unseen ceiling, wall or door width is marked `observed: false` with a wide prior.

## 7. The fix loop

**Declaration** (`docs/fix_loop.md`, committed `3d54e0d` before any fix). The worst gate was G-WALL-VIDEO: 0 of 23 gate walls within 3%, and 6 of 19 rooms paired with LiDAR. The hypothesis: DA3 loses track where the phone turns facing a blank wall up close. At some breaks the heading comes out 60–90° wrong, which the drift step cannot see, because wall directions are blind modulo 90° and loops searched only ±8°. The evidence:
- With ARKit's path, wall-map agreement rose from 0.39–0.55 to 0.77–0.88.
- 3 of 13 breaks were 60–90° off.
- A synthetic 62° break was snapped to the wrong quarter turn.

The fix had three parts: declare breaks, search every quarter turn for loops across a break, and ignore depth from very unsure frames. Predicted: 2 (1–4) gate walls within 3%, and 9 rooms paired.

**Result.** Run #1 (`7bbd7c5`) collapsed because of an implementation error (frames inside long breaks were dropped); `63e77ed` corrected only that. Run #2: gate walls 0 of 6, rooms paired 1 of 19, footprints −18.8 / −35.3 / −48.2% (before −18.6 / −9.4 / −4.9%). **The prediction was badly wrong.**

**Post-mortem.** An ablation on the same cached model outputs isolated the cause:
- **With the fix off**, the video plans reproduce the before run exactly.
- **Ignoring unsure depth alone** gave the first gate wall ever within 3% (1 of 20), but worse footprints.
- **Declaring breaks did the damage.** Per-segment heading analysis against ARKit shows no quarter turn was ever applied: loops across breaks were accepted 0, 5 and 27 times and never agreed on a turn. Meanwhile loosening every break freed stretches that had been held nearly right (median break error 7.2°), and wall directions let some settle a half turn off (one stretch went from 1° to 178° off).

The root cause was partly right: the camera path is the main loss. But the mechanism we bet on, loops across breaks, needs revisits our walks do not have. We should have counted the loops available at each real break before predicting.

**Decision.** The fix is switched off (the best measured configuration); its code and tests stay, and every run regenerates from the listed commits. The next attempt should re-attach a stretch only on positive evidence (a disagreeing loop, or the gyroscope's turn), never loosen held stretches on confidence alone.

**The ceiling loop**, run the same way but outside the scored loop, shows the process working when the cause is measured directly. Its prediction was committed first (`c159802`: 6/6 walks, mean about −6 mm), and the result matched: 6/6 walks, mean −3.2 mm.

## 8. Known failure modes

- **Blank-wall turns (video):** turning within 1 m of a plain wall breaks DA3 tracking and can turn the rest of the walk by up to a quarter turn. The protocol forbids it; the output warns when it happens.
- **Phone pointed down:** walls never seen above 1.5 m cannot be walls-first rooms, so rooms merge (a corridor joined a living room). The protocol requires a ceiling sweep in every room.
- **Mirrors and glass:** depth, from the sensor or the model, sees a room behind them. The protocol asks not to film them straight on. Every run also checks each image with CLIP and warns when it sees a mirror, glass or a wet-look floor, naming the first frame. On our walks the flags are real: a glass partition reflecting the person filming, and a glossy TV screen. The geometry is not corrected; the warning tells the reader where not to trust it.
- **Low light:** DA3 confidence drops and tracking breaks multiply; LiDAR depth is unaffected. The protocol asks for every light on, and a run warns when 30% or more of its images are dark.
- **Wet-look and shiny floors:** LiDAR dropouts leave holes, handled by area-based plane choice; the damage detector may read reflections as stains.
- **Multi-level ceilings:** one plane per room; venue 384651's spread of 43.9 mm comes from a two-level ceiling.
- **Photo tier:** boxes cannot represent L-shaped rooms or corridors. Photos not taken from the doorway leave the camera-side wall too close. Missed doors leave rooms in separate groups.
- **Damage:** zero-shot. On 793 real defect photos (BD3) it flags 62% of damaged photos, with the right class for 42%, and flags 6.5% of plain walls. Cracks are found well (about 88%); peeling paint is flagged but almost never named right, and stains and spalling are found less than half the time. Damage staged synthetically at known size on our real walk (a visible water stain and a 1.2 m crack) was not found (0/2). Each raised its tiles' damage score above the threshold in one frame only, a region needs two, and the stain read as mold. Small damage seen from across a room is missed, so the protocol asks for a close look at any damage.
- **Long videos, cold:** DA3 takes about 10 s per second of video on an M4 (378 s for a 37 s clip), so a 3-minute walk needs about 30 minutes, over our 10-minute target. Runs of cached model outputs take seconds.
- **Fresh model runs are not bit-identical:** DA3 on Apple's GPU gives slightly different outputs on a fresh run, and room building on video depth is sensitive to them. c00a170fe1's video plan was 3 rooms and 19.82 m² from the cache, and 2 rooms and 20.32 m² on a fresh run. Cached runs replay exactly.
- **Standing still and sweeping:** rooms are grown from where the phone walked. On two public ARKitScenes scans, where the phone stayed within about 1 m, the layout found a single 3 m² room in rooms of about 20 m². The protocol's walk along the walls avoids this. It is also why ARKitScenes can test where the sensor puts walls (section 3) but not our layout's wall lengths.
- **Openings:** detection and widths do not repeat between two walks of the same flat: 23 vs 12 openings. Of 7 doors found in both, only one was measured jamb to jamb both times, and those widths differ by 13.0 cm. Doors walked through without both jambs in view get a typical width with a ±0.25 m interval. G-OPEN would fail; the fix is to see both jambs (the protocol's walk-through step) and to refine edges on the images.
- **Non-right-angled rooms:** the walls-first layout needs right angles; otherwise it falls back to floor-traced outlines, which stop at furniture.

## 9. What comes next

In order of expected score gain:
1. A tape-measured benchmark of two of our rooms, captured with Stray Scanner, Polycam (head-to-head), iPhone video and photos, plus a staged-damage room.
2. Opening widths from image edges.
3. Gyroscope-aided break detection for video.
4. A learned room-layout model for photos to replace the box.
