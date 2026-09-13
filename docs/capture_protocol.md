# Capture protocol (one page)

Follow these steps exactly. Every "why" comes from a measured failure, listed in `bench/README.md`.

## Before you start (all tiers)

- Switch on every light and open every internal door fully. Ask people and pets to stay out of the way.
- Clear the floor of anything that blocks a doorway.
- Hold the phone at chest height. Walk slowly, about half your normal pace.
- **Never turn while pointing at a plain wall closer than 1 m.** Step back first, then turn. *Why: this is where tracking is lost.*
- **End the walk where you started.** *Why: returning lets drift be corrected.*
- **Mirrors and glass:** do not film large mirrors or glass doors straight on. At night, close the curtains. *Why: depth sees a room behind the glass that is not there.*

## Tier 1: LiDAR (iPhone 12 Pro or newer Pro / Pro Max)

1. Install **Stray Scanner** (free, App Store). Open it and tap record.
2. Start at the front door, facing into the home.
3. In **every room**:
   1. Walk along the walls about 1 m away from them. *Why: rooms are grown from where you walked. Standing in one spot and sweeping (as public scans do) gave a 3 m² room inside a room of about 20 m².*
   2. Tilt the phone up until you see where the walls meet the ceiling, all the way round. *Why: rooms are split using walls seen above 1.5 m. A walk that pointed the phone down merged a corridor into a living room.*
   3. Walk through each doorway facing forward, so both sides of the door frame are seen. *Why: opening widths.*
4. Return to the front door and stop. Allow about **1 minute per room**.
5. Hand-off: in Stray Scanner, share the recording and save the folder to a Mac (AirDrop or Files).
6. Run `cozmo run <folder>`.

## Tier 2: Video (any iPhone 15 or newer)

1. In **Settings → Camera → Record Video**, choose **1080p HD at 30 fps** and turn **HDR Video off**. *Why: our tests used standard 8-bit video.*
2. Open the **Camera** app, choose Video, use the 1× lens. Do not zoom; do not use Cinematic or Action mode.
3. Walk the same route as for LiDAR: the ceiling sweep in every room, through every doorway, back to the start. Allow about **30 seconds per room**, and keep the clip **under 2 minutes**. *Why: a run costs about 3.4 s per key frame, so the pipeline caps a clip at 120 key frames to stay inside the 10 minute budget. A longer walk is not refused, but its key frames are spaced further apart and the camera path gets less certain; the output says when this happened.*
4. **Turn slowly**: take about 3 seconds for a quarter turn. *Why: faster turns in front of plain walls broke the camera path in our tests.*
5. Hand-off: AirDrop the `.mov` file to the Mac, then run `cozmo run <file>.mov`.

## Tier 3: Photos (any iPhone 15 or newer)

Go room by room, finishing one room before starting the next. **At most 8 photos per room.**
1. Stand in the room's doorway. Hold the phone level at chest height, 1× lens.
2. Take **5 or 6 photos while turning from left to right**, each overlapping the previous one by about a third. Every photo should show some floor and some ceiling. *Why: unrelated views made the camera estimates fail (angle errors of 44–171°); overlapping sweeps kept errors to a few degrees.*
3. Then take one photo of each **other** door of the room, from inside, straight on, with the door open. *Why: rooms are joined into one plan through their doors.*
4. Hand-off: AirDrop all photos to the Mac. Make one folder, for example `home`. Inside it make one folder per room, named after the room (`kitchen`, `bedroom 1`), and drag each room's photos into its folder. They arrive in shooting order.
5. Run `cozmo run home`.

## Output

Every command writes `out/<name>/result.json` and `out/<name>/plan.svg`. Warnings in the JSON say what was not seen, for example a ceiling or a doorway, and how much wider the intervals are because of it.

## Device matrix: what each tier honestly delivers today

| Tier | Devices | Measured so far | Gate status |
|---|---|---|---|
| LiDAR | iPhone 12 Pro+ (Pro models), iPad Pro 2020+ | Ceiling height within 15 mm on 6/6 walks (ARKitScenes laser truth, iPad Pro 2020, after depth-bias correction). Same flat walked twice: footprints agree within 1.6%. Drift correction ablated on 3 walks | G-CEIL met on public data. Walls and openings not yet checked against a tape measure |
| Video | Any iPhone 15+ | Walls typically 8–16% off against LiDAR of the same walk; intervals widened 4.5× to stay honest | G-WALL-VIDEO (±3%) not met |
| Photo | Any iPhone 15+ | Room boxes about 26% off (median), too small more often than too big; intervals widened 5.1× hold on 17 of 18 dimensions. Stitching gets every door right in 40 of 46 noise-free test runs, and in 14 of 46 with photo-like noise | G-WALL-PHOTO and G-PHOTO-STITCH not met |

*iPhone results are from our own three walks. The ceiling numbers come from a 2020 iPad Pro; their transfer to iPhone 15 Pro is unverified until a tape-measured room is captured.*
