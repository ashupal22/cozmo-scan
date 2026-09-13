# Capture protocol

One page: follow it exactly. Each *why* is a failure we measured (`bench/README.md`). What each tier delivers on which
phone: [`device_matrix.md`](device_matrix.md).

## All tiers

- Switch on every light, open every internal door, clear the doorways, and keep people and pets out of the way.
- Hold the phone at chest height and walk at half your normal pace.
- **Never turn while facing a plain wall closer than 1 m**: step back, then turn. *Why: tracking is lost there.*
- **End the walk where you started.** *Why: returning lets drift be corrected.*
- **Mirrors and glass:** do not film them straight on, and close the curtains at night. *Why: depth sees a room behind the glass.*
- **Damage:** stop for 2 seconds about 1 m from any stain, crack or mould (photo tier: make it one of that room's photos). *Why: close-up defects are found 62% of the time; damage seen from across the room was missed.*

## Tier 1: LiDAR (iPhone 12 Pro or newer, Pro or Pro Max)

1. Install **Stray Scanner** (free, App Store). Open it and tap record at the front door.
2. In **every room**: walk along the walls about 1 m from them; tilt the phone up until you see where walls meet the ceiling, all the way round; walk through each doorway facing forward. *Why: rooms grow from where you walk (standing in one spot gave a 3 m² room inside a 20 m² one); rooms are split on walls seen above 1.5 m; seeing both door frames gives the width.*
3. Return to the front door and stop. Allow about **1 minute per room**.
4. In Stray Scanner, share the recording to the Mac (AirDrop or Files), then run `cozmo run <folder>`.

## Tier 2: Video (any iPhone 15 or newer)

1. **Settings → Camera → Record Video: 1080p HD at 30 fps, HDR Video off.** In the Camera app choose Video and the 1× lens. No zoom, no Cinematic or Action mode. *Why: tested on standard 8-bit video.*
2. Walk the LiDAR route: the ceiling sweep in every room, through every doorway, back to the start. About **30 seconds per room, under 2 minutes** in all. *Why: a longer clip gets key frames spaced further apart to keep the runtime bounded, so its camera path is less certain.*
3. **Turn slowly**: about 3 seconds for a quarter turn. *Why: fast turns in front of plain walls broke the camera path.*
4. AirDrop the `.mov` file to the Mac, then run `cozmo run <file>.mov`.

## Tier 3: Photos (any iPhone 15 or newer)

Go room by room, with **2 to 8 photos per room**.

1. Stand in the room's doorway. Hold the phone level at chest height, 1× lens.
2. Take **5 or 6 photos while turning from left to right**, each overlapping the last by about a third, each showing some floor and ceiling. *Why: unrelated views broke the camera estimates (44–171° off); overlapping sweeps kept errors to a few degrees.*
3. Take one photo of each **other** door of the room, from inside, straight on, with the door open. *Why: rooms are joined into one plan through their doors.*
4. AirDrop the photos to the Mac. Make a folder, for example `home`, with one folder per room (`kitchen`, `bedroom 1`). Drag each room's photos into its folder, then run `cozmo run home`.

## Output

Each run writes `out/<name>/result.json`, `plan.svg` and `summary.md` (each room's width × length, ceiling height and door widths, with 90% ranges). The warnings say what was not seen and why a range is wide.
