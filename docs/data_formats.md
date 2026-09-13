# Data formats

What `cozmo run` accepts at each tier, what the test and benchmark data look like, what a run writes, and why each
format was chosen. The tier is detected from the input (`cozmo/ingest/detect.py`), so one capture is one tier and one
command.

## Inputs

### LiDAR tier: a Stray Scanner export folder, unchanged

```
c00a170fe1/
  rgb.mp4                colour video: HEVC, 1920×1440, 60 fps
  depth/000000.png       one per frame: 16-bit PNG, depth in millimetres, 256×192
  confidence/000000.png  one per frame: 8-bit PNG, 0 = low, 1 = medium, 2 = high
  odometry.csv           per frame: timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy
  camera_matrix.csv      3×3 camera matrix
  imu.csv                timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z (recorded, not needed)
```

- `rgb.mp4`, `odometry.csv`, `camera_matrix.csv`, `depth/` and `confidence/` are required. The error names any that are missing.
- `odometry.csv` must contain those 13 columns; extra columns (newer Stray versions add distortion centres) are ignored.
- The pose maps camera coordinates (x right, y down, z forward) to a gravity-aligned world with y up. Camera intrinsics are taken per frame, because fx varies by up to about 2% within one capture.

### Video tier: one video file

- `.mov`, `.mp4` or `.m4v`: the iPhone Camera app's own file, HEVC or H.264, 8 or 10 bit. The phone's rotation flag is applied.
- Pass the file itself, or a folder that holds only that video.
- At least about 4 seconds (12 key frames). Key frames are taken at 3 per second, up to 360 (2 minutes); a longer clip is thinned and the output says so.

### Photo tier: one folder of room folders

```
home/
  kitchen/     IMG_0001.HEIC  IMG_0002.HEIC  ...   2 to 8 photos
  bedroom 1/   IMG_0010.HEIC  ...
```

- `.heic`, `.heif`, `.jpg`, `.jpeg` or `.png`, in any letter case. Other files, such as Live Photo `.MOV` clips, are ignored.
- Each folder name becomes the room's name. A folder holding only photos is one room. Mixing loose photos with room folders stops with an error. Fewer than 2 or more than 8 photos gives a warning.
- EXIF orientation turns each photo upright. The 35 mm-equivalent focal length (which iPhones write) sets the metric scale. Without it the focal length is estimated from the room's lines, which is less accurate.

## Test and benchmark data

| Data | Format | Where |
|---|---|---|
| Our three raw captures | Stray Scanner exports, as above | Google Drive (`data/README.md`); checksums in `data/captures_manifest.json` |
| Video test inputs | A walk's `rgb.mp4` turned upright, timestamps kept: `data/derived/<walk>_upright.mp4` | Made by `bench/video_vs_lidar.py` |
| Photo test sets | `data/derived/<walk>_photo_sweeps/room_N/*.jpg`, with `manifest.json` naming the LiDAR room and frames behind each folder | Made by `bench/make_photo_sets.py` |
| LiDAR laser truth | ARKitScenes raw walks: Stray-like depth (16-bit mm, 256×192, confidence 0–2), poses in `lowres_wide.traj`, and laser-rendered depth in `highres_depth/` (16-bit mm, 1920×1440) | `scripts/fetch_external.py arkitscenes` |
| Multi-room layouts | HouseLayout3D JSON annotations: rooms, walls, doors | `scripts/fetch_external.py houselayout3d` |
| Real defect photos | BD3 test split: a parquet table of photo bytes and a defect label | Downloaded by `bench/damage_bd3.py` |
| Real iPhone photos | Two iPhone 12 Pro HEIC samples from heic.digital | Downloaded by `bench/iphone_photo_check.py` |
| Tape and app measurements | CSV `room, item, tape_m, app_m, notes`, in metres. Items: length, width, ceiling height, door width, and damage width, height and height above floor | Template `bench/templates/measurements.csv`, read by `bench/tape_truth.py` |
| Unit tests | Synthetic rooms, walks and clips generated inside the tests | `tests/` (121 tests) |

To add new test captures, keep one room name across all three tiers and the CSV, so the comparison can match rooms:
`data/captures/<id>/` (Stray export), `<id>.mov` (video), `<id>_photos/<room>/` (photos).

## Output

`out/<name>/`:
- `result.json`: validated against `schema/output.schema.json` (version 0.1.0, ours).
  - Every measurement is `{value, ci_low, ci_high, confidence: 0.9}`, plus `observed: false` when it was never seen.
  - Units are metres and square metres.
  - IDs: rooms `R1`, walls `R1.W1`, openings `R1.O1`, surfaces `R1.FLOOR` / `R1.CEIL`, damage `D1`, flags `F1`, scope items `S1`, rules `R-...`.
- `plan.svg`: the dimensioned floor plan.
- `summary.md`: each room's width × length, ceiling height and opening widths, with ranges.
- `report.png` and `report.pdf`: one A4 landscape page with the plan drawing (rooms, wall lengths, openings, damage markers), the room table with ranges, the plan summary, damage and repairs, and the first warnings. `cozmo report <result.json>` redraws it for any result.

## Why these formats

| Choice | Why |
|---|---|
| Stock apps (the brief's Route 2) | Nothing to build or install beyond free App Store apps, and a non-engineer can follow the steps |
| Stray Scanner for LiDAR | Free. It records what the brief lists for the tier (depth, poses, intrinsics), plus confidence, IMU and video, as open PNG, CSV and MP4 files that need no vendor software. Its depth format is the one Apple's ARKitScenes uses, so the same code runs on public laser truth |
| Depth as 16-bit PNG in millimetres | Lossless (a compressed image would corrupt depth), 1 mm steps, and a standard format |
| Confidence map | Lets fusion drop unreliable returns at edges and on dark or shiny surfaces |
| CSV poses, intrinsics per frame | Readable and easy to check; the intrinsics change during a capture |
| The Camera app's `.mov` for video | The examiners film with their own iPhone and the stock app, so there is no conversion step |
| 1080p at 30 fps, HDR off, 1× lens, no Cinematic or Action mode | 1080p is enough (the model works at 504 px), and 30 fps is enough (3 key frames per second are used). The tests used standard 8-bit video. Zoom changes the focal length, which breaks the scale. Cinematic and Action modes crop and stabilise the image, which bends geometry |
| Under 2 minutes of video | Every key frame is used up to 2 minutes, which takes about 13 minutes to run. With key frames 1 s apart, a 115 s test walk collapsed to one room |
| HEIC photos (JPEG and PNG also accepted) | HEIC is the iPhone default. Its focal length sets the scale, read within 1% on real iPhone 12 Pro photos |
| One folder per room, 2 to 8 photos | Exactly the brief's photo tier |
| 5 or 6 overlapping photos from the doorway, one per other door | Unrelated views gave camera errors of 44–171°, while overlapping sweeps stayed within a few degrees. Rooms are joined through their doors |
| Test inputs cut from the LiDAR walks | No separate iPhone 15 captures were available. Cutting them from the same walk gives a known reference, the LiDAR plan of the same moment. Disclosed as stand-ins |
| ARKitScenes, HouseLayout3D, BD3, heic.digital samples | The public data closest to each need: LiDAR with registered laser truth, real multi-room layouts with doors, labelled defect photos, and real iPhone HEIC files |
| CSV for tape measurements | The simplest format for someone holding a tape; it feeds the head-to-head script directly |
| JSON with a schema, SVG, Markdown summary | JSON can be checked by machine, and every value carries its range as the brief asks. SVG opens in any browser. The summary is quick to read against a laser measurer |
| Raw captures kept out of git | They show a private home and are large. Checksums let anyone verify copies; public data is fetched by script, as the brief asks for large files |
