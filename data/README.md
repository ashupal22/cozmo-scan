# Data

Capture data is kept out of git. Download it from Google Drive:

https://drive.google.com/drive/folders/1rvcx0uEIwU6mIlEi8m5SF88jHK6ubAOu

Put each capture folder under `data/captures/`, or set `COZMO_DATA` to wherever the folders already are.

```
data/captures/
  c7d28f72c6/
    rgb.mp4             HEVC video, 1920×1440, 60 fps
    depth/000000.png    16-bit LiDAR depth in millimetres, 256×192
    confidence/         depth confidence, 0 = low, 1 = medium, 2 = high
    odometry.csv        per-frame ARKit pose and intrinsics
    imu.csv             accelerometer and gyroscope
    camera_matrix.csv   camera intrinsics
```

## Captures

All three were recorded with the Stray Scanner app on an iPhone Pro.

| ID | Length | Walked | What it covers |
|---|---|---|---|
| `c7d28f72c6` | 215 s | ~100 m | Whole apartment: T-shaped hallway and several rooms. Ceiling is visible in part of it |
| `1a8384c3f6` | 115 s | ~54 m | Same apartment, a second walk. Ceiling never in view; two pose jumps near the end |
| `c00a170fe1` | 37 s | ~14 m | Part of the apartment. Ceiling never in view |

There is no tape or laser ground truth yet.
