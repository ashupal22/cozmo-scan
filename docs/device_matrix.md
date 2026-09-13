# Device matrix

Which tier runs on which hardware, and the accuracy each tier honestly delivers today. The capture steps are in
[`capture_protocol.md`](capture_protocol.md). The phone only captures; processing runs on the computer (tested on an
Apple M4, 16 GB, macOS).

| Tier | Devices | Measured so far | Gate status |
|---|---|---|---|
| LiDAR | iPhone 12 Pro+ (Pro models), iPad Pro 2020+ | Ceiling height within 15 mm on 6/6 walks, wall-to-wall distances within 1.36 cm on 4/4 pairs (ARKitScenes laser truth, iPad Pro 2020, after depth-bias correction). Same flat walked twice: footprints agree within 1.6%. Drift correction ablated on 3 walks | G-CEIL met on public data. Walls and openings not yet checked against a tape measure |
| Video | Any iPhone 15+ | Walls typically 8–16% off against LiDAR of the same walk; intervals widened 4.5× to stay honest | G-WALL-VIDEO (±3%) not met |
| Photo | Any iPhone 15+ | Room boxes about 26% off (median), too small more often than too big; intervals widened 5.1× hold on 17 of 18 dimensions. Stitching gets every door right in 40 of 46 noise-free test runs, and in 14 of 46 with photo-like noise | G-WALL-PHOTO and G-PHOTO-STITCH not met |

*iPhone results are from our own three walks. The ceiling numbers come from a 2020 iPad Pro; their transfer to iPhone 15 Pro is unverified until a tape-measured room is captured.*
