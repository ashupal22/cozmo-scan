# Data we use

We have three Stray Scanner recordings of one apartment, and nothing else: no tape or laser measurements, no photo-tier or video-tier recordings, no damaged room. Cozmo provides no data. The brief allows *"any pretrained model, dataset or API with disclosure"*, so we add public datasets for two jobs:

1. **Measure accuracy against real ground truth**, which our own recordings cannot give.
2. **Test parts of the pipeline** our recordings do not exercise: multi-room joining, door logic, damage classes.

Public data **does not replace** the benchmark the brief asks us to build ourselves. See [What public data cannot cover](#what-public-data-cannot-cover).

## Datasets

| Dataset | What we use it for | Brief parts it supports | Ground truth | License / access | Download | Status |
|---|---|---|---|---|---|---|
| **Our 3 Stray Scanner recordings** | Development on the real capture route. Repeatability: two walks of the same apartment compared against each other. Drift ablation. Runtime | G-REPEAT (walk vs walk, no GT), G-DRIFT, A-RUNTIME, A-DETERMINISM | None, consistency checks only | Ours; Google Drive link in `data/README.md` | ~420 MB video + depth | Have |
| **[ARKitScenes](https://github.com/apple/ARKitScenes)** raw, Validation venues 381644 and 384651, 3 walks each | LiDAR-tier accuracy. Same depth format as Stray Scanner (256×192 mm depth, confidence 0–2, per-frame intrinsics and poses), plus depth rendered from Faro laser scans and registered to each frame (`highres_depth`, 1920×1440). Three walks per venue give repeatability **with** ground truth. RGB frames of the same walks feed video/photo-tier development | G-CEIL, G-CEIL-SPREAD, A-WALL-LIDAR, G-REPEAT, G-DRIFT; development input for G-WALL-VIDEO / G-WALL-PHOTO | Laser-scanner depth per frame (10 fps, filtered) | Apple ARKitScenes license: evaluation and publishing numbers allowed, no implied Apple endorsement. Direct download | 3.0 GB (6 walks) | Downloaded with `scripts/fetch_external.py arkitscenes` |
| **[HouseLayout3D](https://huggingface.co/datasets/houselayout3d/HouseLayout3D)** | Ground-truth walls, floors, ceilings, doors (4 corners, opening direction), windows and stairs for 16 real buildings, 317 rooms. We cut buildings into rooms, add realistic noise, and check that the stitch solver rejoins them through the right doors | G-PHOTO-STITCH (solver), A-ADJ, G-OPEN (width logic) | Manual annotation | MIT | 32 MB | Downloaded |
| **[BD3](https://github.com/Praveenkottari/BD3-Dataset)** building defects | Class check for damage detection: 3,965 phone photos of algae, major/minor crack, peeling, spalling, stain, normal | A-DMG-DETECT (class only) | Image-level labels | Original: no license stated. Used via its CC-BY-4.0 re-release `chandrabhuma/building_defect_vqa` (test split, 793 images): local evaluation only, never redistributed or used for training | 157 MB (test split) | **Used**: `bench/damage_bd3.py` |
| **GDD** (glass) / **MSD** (mirror) | Check that mirror and glass masks work before trusting depth near them | Brief constraint: mirrors and glass | Pixel masks | Research use, by request | Small | Optional |
| **[ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/)** + **[MultiViewRoomLayout](https://github.com/ghanning/MultiViewRoomLayout)** | The closest public match: iPhone 13 Pro LiDAR (256×192) plus laser scans, and multi-room layout annotations (80 non-cuboid scenes) | A-WALL-LIDAR, G-PHOTO-STITCH, openings | Laser + manual layouts | Application required, non-commercial; annotations MIT | Large | Optional: apply if time allows |

### Not used, and why

| Dataset | Reason |
|---|---|
| Aria Synthetic Environments | Synthetic, fisheye Aria sensor, 23 TB total |
| Zillow Indoor Dataset (ZInD) | 360° panoramas, not iPhone stills; registration required |
| Structured3D | Synthetic renders; agreement form |
| ARKitScenes raw Faro point clouds | ~1.9 GB each, and not registered to the ARKit poses (the transform is not published). `highres_depth` gives the same laser truth already aligned |

## Caveats we report

- **ARKitScenes was captured on a 2020 iPad Pro,** not an iPhone 15. The LiDAR format matches, but noise levels may differ.
- **ARKitScenes `highres_depth` frames were filtered** to those that agree with the device depth, which may make LiDAR look slightly better than it is. We use it for planes averaged over many frames, where single-frame outliers matter less.
- **HouseLayout3D has no images** (Matterport3D frames need a separate agreement), so it tests layout logic, not perception. The Hugging Face snapshot we downloaded (2026-09-13) has `structures`, `doors`, `stairs` and `poses`, but no `windows` folder, although its README lists one.

## What public data cannot cover

The brief says *"you build the benchmark set yourself"*. These rows need a device and a measuring tool, and stay marked **not done** in the compliance matrix until they exist:

| Brief requirement | Needs |
|---|---|
| Same rooms captured at all three tiers, with iPhone 15 or newer | An iPhone (Pro for LiDAR) |
| A furnished room with staged damage spanning two classes | Removable panels and a phone |
| Laser or tape ground truth on everything | A tape measure is enough to start |
| Head-to-head against a consumer app on 2 rooms | The app on a phone |
| One room captured twice at the same tier | We have this for the apartment (LiDAR), without ground truth |

## Machine limits

The development machine has about 10 GB of free disk and downloads at about 1.9 MB/s. That rules out large raw datasets, and model weights chosen in later steps have to fit too.
