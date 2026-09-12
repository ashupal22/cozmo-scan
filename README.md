# cozmo-scan

Turn iPhone photos, a video, or a LiDAR scan of a home into a measured floor plan with damage regions, a repair list, and an honest ± range on every number.

Built for the Cozmo AI Applied AI case study.

## Status

Early work. The repo currently holds exploration scripts that check the LiDAR captures: floor and ceiling planes, heading drift, and how well two walks of the same apartment agree. The pipeline itself is not built yet.

## Output format and gates

- [`schema/output.schema.json`](schema/output.schema.json): the JSON every run must produce. It's a stand-in until Cozmo shares their published schema. See [`schema/example_output.json`](schema/example_output.json).
- [`docs/gates.md`](docs/gates.md): every pass/fail target, each marked as from the brief or as our assumption.

## Data

Raw captures are **not** stored in git, because they are large and show a private home. Download them from Google Drive:

https://drive.google.com/drive/folders/1rvcx0uEIwU6mIlEi8m5SF88jHK6ubAOu

See [`data/README.md`](data/README.md) for where to put them.

## Run the exploration scripts

```bash
python -m pip install -r explore/requirements.txt
export COZMO_DATA=data/captures        # folder that contains the capture folders

python explore/probe_basic.py   c00a170fe1              # floor plane, pose jumps, top-down map
python explore/probe_normals.py c00a170fe1 c7d28f72c6   # normal-aware floor/ceiling, heading drift
python explore/probe_two_walks.py c7d28f72c6 1a8384c3f6 # align two walks of the same apartment
```

Outputs (images, cached points) go to `explore/out/`, which git ignores.
