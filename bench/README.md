# Benchmarks

Every number here can be regenerated from raw data with the command shown. Results are stored in `bench/results/`, and git history keeps the before/after versions.

## Floor and ceiling vs laser ground truth (ARKitScenes)

```bash
python scripts/fetch_external.py arkitscenes     # once, ~3 GB
python bench/arkitscenes_planes.py               # ~10 min on an Apple M4
```

This compares device LiDAR against depth rendered from Faro laser scans on 6 walks in 2 venues (3 walks each). Targets from `docs/gates.md`: **G-CEIL**, ceiling error ≤ 15 mm; **G-CEIL-SPREAD**, spread across walks of the same room ≤ 10 mm.

### Before and after choosing planes by area (commit `1749867`)

Negative ceiling error means the ceiling reads too low; positive floor error means the floor reads too high.

| Walk | Venue | Ceiling error, before | Ceiling error, after | Floor error, before | Floor error, after |
|---|---|---|---|---|---|
| 41069048 | 381644 | −28.1 mm | −28.1 mm | +16.1 mm | +16.1 mm |
| 41069050 | 381644 | −26.9 mm | −26.9 mm | +14.3 mm | +14.3 mm |
| 41069051 | 381644 | −23.6 mm | −23.6 mm | +10.7 mm | +10.7 mm |
| 41142278 | 384651 | **−221.2 mm** | −20.9 mm | **+207.7 mm** | +7.4 mm |
| 41142280 | 384651 | −11.2 mm | −11.2 mm | +12.1 mm | +12.1 mm |
| 41142281 | 384651 | −19.8 mm | −19.8 mm | +11.0 mm | +11.0 mm |
| **Within 15 mm** | | **1/6** | **1/6** | | |
| **Mean** | | −55.1 mm | −21.8 mm | | |

### What the numbers say

- **The fix worked for the problem it targeted.** On walk 41142278 a small raised surface seen up close had more points than the floor. Choosing by covered area picks the real floor, so the pipeline's choice now matches the laser on all 6 walks.
- **The remaining error is a consistent bias, not noise.** Every walk reads the ceiling 11–28 mm low and the floor 7–16 mm high. Per pixel, device depth is 9–16 mm shorter than laser depth between 0.3 and 2 m: the sensor sees surfaces slightly closer than they are. In the brief's words this is **repeatable but biased**, and it fails G-CEIL.
- **Spread:** venue 381644 spreads 9.2 mm across its walks (passes G-CEIL-SPREAD). Venue 384651 spreads 42.6 mm, but its own laser ceiling heights also differ by 34 mm between walks, because different walks see different parts of a ceiling with more than one level. A fair spread test must compare the same ceiling plane in the same room. That's the next revision of this benchmark.

### Caveats

- ARKitScenes was captured on a 2020 iPad Pro, not an iPhone 15 or newer.
- Laser-rendered frames were pre-filtered to agree with the device depth, which may flatter the device slightly.

### Next candidate fix

Calibrate the depth bias (as a constant offset or proportional to range, whichever fits the residuals) on held-out walks, then re-run. This is a candidate for the brief's fix loop, pending the full benchmark on all gates.
