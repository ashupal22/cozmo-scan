"""Build the output JSON (schema/output.schema.json) from LiDAR geometry, with provisional intervals.

Provisional error model, to be replaced by conformal calibration on benchmark residuals (step 8):
- Device depth reads surfaces about 13 mm closer than the laser ground truth on ARKitScenes
  (bench/results/arkitscenes_planes.json). Until that bias is calibrated it is carried as an
  uncertainty of DEPTH_BIAS_M on every measured surface, not silently corrected.
- A wall face's uncertainty combines the spread of its wall points (reduced by the number of
  roughly independent samples) with the depth bias. Unsnapped faces, outlined from floor cells
  only, get UNSNAPPED_FACE_M.
- A wall's length depends on the two neighbouring faces; room area shifts with every face at once.
- Ceiling height combines the floor and ceiling plane standard errors with the bias on both.
- Video tier (video_errors): wall positions come from a learned depth model, so the per-face term is
  larger, and the overall scale comes from a monocular metric-depth model. A scale error stretches
  every length in the capture by the same factor, so it is added per measurement in proportion to
  its size (twice that for areas). The model above is then widened by VIDEO_INTERVAL_SCALE, set from
  the errors the video tier actually makes (bench/calibrate_intervals.py).
All intervals are nominal 90% (value +/- 1.645 sigma).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cozmo.geometry.planes import HorizontalPlane
from cozmo.geometry.rooms import RoomMap
from cozmo.geometry.walls import OpeningOnWall, RoomOutline

Z90 = 1.645
DEPTH_BIAS_M = 0.013
UNSNAPPED_FACE_M = 0.05
POINTS_PER_SAMPLE = 100
DOORWAY_WIDTH_SIGMA_M = 0.05
NOMINAL_DOOR_WIDTH_SIGMA_M = 0.15  # jambs not seen: a typical door width with a wide range
CEILING_PRIOR_M = (2.5, 2.2, 3.2)  # value, low, high when no ceiling was seen anywhere
VIDEO_FACE_M = 0.03
VIDEO_OPENING_M = 0.08
# Multiplier on every video interval, from bench/calibrate_intervals.py on bench/results/video_vs_lidar.json
# (commit 1d00698): 9 paired walls on two flats, error / own sigma 1.5-7.4, factor 4.49. Only 1 of 9 intervals
# held the LiDAR length before. Thin evidence (no leave-one-walk-out check was possible): re-measure after
# any change to the video tier.
VIDEO_INTERVAL_SCALE = 4.5


@dataclass(frozen=True)
class ErrorModel:
    """Error terms of one tier. face_m: systematic error of one wall face or plane. scale_sigma: relative
    error of the overall scale, shared by every length in the capture (0 when depth is measured)."""
    face_m: float = DEPTH_BIAS_M
    opening_m: float = DOORWAY_WIDTH_SIGMA_M
    scale_sigma: float = 0.0
    interval_scale: float = 1.0      # widens every interval of the tier, from calibration on benchmark errors


LIDAR_ERRORS = ErrorModel()


def video_errors(scale_sigma: float) -> ErrorModel:
    return ErrorModel(VIDEO_FACE_M, VIDEO_OPENING_M, scale_sigma, VIDEO_INTERVAL_SCALE)


def measurement(value: float, sigma: float, digits: int = 3, observed: bool = True, method: str | None = None) -> dict:
    m = {"value": round(value, digits), "ci_low": round(value - Z90 * sigma, digits),
         "ci_high": round(value + Z90 * sigma, digits), "confidence": 0.9}
    if not observed:
        m["observed"] = False
    if method:
        m["method"] = method
    return m


def face_sigma(outline: RoomOutline, k: int, errors: ErrorModel = LIDAR_ERRORS) -> float:
    wall = outline.walls[k]
    if wall.support == 0:
        return max(UNSNAPPED_FACE_M, errors.face_m)
    samples = max(wall.support / POINTS_PER_SAMPLE, 1.0)
    return float(np.hypot(wall.face_spread_m / np.sqrt(samples), errors.face_m))


def build_document(capture_info: dict, floor: HorizontalPlane, room_map: RoomMap,
                   outlines: dict[int, RoomOutline], ceilings: dict[int, HorizontalPlane | None],
                   openings: list[OpeningOnWall], runtime_s: float,
                   drift: dict | None = None, errors: ErrorModel = LIDAR_ERRORS) -> tuple[dict, list[str]]:
    """`drift` is DriftReport.to_schema(), or None when drift correction was switched off."""
    scale, widen = errors.scale_sigma, errors.interval_scale
    warnings = ["opening widths are coarse (5 cm plan grid); image-edge refinement not built yet",
                "damage detection, concealed-damage rules and scope are not built yet"]
    if drift is None:
        warnings.insert(0, "drift correction switched off: phone poses used as recorded")
    seen = [c.height - floor.height for c in ceilings.values() if c is not None]

    rooms, footprint, footprint_sigma = [], 0.0, 0.0
    opening_ids: dict[int, str] = {}  # index in `openings` -> id
    for rid, outline in sorted(outlines.items()):
        name = f"R{rid}"
        n = len(outline.walls)
        sig = [face_sigma(outline, k, errors) for k in range(n)]
        walls = []
        for k, wall in enumerate(outline.walls):
            length_sigma = widen * float(np.linalg.norm([sig[(k - 1) % n], sig[(k + 1) % n], scale * wall.length_m]))
            walls.append({"id": f"{name}.W{k + 1}",
                          "start": [round(float(v), 3) for v in wall.start],
                          "end": [round(float(v), 3) for v in wall.end],
                          "length_m": measurement(wall.length_m, length_sigma,
                                                  method="corner to corner, inside faces")})
        area_sigma = widen * float(np.hypot(outline.perimeter_m * np.mean(sig), 2 * scale * outline.area_m2))
        footprint += outline.area_m2
        footprint_sigma += area_sigma

        ceiling = ceilings.get(rid)
        if ceiling is not None:
            height = ceiling.height - floor.height
            height_sigma = widen * float(np.sqrt(floor.standard_error ** 2 + ceiling.standard_error ** 2
                                                 + 2 * errors.face_m ** 2 + (scale * height) ** 2))
            ceiling_m = measurement(height, height_sigma, method="floor and ceiling plane fit")
        else:
            if seen:
                prior = float(np.median(seen))
                ceiling_m = {"value": round(prior, 3), "ci_low": round(min(seen) - 0.1, 3),
                             "ci_high": round(max(seen) + 0.1, 3), "confidence": 0.9, "observed": False,
                             "method": "not observed; range of ceilings seen in other rooms"}
            else:
                v, lo, hi = CEILING_PRIOR_M
                ceiling_m = {"value": v, "ci_low": lo, "ci_high": hi, "confidence": 0.9, "observed": False,
                             "method": "not observed anywhere; typical residential range"}
            warnings.append(f"{name}: ceiling not observed; height is a prior with a wide range")

        room_openings = []
        for index, op in enumerate(openings):
            if op.room_id != rid:
                continue
            oid = f"{name}.O{len(room_openings) + 1}"
            opening_ids[index] = oid
            room_openings.append({"id": oid, "type": "opening", "wall_id": f"{name}.W{op.wall_index + 1}",
                                  "width_m": measurement(op.width_m, widen * float(np.hypot(errors.opening_m, scale * op.width_m)),
                                                         method="gap between the jambs")
                                  if op.measured else
                                  measurement(op.width_m, NOMINAL_DOOR_WIDTH_SIGMA_M, observed=False,
                                              method="walked through, jambs not seen: typical door width"),
                                  "offset_along_wall_m": measurement(op.offset_m, widen * float(np.hypot(errors.opening_m, scale * op.offset_m))),
                                  "connects_to": f"R{op.other_room}" if op.other_room is not None else None})

        rooms.append({"id": name, "label": f"room {rid}",
                      "polygon": [[round(float(x), 3), round(float(z), 3)] for x, z in outline.vertices],
                      "floor_area_m2": measurement(outline.area_m2, area_sigma, digits=2),
                      "perimeter_m": measurement(outline.perimeter_m, widen * float(np.hypot(n * np.mean(sig), scale * outline.perimeter_m)),
                                                 digits=2),
                      "ceiling_height_m": ceiling_m, "walls": walls, "openings": room_openings})

    adjacency = {}
    for index, op in enumerate(openings):
        if index not in opening_ids:
            continue
        if op.other_room is None:
            continue  # a door to space that was not scanned
        key = tuple(sorted((op.room_id, op.other_room)))
        adjacency.setdefault(key, []).append(opening_ids[index])
    plan = {
        "footprint_area_m2": measurement(footprint, footprint_sigma, digits=2),
        "adjacency": [{"room_a": f"R{a}", "room_b": f"R{b}", "via": via} for (a, b), via in sorted(adjacency.items())],
        "stitch": {"method": "single_map" if len(rooms) > 1 else "single_room", "low_confidence": False},
        "drift": drift or {"enabled": False, "correction": [], "notes": "switched off"},
    }
    document = {
        "schema_version": "0.1.0",
        "capture": {**capture_info, "runtime_s": round(runtime_s, 1)},
        "units": {"length": "m", "area": "m2"},
        "rooms": rooms,
        "plan": plan,
        "damage": [],
        "concealed_flags": [],
        "scope": [],
        "quality": {"low_confidence": any(not c.get("observed", True) for c in
                                          (r["ceiling_height_m"] for r in rooms)), "warnings": warnings},
    }
    return document, warnings
