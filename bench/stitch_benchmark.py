"""Stitch solver benchmark: rejoin real buildings from separately framed rooms (G-PHOTO-STITCH, A-ADJ).

Test cases come from HouseLayout3D (bench/houselayout_properties.py): rooms that doors join, with the true
arrangement. Each room is handed to the solver the way the photo tier reports it, in its own frame: a random
quarter turn plus a small turn, a random shift, a scale error, a jittered outline, and the doors seen on its
walls (centre on the inside face, outward normal, width). Door widths and positions are noisy; some doors are
missed, and some walls get a door that is not there. Doors to rooms outside the case stay in as distractors.

Scored per case:
  adjacency  pairs of rooms the solver joined through a door, against the true pairs: precision, recall, and
             whether the sets are identical (the brief's "correct adjacency")
  doors      door pairs the solver made, against the true doors: a pair is right only when both sides are the
             same real door. Rooms can be joined through the wrong doors and still count as adjacent, which
             misplaces them, so this is the stricter score
  islands    1 means one connected plan
  overlap    total overlap between rooms in the stitched plan (m2)
  placement  room centre error after one best rigid fit of the stitched plan onto the truth (largest island)

    python bench/stitch_benchmark.py [--seeds 2] [--levels exact photo hard] [--set HULL_PER_M2=0.5] [--out FILE]
Writes bench/results/stitch_benchmark.json. --set overrides a solver weight for an ablation (recorded in the output).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cozmo.stitch.solver as solver  # noqa: E402
from cozmo.stitch.solver import Door, Placement, Room, rot, stitch  # noqa: E402

CASES = ROOT / "data" / "derived" / "houselayout_properties.json"
OUT = ROOT / "bench" / "results" / "stitch_benchmark.json"
DOOR_ON_WALL_M = 0.5          # a door further than this from a room's outline is not seen from that room
LEVELS = {
    #        scale sigma, outline jitter m, turn sigma deg, door width sigma m, door slide sigma m, miss, false
    "exact": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "photo": (0.03, 0.02, 1.5, 0.05, 0.05, 0.10, 0.05),
    "hard": (0.06, 0.05, 3.0, 0.08, 0.10, 0.20, 0.10),
}


def code_commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                          capture_output=True, text=True).stdout.strip()


def observe_case(case: dict, level: tuple, rng: np.random.Generator):
    """Rooms as the photo tier would report them, each room's true plan polygon, and for every observed door the
    index of the real door it shows (None for a door that is not there)."""
    scale_sd, jitter, turn_sd, width_sd, slide_sd, miss, false = level
    rooms, truths, door_ids = [], [], []
    for row in case["rooms"]:
        truth = Polygon(row["polygon"])
        if not truth.is_valid:
            truth = truth.buffer(0)
            if truth.geom_type == "MultiPolygon":
                truth = max(truth.geoms, key=lambda g: g.area)
        centroid = np.array(truth.centroid.coords[0])
        placement = Placement(float(rng.integers(4)) * np.pi / 2 + np.radians(rng.normal(0, turn_sd)),
                              tuple(rng.uniform(-20, 20, 2)))
        scale = 1.0 + rng.normal(0, scale_sd)
        to_local = lambda xy: ((np.asarray(xy, float) - centroid) * scale + centroid - placement.shift) @ rot(placement.angle)
        outline = np.array(truth.exterior.coords)[:-1]
        local = to_local(outline) + rng.normal(0, jitter, outline.shape)
        doors, ids = [], []
        for door_index, d in enumerate(case["doors"]):
            if row["id"] not in d["rooms"] or rng.random() < miss:
                continue
            centre = Point(d["centre"])
            if truth.exterior.distance(centre) > DOOR_ON_WALL_M:
                continue
            on_wall = np.array(truth.exterior.interpolate(truth.exterior.project(centre)).coords[0])
            normal = np.array(d["normal"], float)
            if truth.contains(Point(*(on_wall + 0.3 * normal))):
                normal = -normal                      # make it point out of this room
            along = np.array([-normal[1], normal[0]])
            on_wall = on_wall + along * rng.normal(0, slide_sd)
            doors.append(Door(tuple(to_local(on_wall)), tuple(rot(placement.angle).T @ normal),
                              float(max(0.4, d["width_m"] + rng.normal(0, width_sd)))))
            ids.append(door_index)
        if rng.random() < false and len(outline) >= 3:
            edge = int(rng.integers(len(outline)))
            a, b = outline[edge], outline[(edge + 1) % len(outline)]
            if np.linalg.norm(b - a) > 1.2:
                point = a + (b - a) * rng.uniform(0.3, 0.7)
                direction = (b - a) / np.linalg.norm(b - a)
                normal = np.array([direction[1], -direction[0]])
                if truth.contains(Point(*(point + 0.3 * normal))):
                    normal = -normal
                doors.append(Door(tuple(to_local(point)), tuple(rot(placement.angle).T @ normal), 0.8))
                ids.append(None)
        rooms.append(Room(str(row["id"]), local, doors))
        truths.append(truth)
        door_ids.append(ids)
    return rooms, truths, door_ids


def true_adjacency(case: dict) -> set[frozenset]:
    index = {row["id"]: k for k, row in enumerate(case["rooms"])}
    return {frozenset((index[a], index[b])) for a, b in (d["rooms"] for d in case["doors"])
            if a is not None and b is not None and a != b}


def placement_error(rooms: list[Room], truths: list[Polygon], result) -> tuple[float | None, float | None]:
    """Median and max room-centre error of the largest island after one rigid fit onto the truth."""
    placed = sorted(result.placements)
    if len(placed) < 2:
        return None, None
    got = np.array([Polygon(result.placements[k].apply(rooms[k].polygon)).centroid.coords[0] for k in placed])
    want = np.array([truths[k].centroid.coords[0] for k in placed])
    # largest island: rooms connected through door pairs
    parent = {k: k for k in placed}

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k

    for p in result.pairs:
        parent[find(p.room_a)] = find(p.room_b)
    groups: dict[int, list[int]] = {}
    for i, k in enumerate(placed):
        groups.setdefault(find(k), []).append(i)
    idx = max(groups.values(), key=len)
    if len(idx) < 2:
        return None, None
    g, w = got[idx], want[idx]
    gm, wm = g.mean(axis=0), w.mean(axis=0)
    U, _, Vt = np.linalg.svd((w - wm).T @ (g - gm))
    R = U @ np.diag([1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    err = np.linalg.norm((g - gm) @ R.T + wm - w, axis=1)
    return float(np.median(err)), float(err.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="override a solver constant")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    overrides = {}
    for item in args.set:
        name, value = item.split("=", 1)
        if not hasattr(solver, name):
            raise SystemExit(f"cozmo.stitch.solver has no constant {name}")
        setattr(solver, name, type(getattr(solver, name))(value))
        overrides[name] = getattr(solver, name)
    commit = code_commit()
    cases = json.loads(CASES.read_text())["cases"]
    rows = []
    for level in args.levels:
        for c, case in enumerate(cases):
            truth_adj = true_adjacency(case)
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * c + seed)
                rooms, truths, door_ids = observe_case(case, LEVELS[level], rng)
                t0 = time.time()
                result = stitch(rooms)
                seconds = time.time() - t0
                got = result.adjacency()
                hit = len(got & truth_adj)
                right_doors = sum(1 for q in result.pairs if door_ids[q.room_a][q.door_a] is not None
                                  and door_ids[q.room_a][q.door_a] == door_ids[q.room_b][q.door_b])
                # real doors both of whose sides were observed: the most the solver could pair
                seen = [set(i for i in ids if i is not None) for ids in door_ids]
                pairable = sum(1 for i, d in enumerate(case["doors"]) if None not in d["rooms"]
                               and sum(i in s for s in seen) == 2)
                median_err, max_err = placement_error(rooms, truths, result)
                rows.append({"level": level, "case": f"{case['scene']}@{case['level_z']}", "rooms": len(rooms),
                             "seed": seed, "true_pairs": len(truth_adj), "found_pairs": len(got),
                             "precision": round(hit / len(got), 3) if got else None,
                             "recall": round(hit / len(truth_adj), 3) if truth_adj else None,
                             "exact_adjacency": got == truth_adj, "islands": result.islands,
                             "door_pairs": len(result.pairs), "right_door_pairs": right_doors, "pairable_doors": pairable,
                             "door_precision": round(right_doors / len(result.pairs), 3) if result.pairs else None,
                             "door_recall": round(right_doors / pairable, 3) if pairable else None,
                             "all_doors_right": right_doors == pairable and right_doors == len(result.pairs),
                             "overlap_m2": round(result.overlap_m2, 2),
                             "placement_error_m_median": round(median_err, 3) if median_err is not None else None,
                             "placement_error_m_max": round(max_err, 3) if max_err is not None else None,
                             "seconds": round(seconds, 2)})
                r = rows[-1]
                print(f"{level:5s} {r['case']:22s} rooms {r['rooms']:2d} pairs {r['found_pairs']}/{r['true_pairs']} "
                      f"P {r['precision']} R {r['recall']} exact {r['exact_adjacency']} doors {r['right_door_pairs']}/{r['pairable_doors']} "
                      f"(made {r['door_pairs']}) islands {r['islands']} "
                      f"overlap {r['overlap_m2']} err {r['placement_error_m_median']} {r['seconds']} s", flush=True)
    summary = {}
    for level in args.levels:
        sel = [r for r in rows if r["level"] == level]
        prec = [r["precision"] for r in sel if r["precision"] is not None]
        rec = [r["recall"] for r in sel if r["recall"] is not None]
        errs = [r["placement_error_m_median"] for r in sel if r["placement_error_m_median"] is not None]
        dprec = [r["door_precision"] for r in sel if r["door_precision"] is not None]
        drec = [r["door_recall"] for r in sel if r["door_recall"] is not None]
        summary[level] = {"runs": len(sel), "exact_adjacency": f"{sum(r['exact_adjacency'] for r in sel)}/{len(sel)}",
                          "all_doors_right": f"{sum(r['all_doors_right'] for r in sel)}/{len(sel)}",
                          "door_precision_mean": round(float(np.mean(dprec)), 3) if dprec else None,
                          "door_recall_mean": round(float(np.mean(drec)), 3) if drec else None,
                          "one_island": f"{sum(r['islands'] == 1 for r in sel)}/{len(sel)}",
                          "precision_mean": round(float(np.mean(prec)), 3) if prec else None,
                          "recall_mean": round(float(np.mean(rec)), 3) if rec else None,
                          "overlap_m2_median": round(float(np.median([r["overlap_m2"] for r in sel])), 2),
                          "placement_error_m_median": round(float(np.median(errs)), 3) if errs else None,
                          "seconds_median": round(float(np.median([r["seconds"] for r in sel])), 2)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"benchmark": "stitch", "code_commit": commit, "cases": str(CASES.relative_to(ROOT)),
                               "overrides": overrides, "levels": {k: LEVELS[k] for k in args.levels},
                               "summary": summary, "runs": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
