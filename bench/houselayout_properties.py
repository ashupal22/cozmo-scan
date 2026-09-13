"""Stitching test cases from HouseLayout3D (MIT): real buildings cut into rooms that doors join.

HouseLayout3D gives walls, floors and ceilings as a mesh, and doors as rectangles, but no room labels. Per floor
level (a horizontal floor area with walls rising from it), walls are cut at 1 m, doors are treated as closed,
and the connected floor regions are the rooms. Open passages without a door stay inside one room, which is
also how a person would put an open-plan space in one photo folder. A door joins the rooms found a little way
out on either side of it, or a room and the outside.

A test case ("property") is a group of at least MIN_ROOMS rooms that doors connect. It holds each room's
outline and area, and each door's centre, width, direction and rooms, in the building's frame: the true
arrangement a stitch solver must recover from separately framed rooms.

    python bench/houselayout_properties.py
Writes data/derived/houselayout_properties.json (data, not committed).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "external" / "houselayout3d"
OUT = ROOT / "data" / "derived" / "houselayout_properties.json"
CELL = 0.03
CUT_ABOVE_FLOOR_M = 1.0
MIN_ROOM_M2 = 1.0
MIN_ROOMS = 3
PROBE_M = (0.15, 0.25, 0.4, 0.6)
SIMPLIFY_M = 0.05


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    V, F = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V), np.array(F)


def cut_segments(triangles: np.ndarray, z: float) -> list[tuple[np.ndarray, np.ndarray]]:
    segments = []
    for tri in triangles:
        d = tri[:, 2] - z
        points = [tri[a, :2] + d[a] / (d[a] - d[b]) * (tri[b, :2] - tri[a, :2])
                  for a, b in ((0, 1), (1, 2), (2, 0)) if (d[a] < 0) != (d[b] < 0)]
        if len(points) == 2:
            segments.append((points[0], points[1]))
    return segments


def floor_levels(tri: np.ndarray, normal: np.ndarray, area: np.ndarray) -> list[float]:
    horizontal, wall = np.abs(normal[:, 2]) > 0.9, np.abs(normal[:, 2]) < 0.3
    z = np.round(tri[:, :, 2].mean(axis=1) / 0.1) * 0.1
    levels = []
    for level in np.unique(z[horizontal]):
        if area[horizontal & (z == level)].sum() < 5:
            continue
        rising = wall & (tri[:, :, 2].min(axis=1) < level + 0.3) & (tri[:, :, 2].max(axis=1) > level + 1.2)
        if rising.sum() > 10 and all(abs(level - other) > 1.5 for other in levels):
            levels.append(float(level))
    return levels


def scene_properties(scene: str) -> list[dict]:
    V, F = load_obj(SOURCE / "structures" / f"{scene}.obj")
    doors = json.loads((SOURCE / "doors" / f"{scene}.json").read_text())["doors"]
    tri = V[F]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(normal, axis=1) / 2
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    horizontal, wall = np.abs(normal[:, 2]) > 0.9, np.abs(normal[:, 2]) < 0.3
    lo = V[:, :2].min(axis=0) - 0.5
    size = np.ceil((V[:, :2].max(axis=0) + 0.5 - lo) / CELL).astype(int)
    to_px = lambda xy: np.round((np.asarray(xy) - lo) / CELL).astype(np.int32)
    to_m = lambda px: np.asarray(px, float) * CELL + lo
    cases = []
    for z0 in floor_levels(tri, normal, area):
        floor = np.zeros((size[1], size[0]), np.uint8)
        for t in tri[horizontal & (np.abs(tri[:, :, 2].mean(axis=1) - z0) < 0.15)]:
            cv2.fillPoly(floor, [to_px(t[:, :2])], 1)
        blocked = np.zeros_like(floor)
        for a, b in cut_segments(tri[wall], z0 + CUT_ABOVE_FLOOR_M):
            cv2.line(blocked, tuple(to_px(a)), tuple(to_px(b)), 1, 2)
        level_doors = []
        for d in doors:
            corners = np.array(d["vertices"])
            if not corners[:, 2].min() < z0 + CUT_ABOVE_FLOOR_M < corners[:, 2].max():
                continue
            bottom = corners[np.argsort(corners[:, 2])[:2], :2]
            n = np.array(d["normal"][:2], float)
            n /= np.linalg.norm(n) + 1e-12
            cv2.line(blocked, tuple(to_px(bottom[0])), tuple(to_px(bottom[1])), 1, 2)
            level_doors.append((bottom.mean(axis=0), float(np.linalg.norm(bottom[1] - bottom[0])), n, bottom))
        free = ((floor > 0) & (cv2.dilate(blocked, np.ones((3, 3), np.uint8)) == 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
        rooms = {k for k in range(1, count) if stats[k, cv2.CC_STAT_AREA] * CELL ** 2 >= MIN_ROOM_M2}

        def room_at(point):
            q = to_px(point)
            if 0 <= q[0] < size[0] and 0 <= q[1] < size[1] and labels[q[1], q[0]] in rooms:
                return int(labels[q[1], q[0]])
            return None

        door_rows = []
        for centre, width, n, bottom in level_doors:
            sides = [next((r for r in (room_at(centre + s * m * n) for m in PROBE_M) if r is not None), None) for s in (1, -1)]
            if sides[0] is not None and sides[0] == sides[1]:
                continue                        # a door inside one room region (e.g. a cupboard)
            if sides[0] is None and sides[1] is None:
                continue
            door_rows.append({"centre": [round(float(v), 3) for v in centre], "width_m": round(width, 3),
                              "normal": [round(float(v), 4) for v in n], "rooms": sides,
                              "ends": [[round(float(v), 3) for v in p] for p in bottom]})
        # groups of rooms joined by doors
        parent = {r: r for r in rooms}

        def find(r):
            while parent[r] != r:
                parent[r] = parent[parent[r]]
                r = parent[r]
            return r

        for d in door_rows:
            a, b = d["rooms"]
            if a is not None and b is not None:
                parent[find(a)] = find(b)
        groups: dict[int, list[int]] = {}
        for r in rooms:
            groups.setdefault(find(r), []).append(r)
        for members in groups.values():
            if len(members) < MIN_ROOMS:
                continue
            member_set = set(members)
            room_rows = []
            for r in sorted(members):
                contours, _ = cv2.findContours((labels == r).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                outline = max(contours, key=cv2.contourArea)
                simple = cv2.approxPolyDP(outline, SIMPLIFY_M / CELL, True)[:, 0, :]
                room_rows.append({"id": int(r), "area_m2": round(float(stats[r, cv2.CC_STAT_AREA] * CELL ** 2), 2),
                                  "polygon": [[round(float(v), 3) for v in to_m(p)] for p in simple]})
            case_doors = [d for d in door_rows if any(x in member_set for x in d["rooms"] if x is not None)]
            for d in case_doors:
                d["rooms"] = [x if x in member_set else None for x in d["rooms"]]
            inner = sum(1 for d in case_doors if None not in d["rooms"])
            cases.append({"scene": scene, "level_z": round(z0, 2), "rooms": room_rows, "doors": case_doors,
                          "inner_doors": inner, "outer_doors": len(case_doors) - inner,
                          "area_m2": round(sum(r["area_m2"] for r in room_rows), 2)})
    return cases


def main():
    cases = []
    for doors in sorted((SOURCE / "doors").glob("*.json")):
        if (SOURCE / "structures" / f"{doors.stem}.obj").is_file():
            cases += scene_properties(doors.stem)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": "HouseLayout3D (MIT)", "cases": cases}, indent=1) + "\n")
    for c in cases:
        print(f"{c['scene']} z={c['level_z']:5.1f}: {len(c['rooms'])} rooms, {c['area_m2']} m2, "
              f"{c['inner_doors']} doors between rooms, {c['outer_doors']} to elsewhere")
    print(f"{len(cases)} cases, {sum(len(c['rooms']) for c in cases)} rooms; wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
