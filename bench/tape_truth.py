"""Tape-measured truth for our own rooms: every tier's error, and the head-to-head against a consumer app (G-H2H).

Input: a measurements CSV (template: see the brief's benchmark set) with columns room, item, tape_m, app_m, notes.
`item` is one of: length, width (wall to wall), ceiling height, door width (clear gap between the frame edges);
damage rows (room "damage N") hold width, height and bottom above floor. `app_m` is the consumer app's value for the
same item, read off its export.

Rooms are tied to each plan by name: --lidar-room "room A=R1" names the LiDAR plan room, --photo-room "room A=bedroom"
the photo folder. Video rooms are matched automatically to the closest main dimensions (disclosed in the output).
A room's length and width are the extents of its outline along its own wall directions. A door is compared with the
measured opening of that room whose width is closest (disclosed). Head-to-head: ours beats or ties the app on a
dimension when our error is at most the app's error plus TIE_M.

    python bench/tape_truth.py --csv measurements.csv --lidar out/lidar/result.json --photo out/photo/result.json \
        --video out/video/result.json --lidar-room "room A=R1" --lidar-room "room B=R4" \
        --photo-room "room A=bedroom" --photo-room "room B=kitchen" --app "magicplan 9.x (free)"
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "results" / "tape_truth.json"
TIE_M = 0.01


def dominant_angle(poly: np.ndarray) -> float:
    edges = np.roll(poly, -1, axis=0) - poly
    length = np.linalg.norm(edges, axis=1)
    angle = np.arctan2(edges[:, 1], edges[:, 0])
    return float(np.angle(np.sum(length * np.exp(4j * angle))) / 4)


def main_dims(room: dict) -> tuple[list[float], list[list[float]]]:
    """Sorted (width, length) of a room outline along its own wall directions, each with the interval of the wall
    closest to it in length."""
    poly = np.asarray(room["polygon"], float)
    t = dominant_angle(poly)
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    ext = np.sort(np.ptp(poly @ R.T, axis=0))
    intervals = []
    for e in ext:
        w = min(room["walls"], key=lambda w: abs(w["length_m"]["value"] - e))["length_m"]
        half_low, half_high = w["value"] - w["ci_low"], w["ci_high"] - w["value"]
        intervals.append([round(e - half_low, 3), round(e + half_high, 3)])
    return [round(float(e), 3) for e in ext], intervals


def read_truth(path: Path) -> dict:
    truth = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            room, item = row["room"].strip(), row["item"].strip().split(" (")[0]
            if not row.get("tape_m", "").strip():
                continue
            entry = truth.setdefault(room, {})
            entry[item] = {"tape": float(row["tape_m"]), "app": float(row["app_m"]) if row.get("app_m", "").strip() else None,
                           "notes": row.get("notes", "").strip()}
    return truth


def compare_room(name: str, measured: dict, room: dict | None) -> list[dict]:
    rows = []
    if room is None:
        return rows
    dims, intervals = main_dims(room)
    tape_dims = sorted(v for k, v in ((k, m["tape"]) for k, m in measured.items()) if k in ("length", "width"))
    items = sorted((k for k in measured if k in ("length", "width")), key=lambda k: measured[k]["tape"])
    for k, ours, iv in zip(items, dims, intervals):
        rows.append({"room": name, "item": k, "tape": measured[k]["tape"], "ours": ours, "interval": iv,
                     "app": measured[k]["app"]})
    if "ceiling height" in measured:
        c = room["ceiling_height_m"]
        rows.append({"room": name, "item": "ceiling height", "tape": measured["ceiling height"]["tape"], "ours": c["value"],
                     "interval": [c["ci_low"], c["ci_high"]], "app": measured["ceiling height"]["app"],
                     "observed": c.get("observed", True)})
    if "door width" in measured and room["openings"]:
        tape = measured["door width"]["tape"]
        op = min(room["openings"], key=lambda o: abs(o["width_m"]["value"] - tape))
        w = op["width_m"]
        rows.append({"room": name, "item": "door width", "tape": tape, "ours": w["value"], "interval": [w["ci_low"], w["ci_high"]],
                     "app": measured["door width"]["app"], "opening": op["id"], "matched": "closest width",
                     "observed": w.get("observed", True)})
    for r in rows:
        r["error_m"] = round(r["ours"] - r["tape"], 3)
        r["error_pct"] = round(100 * (r["ours"] - r["tape"]) / r["tape"], 1)
        r["interval_holds"] = bool(r["interval"][0] <= r["tape"] <= r["interval"][1])
        if r["app"] is not None:
            r["app_error_m"] = round(r["app"] - r["tape"], 3)
            r["beat_or_tie"] = bool(abs(r["error_m"]) <= abs(r["app_error_m"]) + TIE_M)
    return rows


def rooms_by_id(doc: dict) -> dict:
    return {r["id"]: r for r in doc["rooms"]}


def closest_room(doc: dict, measured: dict) -> dict | None:
    tape = sorted(m["tape"] for k, m in measured.items() if k in ("length", "width"))
    if len(tape) != 2:
        return None
    return min(doc["rooms"], key=lambda r: float(np.abs(np.array(main_dims(r)[0]) - np.array(tape)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--lidar")
    ap.add_argument("--photo")
    ap.add_argument("--video")
    ap.add_argument("--lidar-room", action="append", default=[])
    ap.add_argument("--photo-room", action="append", default=[])
    ap.add_argument("--app", default="", help="consumer app name and version")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    truth = read_truth(Path(args.csv))
    rooms = {k: v for k, v in truth.items() if not k.lower().startswith("damage")}
    report = {"benchmark": "tape_truth", "app": args.app, "tie_m": TIE_M, "tiers": {}}
    for tier, path, mapping in (("lidar", args.lidar, args.lidar_room), ("photo", args.photo, args.photo_room),
                                ("video", args.video, [])):
        if not path:
            continue
        doc = json.loads(Path(path).read_text())
        names = dict(m.split("=", 1) for m in mapping)
        by_id = rooms_by_id(doc)
        by_label = {r["label"]: r for r in doc["rooms"]}
        rows = []
        for name, measured in rooms.items():
            if tier == "video":
                room = closest_room(doc, measured)
            else:
                key = names.get(name)
                room = by_id.get(key) or by_label.get(key) if key else None
            got = compare_room(name, measured, room)
            for r in got:
                r["plan_room"] = room["id"] if room else None
                if tier == "video":
                    r["matched_room"] = "closest main dimensions"
            rows += got
        dims = [r for r in rows if r["item"] in ("length", "width")]
        summary = {"dimensions": len(rows),
                   "abs_error_cm_median": round(100 * float(np.median([abs(r["error_m"]) for r in rows])), 1) if rows else None,
                   "walls_within_gate": None, "intervals_hold": f"{sum(r['interval_holds'] for r in rows)}/{len(rows)}"}
        gate = {"lidar": lambda r: abs(r["error_m"]) <= max(0.02, 0.01 * r["tape"]),
                "video": lambda r: abs(r["error_pct"]) <= 3, "photo": lambda r: abs(r["error_pct"]) <= 8}[tier]
        summary["walls_within_gate"] = f"{sum(gate(r) for r in dims)}/{len(dims)}"
        report["tiers"][tier] = {"source": path, "rows": rows, "summary": summary}
        print(tier, json.dumps(summary))
    lidar = report["tiers"].get("lidar", {}).get("rows", [])
    shared = [r for r in lidar if r.get("app") is not None]
    if shared:
        wins = sum(r["beat_or_tie"] for r in shared)
        report["head_to_head"] = {"app": args.app, "shared_dimensions": len(shared), "beat_or_tie": wins,
                                  "share": round(wins / len(shared), 3), "gate_met": wins / len(shared) >= 0.7,
                                  "rows": [{k: r[k] for k in ("room", "item", "tape", "ours", "error_m", "app", "app_error_m", "beat_or_tie")}
                                           for r in shared]}
        print("head-to-head", json.dumps({k: v for k, v in report["head_to_head"].items() if k != "rows"}))
    damage = {k: v for k, v in truth.items() if k.lower().startswith("damage")}
    if damage:
        report["staged_damage"] = damage
        for tier in ("photo", "video", "lidar"):
            path = getattr(args, tier)
            if path:
                report["staged_damage_found_" + tier] = json.loads(Path(path).read_text()).get("damage", [])
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
