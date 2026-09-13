"""A short summary of a result for people (summary.md next to result.json): each room's width and length, ceiling
height and opening widths with their 90% ranges, the numbers someone with a laser measurer compares first."""
from __future__ import annotations

import numpy as np


def _dominant_angle(poly: np.ndarray) -> float:
    edges = np.roll(poly, -1, axis=0) - poly
    length = np.linalg.norm(edges, axis=1)
    return float(np.angle(np.sum(length * np.exp(4j * np.arctan2(edges[:, 1], edges[:, 0])))) / 4)


def main_dimensions(room: dict) -> tuple[list[float], list[list[float]]]:
    """Sorted (width, length) of a room outline along its own wall directions, each with the range of the wall closest
    to it in length."""
    poly = np.asarray(room["polygon"], float)
    t = _dominant_angle(poly)
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    ext = np.sort(np.ptp(poly @ R.T, axis=0))
    intervals = []
    for e in ext:
        m = min(room["walls"], key=lambda w: abs(w["length_m"]["value"] - e))["length_m"]
        intervals.append([round(float(e) - (m["value"] - m["ci_low"]), 3), round(float(e) + (m["ci_high"] - m["value"]), 3)])
    return [round(float(e), 3) for e in ext], intervals


def _range(m: dict, digits: int = 2) -> str:
    text = f"{m['value']:.{digits}f} ({m['ci_low']:.{digits}f} to {m['ci_high']:.{digits}f})"
    return text + ", not seen" if m.get("observed", True) is False else text


def summary_markdown(doc: dict) -> str:
    capture, plan = doc["capture"], doc["plan"]
    lines = [f"# {capture['id']}: {capture['tier']} tier", "",
             f"{len(doc['rooms'])} rooms, footprint {_range(plan['footprint_area_m2'])} m². Lengths in metres, each with its "
             f"90% range. Width × length is the room's extent along its own walls.", "",
             "| Room | Width × length | Floor area (m²) | Ceiling height | Openings: width | Walls: length |",
             "|---|---|---|---|---|---|"]
    for room in doc["rooms"]:
        (w, l), (wi, li) = main_dimensions(room)
        openings = "; ".join(f"{o['id'].split('.')[-1]} {_range(o['width_m'])}" for o in room["openings"]) or "none found"
        walls = ", ".join(f"{x['id'].split('.')[-1]} {x['length_m']['value']:.2f}" for x in room["walls"])
        lines.append(f"| {room['id']} ({room['label']}) | {w:.2f} ({wi[0]:.2f} to {wi[1]:.2f}) × {l:.2f} ({li[0]:.2f} to {li[1]:.2f}) "
                     f"| {_range(room['floor_area_m2'])} | {_range(room['ceiling_height_m'])} | {openings} | {walls} |")
    if doc["damage"]:
        lines += ["", "Damage:"] + [f"- {d['id']} {d['class']} on {d['surface_id']}: {d['width_m']['value']:.2f} × "
                                    f"{d['height_m']['value']:.2f} m (confidence {d.get('detection_confidence', 0):.2f})"
                                    for d in doc["damage"]]
    if doc["quality"]["warnings"]:
        lines += ["", "Warnings:"] + [f"- {w}" for w in doc["quality"]["warnings"]]
    return "\n".join(lines) + "\n"
