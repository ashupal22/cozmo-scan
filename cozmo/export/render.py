"""Draw the plan from an output document as SVG: rooms, wall lengths with ranges, openings."""
from __future__ import annotations

import numpy as np

PX_PER_M = 80
MARGIN_PX = 70
FILLS = ["#e8eef7", "#eef5e9", "#f7efe6", "#f1e9f5", "#e6f3f3", "#f6f3e2", "#f3e6ea", "#ebebeb"]


def _rotation(yaw_deg: float) -> np.ndarray:
    t = np.radians(yaw_deg)
    return np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])


def _range(m: dict, digits: int = 2) -> str:
    half = (m["ci_high"] - m["ci_low"]) / 2
    return f"{m['value']:.{digits}f} ± {half:.{digits}f}"


def render_svg(document: dict, yaw_deg: float = 0.0) -> str:
    """Plan view seen from above (x right, z down), turned so the main walls are horizontal."""
    R = _rotation(yaw_deg)
    rooms = document["rooms"]
    if not rooms:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="80"><text x="10" y="40">no rooms</text></svg>'
    all_pts = np.vstack([np.array(r["polygon"]) @ R.T for r in rooms])
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    width = int((hi[0] - lo[0]) * PX_PER_M + 2 * MARGIN_PX)
    height = int((hi[1] - lo[1]) * PX_PER_M + 2 * MARGIN_PX + 40)

    def px(xz):
        uv = np.asarray(xz) @ R.T
        return (uv - lo) * PX_PER_M + MARGIN_PX

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           'font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    for i, room in enumerate(rooms):
        poly = px(room["polygon"])
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in poly)
        out.append(f'<polygon points="{points}" fill="{FILLS[i % len(FILLS)]}" stroke="#222" stroke-width="3" '
                   'stroke-linejoin="miter"/>')
    for room in rooms:
        center = px(room["polygon"]).mean(axis=0)
        ceiling = room["ceiling_height_m"]
        ceiling_text = ("ceiling " + _range(ceiling) + " m") if ceiling.get("observed", True) else "ceiling not seen"
        out.append(f'<text x="{center[0]:.0f}" y="{center[1] - 8:.0f}" text-anchor="middle" font-size="14" '
                   f'font-weight="bold" fill="#111">{room["label"]}</text>')
        out.append(f'<text x="{center[0]:.0f}" y="{center[1] + 9:.0f}" text-anchor="middle" font-size="11" '
                   f'fill="#333">{_range(room["floor_area_m2"])} m²</text>')
        out.append(f'<text x="{center[0]:.0f}" y="{center[1] + 24:.0f}" text-anchor="middle" font-size="10" '
                   f'fill="#555">{ceiling_text}</text>')
        for wall in room["walls"]:
            a, b = px(wall["start"]), px(wall["end"])
            if np.linalg.norm(b - a) < 45:
                continue  # too short to label legibly
            mid = (a + b) / 2
            inward = center - mid
            inward = inward / (np.linalg.norm(inward) + 1e-9) * 14
            angle = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
            angle = angle - 180 if angle > 90 else angle + 180 if angle < -90 else angle
            x, y = mid + inward
            out.append(f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="middle" font-size="9" fill="#1f3a93" '
                       f'transform="rotate({angle:.1f} {x:.0f} {y:.0f})">{_range(wall["length_m"])}</text>')
        walls = {w["id"]: w for w in room["walls"]}
        for opening in room["openings"]:
            wall = walls[opening["wall_id"]]
            start, end = np.array(wall["start"]), np.array(wall["end"])
            direction = (end - start) / (np.linalg.norm(end - start) + 1e-9)
            c = start + direction * opening["offset_along_wall_m"]["value"]
            half = direction * opening["width_m"]["value"] / 2
            p, q = px(c - half), px(c + half)
            out.append(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}" stroke="#ffffff" '
                       'stroke-width="5"/>')
            out.append(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}" stroke="#c0392b" '
                       'stroke-width="1.5" stroke-dasharray="3,2"/>')
    footprint = document["plan"]["footprint_area_m2"]
    cap = document["capture"]
    out.append(f'<text x="{MARGIN_PX}" y="{height - 18}" font-size="12" fill="#333">{cap["id"]} · {cap["tier"]} · '
               f'{len(rooms)} rooms · footprint {_range(footprint)} m² · ranges are 90%</text>')
    out.append("</svg>")
    return "\n".join(out)
