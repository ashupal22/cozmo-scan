"""A one-page visual report of a result, written next to result.json as report.png and report.pdf.

It shows the floor plan (rooms, wall lengths, openings, damage markers) and, beside it, what is otherwise buried in the
JSON:
- a table of rooms: width × length, area, ceiling height and openings, with their 90% ranges;
- the plan: footprint, which rooms connect, how rooms were joined, drift correction;
- damage regions, concealed-damage flags and repair scope;
- the first warnings.

Drawn with matplotlib's object-oriented API (no pyplot, no browser), so it works on any machine and never touches a
GUI backend.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from cozmo.export.summary import main_dimensions

A4_LANDSCAPE_IN = (11.69, 8.27)
FILLS = ["#dbe7f5", "#dff0d8", "#f7e5d0", "#eadcf2", "#d6eeee", "#f4efcf", "#f3dbe2", "#e6e6e6"]
INK, MUTED, RULE = "#1d1d1f", "#5f6368", "#c9ccd1"
WALL_TEXT, OPENING, DAMAGE = "#1f3a93", "#1e8449", "#c0392b"
TIER_COLOURS = {"lidar": "#1f5fa8", "video": "#b35c00", "photo": "#6d3fa0"}
MAX_ROOM_ROWS = 14
MAX_WARNINGS = 6
MIN_LABELLED_WALL_M = 1.0
LABEL_CLEAR_M = 0.55      # keep wall lengths off the room label
LABEL_GAP_M = 0.35        # and off each other


def _rng(m: dict, digits: int = 2) -> str:
    return f"{m['value']:.{digits}f} ({m['ci_low']:.{digits}f}–{m['ci_high']:.{digits}f})"


def _pm(value: float, interval: list, digits: int = 2) -> str:
    return f"{value:.{digits}f}±{(interval[1] - interval[0]) / 2:.{digits}f}"


def _dominant_angle(polygons: list[np.ndarray]) -> float:
    total = 0j
    for poly in polygons:
        edges = np.roll(poly, -1, axis=0) - poly
        total += np.sum(np.linalg.norm(edges, axis=1) * np.exp(4j * np.arctan2(edges[:, 1], edges[:, 0])))
    return float(np.angle(total) / 4) if total else 0.0


def _draw_plan(ax, document: dict) -> None:
    from matplotlib.patches import Polygon as Patch
    from shapely.geometry import Polygon

    rooms = document["rooms"]
    ax.set_axis_off()
    if not rooms:
        ax.text(0.5, 0.5, "No rooms found", ha="center", va="center", fontsize=14, color=MUTED, transform=ax.transAxes)
        return
    a = _dominant_angle([np.asarray(r["polygon"], float) for r in rooms])
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])

    def view(points):
        return (np.asarray(points, float) @ R.T) * np.array([1.0, -1.0])   # walls horizontal, seen from above

    small = len(rooms) > 8
    walls_by_id, anchor, placed = {}, {}, []
    for i, room in enumerate(rooms):
        poly = view(room["polygon"])
        ax.add_patch(Patch(poly, closed=True, facecolor=FILLS[i % len(FILLS)], edgecolor=INK, linewidth=1.6, joinstyle="miter"))
        shape = Polygon(poly)
        centre = np.array((shape if shape.is_valid else shape.buffer(0)).representative_point().coords[0])
        anchor[room["id"]] = centre
        ax.text(*centre, f"{room['id']}\n{room['floor_area_m2']['value']:.1f} m²", ha="center", va="center",
                fontsize=6.5 if small else 8, color=INK, fontweight="bold", linespacing=1.15)
        for wall in room["walls"]:
            walls_by_id[wall["id"]] = wall
            p, q = view(wall["start"]), view(wall["end"])
            if wall["length_m"]["value"] < MIN_LABELLED_WALL_M:
                continue
            mid = (p + q) / 2
            inward = centre - mid
            inward = inward / (np.linalg.norm(inward) + 1e-9) * (0.16 if small else 0.2)
            angle = np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0]))
            angle = angle - 180 if angle > 90 else angle + 180 if angle < -90 else angle
            spot = mid + inward
            if np.linalg.norm(spot - centre) < LABEL_CLEAR_M or any(np.linalg.norm(spot - other) < LABEL_GAP_M for other in placed):
                continue      # would sit on the room label or on another wall's length
            placed.append(spot)
            ax.text(*spot, f"{wall['length_m']['value']:.2f}", ha="center", va="center", rotation=angle,
                    rotation_mode="anchor", fontsize=5 if small else 6, color=WALL_TEXT)
        walls = {w["id"]: w for w in room["walls"]}
        for opening in room["openings"]:
            wall = walls.get(opening["wall_id"])
            if wall is None:
                continue
            s, e = np.asarray(wall["start"], float), np.asarray(wall["end"], float)
            d = (e - s) / (np.linalg.norm(e - s) + 1e-9)
            c = s + d * opening["offset_along_wall_m"]["value"] if "offset_along_wall_m" in opening else (s + e) / 2
            half = d * opening["width_m"]["value"] / 2
            seg = view(np.array([c - half, c + half]))
            ax.plot(seg[:, 0], seg[:, 1], color="white", linewidth=3.2, solid_capstyle="butt", zorder=3)
            ax.plot(seg[:, 0], seg[:, 1], color=OPENING, linewidth=1.4, solid_capstyle="butt", zorder=4)
    for region in document["damage"]:
        room_id, part = region["surface_id"].split(".")
        wall = walls_by_id.get(region["surface_id"])
        if wall is not None:
            s, e = np.asarray(wall["start"], float), np.asarray(wall["end"], float)
            u = float(np.mean([pt[0] for pt in region.get("outline_uv", [[0.0, 0.0]])]))
            point = view(s + (e - s) / (np.linalg.norm(e - s) + 1e-9) * u)
        else:
            point = anchor.get(room_id, np.zeros(2)) + np.array([0.0, -0.35])
        ax.scatter(*point, marker="X", s=46, color=DAMAGE, zorder=5, linewidths=0.4, edgecolors="white")
        ax.text(point[0] + 0.12, point[1] + 0.12, region["id"], fontsize=6, color=DAMAGE, fontweight="bold", zorder=6)
    ax.autoscale_view()
    ax.set_aspect("equal")
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    pad = 0.04 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - 2 * pad, y1 + pad)
    bar_y = y0 - 1.2 * pad
    ax.plot([x0, x0 + 1.0], [bar_y, bar_y], color=INK, linewidth=2)
    ax.text(x0 + 0.5, bar_y - 0.35 * pad, "1 m", ha="center", va="top", fontsize=7, color=INK)


class _Column:
    """Top-down text in an axes, in axes coordinates, with a check that it stays on the page."""

    def __init__(self, ax):
        self.ax, self.y = ax, 0.995

    def fits(self, lines: float = 1) -> bool:
        return self.y - 0.0185 * lines > 0.01

    def heading(self, text: str) -> None:
        self.y -= 0.012
        self.ax.text(0.0, self.y, text, fontsize=10, fontweight="bold", color=INK, va="top", transform=self.ax.transAxes)
        self.y -= 0.028
        self.ax.plot([0, 1], [self.y + 0.004, self.y + 0.004], color=RULE, linewidth=0.6, transform=self.ax.transAxes)

    def line(self, text: str, size: float = 6.8, colour: str = INK, bold: bool = False, x: float = 0.0) -> None:
        self.ax.text(x, self.y, text, fontsize=size, color=colour, va="top", fontweight="bold" if bold else "normal",
                     transform=self.ax.transAxes)

    def advance(self, lines: float = 1) -> None:
        self.y -= 0.0185 * lines

    def wrapped(self, text: str, width: int = 92, max_lines: int = 2, size: float = 6.6, colour: str = INK) -> None:
        parts = textwrap.wrap(text, width) or [""]
        if len(parts) > max_lines:
            parts = parts[:max_lines]
            parts[-1] = parts[-1][: width - 1].rstrip() + "…"
        for part in parts:
            if not self.fits():
                return
            self.line(part, size=size, colour=colour)
            self.advance()


def _rooms_table(col: _Column, rooms: list[dict]) -> None:
    col.heading("Rooms")
    xs = [0.0, 0.1, 0.43, 0.63, 0.8]
    for x, head in zip(xs, ["Room", "Width × length (m)", "Area (m²)", "Ceiling (m)", "Openings (m)"]):
        col.line(head, size=6.6, colour=MUTED, bold=True, x=x)
    col.advance(1.1)
    for room in rooms[:MAX_ROOM_ROWS]:
        (w, l), (wi, li) = main_dimensions(room)
        ceiling = room["ceiling_height_m"]
        ceiling_text = (f"{ceiling['value']:.2f}*" if ceiling.get("observed", True) is False
                        else _pm(ceiling["value"], [ceiling["ci_low"], ceiling["ci_high"]]))
        widths = [("~" if o["width_m"].get("observed", True) is False else "") + f"{o['width_m']['value']:.2f}"
                  for o in room["openings"]]
        openings = ", ".join(widths[:3]) + (f" +{len(widths) - 3}" if len(widths) > 3 else "") if widths else "—"
        area = room["floor_area_m2"]
        cells = [room["id"], f"{_pm(w, wi)} × {_pm(l, li)}", f"{area['value']:.1f} ({area['ci_low']:.1f}–{area['ci_high']:.1f})",
                 ceiling_text, openings]
        for x, cell in zip(xs, cells):
            col.line(cell, x=x)
        col.advance()
    if len(rooms) > MAX_ROOM_ROWS:
        col.line(f"+ {len(rooms) - MAX_ROOM_ROWS} more rooms in result.json", colour=MUTED)
        col.advance()


def render_report(document: dict, out_dir) -> tuple[Path, Path]:
    """Write report.png and report.pdf for `document` into `out_dir`; return their paths."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture, plan, rooms = document["capture"], document["plan"], document["rooms"]
    tier = capture["tier"]
    fig = Figure(figsize=A4_LANDSCAPE_IN, facecolor="white")
    FigureCanvasAgg(fig)

    title = fig.text(0.025, 0.962, capture["id"], fontsize=15, fontweight="bold", color=INK, va="center")
    title_end = fig.transFigure.inverted().transform((title.get_window_extent(fig.canvas.get_renderer()).x1, 0))[0]
    fig.text(title_end + 0.015, 0.962, f"  {tier.upper()} TIER  ", fontsize=8.5, fontweight="bold",
             color="white", va="center", bbox=dict(boxstyle="round,pad=0.35", facecolor=TIER_COLOURS.get(tier, MUTED), edgecolor="none"))
    runtime = f" · {capture['runtime_s']:.0f} s" if capture.get("runtime_s") is not None else ""
    version = f" · pipeline {capture['pipeline_version']}" if capture.get("pipeline_version") else ""
    fig.text(0.025, 0.925, f"{len(rooms)} rooms · footprint {_rng(plan['footprint_area_m2'])} m²{runtime}{version}",
             fontsize=9, color=MUTED, va="center")
    fig.text(0.025, 0.02, "Lengths in metres. ± and (low–high) are 90% ranges. * not seen: a typical range, not a measurement. "
             "~ width not measured. Plan: blue numbers are wall lengths, green lines are openings, red × are damage regions.",
             fontsize=6.8, color=MUTED)

    _draw_plan(fig.add_axes([0.02, 0.06, 0.55, 0.84]), document)

    ax = fig.add_axes([0.6, 0.06, 0.38, 0.84])
    ax.set_axis_off()
    col = _Column(ax)
    _rooms_table(col, rooms)

    col.heading("Plan")
    adjacency = ", ".join(f"{a['room_a']}–{a['room_b']}" for a in plan["adjacency"]) or "none found"
    col.wrapped(f"Footprint {_rng(plan['footprint_area_m2'])} m². Connections: {adjacency}.", max_lines=3)
    stitch = plan["stitch"]
    joined = {"single_map": "one continuous scan of all rooms", "single_room": "a single room",
              "door_matching": "rooms joined through matching doors"}.get(stitch["method"], stitch["method"])
    col.wrapped(f"Plan built from {joined}" + (" (low confidence)." if stitch.get("low_confidence") else "."))
    drift = plan["drift"]
    if drift.get("enabled"):
        loops = drift.get("loop_closures")
        col.wrapped("Drift correction on" + (f": {loops} loop closure{'s' if loops != 1 else ''}." if loops is not None else "."))
    else:
        col.wrapped(f"Drift correction off: {drift.get('notes', 'not applied')}.", max_lines=2)

    col.heading("Damage and repairs")
    damage, flags, scope = document["damage"], document["concealed_flags"], document["scope"]
    if not damage:
        col.wrapped("No damage regions found, so no concealed-damage flags and no repair items.")
    for region in damage[:4]:
        col.wrapped(f"{region['id']} {region['class'].replace('_', ' ')} on {region['surface_id']}: "
                    f"{region['width_m']['value']:.2f} × {region['height_m']['value']:.2f} m"
                    + (f", confidence {region['detection_confidence']:.2f}" if "detection_confidence" in region else ""), max_lines=1)
    for flag in flags[:3]:
        col.wrapped(f"{flag['id']} {flag['rule_id']} ({flag.get('severity', '')}): {flag['description']}", max_lines=1)
    for item in scope[:3]:
        q = item["quantity"]
        col.wrapped(f"{item['id']} {item['surface_id']}: {item['description']}, {q['value']:.2f} {item['unit']}", max_lines=1)
    extra = max(len(damage) - 4, 0) + max(len(flags) - 3, 0) + max(len(scope) - 3, 0)
    if extra:
        col.wrapped(f"+ {extra} more damage, flag and scope entries in result.json", colour=MUTED, max_lines=1)

    warnings = document["quality"]["warnings"]
    col.heading(f"Warnings ({len(warnings)})")
    for warning in warnings[:MAX_WARNINGS]:
        if not col.fits(2):
            break
        col.wrapped("• " + warning, max_lines=2, colour=INK)
    if len(warnings) > MAX_WARNINGS and col.fits():
        col.wrapped(f"+ {len(warnings) - MAX_WARNINGS} more in result.json", colour=MUTED, max_lines=1)

    png, pdf = out_dir / "report.png", out_dir / "report.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    return png, pdf
