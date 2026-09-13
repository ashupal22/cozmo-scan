"""Repair scope line items, keyed to surfaces (output contract: `scope`).

Each damage region adds the repair its class needs on its surface, and repainting the whole surface (paint is
applied wall to wall, so a patched wall is repainted to its corners). Each concealed-damage flag adds an
inspection of the cavity behind the surface. Quantities come from the measured plan and carry its intervals:
a wall's area is its length times the room's ceiling height, a ceiling's is the room's floor area.
"""
from __future__ import annotations

import numpy as np

REPAIRS = {
    "water_stain": ("seal the stain with stain-blocking primer", "m2"),
    "mold": ("clean and treat mold (surface remediation), 0.3 m beyond the visible edge", "m2"),
    "crack": ("tape, fill and sand the crack", "m"),
    "peeling_paint": ("scrape loose paint and prime", "m2"),
    "hole": ("patch the hole in the plasterboard", "each"),
}
MOLD_MARGIN_M = 0.3


def _m(value: float, low: float, high: float, method: str | None = None) -> dict:
    m = {"value": round(value, 3), "ci_low": round(max(low, 0.0), 3), "ci_high": round(high, 3), "confidence": 0.9}
    if method:
        m["method"] = method
    return m


def _surface_area(document: dict, surface_id: str) -> dict:
    room_id, part = surface_id.split(".")
    room = next(r for r in document["rooms"] if r["id"] == room_id)
    if part in ("FLOOR", "CEIL"):
        a = room["floor_area_m2"]
        return _m(a["value"], a["ci_low"], a["ci_high"], "room floor area")
    wall = next(w for w in room["walls"] if w["id"] == surface_id)
    length, height = wall["length_m"], room["ceiling_height_m"]
    value = length["value"] * height["value"]
    rel = np.hypot((length["ci_high"] - length["value"]) / max(length["value"], 1e-6),
                   (height["ci_high"] - height["value"]) / max(height["value"], 1e-6))
    return _m(value, value * (1 - rel), value * (1 + rel), "wall length x ceiling height")


def scope_items(document: dict, damage: list[dict], flags: list[dict]) -> list[dict]:
    items = []

    def add(surface_id, description, unit, quantity, because):
        items.append({"id": f"S{len(items) + 1}", "surface_id": surface_id, "description": description, "unit": unit,
                      "quantity": quantity, "because": because})

    painted = {}
    for region in damage:
        what, unit = REPAIRS[region["class"]]
        area, width, height = region["area_m2"], region["width_m"], region["height_m"]
        if unit == "m2" and region["class"] == "mold":
            grow = lambda v: (v + 2 * MOLD_MARGIN_M)  # noqa: E731
            q = _m(grow(width["value"]) * grow(height["value"]), grow(width["ci_low"]) * grow(height["ci_low"]),
                   grow(width["ci_high"]) * grow(height["ci_high"]))
        elif unit == "m2":
            q = _m(area["value"], area["ci_low"], area["ci_high"])
        elif unit == "m":
            long = "width_m" if width["value"] >= height["value"] else "height_m"
            q = _m(region[long]["value"], region[long]["ci_low"], region[long]["ci_high"])
        else:
            q = _m(1, 1, 1)
        add(region["surface_id"], what, unit, q, {"damage_ids": [region["id"]]})
        painted.setdefault(region["surface_id"], []).append(region["id"])
    for surface_id, ids in painted.items():
        add(surface_id, "repaint the whole surface (two coats)", "m2", _surface_area(document, surface_id),
            {"damage_ids": ids})
    for flag in flags:
        add(flag["surface_id"], f"inspect the concealed cavity: {flag['description'].split(' (because')[0]}", "each",
            _m(1, 1, 1), {"rule_ids": [flag["rule_id"]], "damage_ids": list(flag["evidence"])})
    return items
