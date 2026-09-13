"""Concealed-damage flags: what a visible damage region suggests behind or above its surface.

RULES is the rules file that every flag's `rule_id` refers to; docs/damage_rules.md lists the same table with
the reasoning. A rule fires on one damage region when the region's class, surface kind and position match.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rule:
    id: str
    when: str                      # the evidence, in words
    flag: str                      # what may be hidden
    severity: str
    classes: frozenset
    kinds: frozenset               # wall, floor, ceiling
    max_bottom_m: float | None = None    # walls: region starts at most this high above the floor
    min_length_m: float | None = None    # region at least this long in one direction
    extra: dict = field(default_factory=dict)


RULES = [
    Rule("R-CEIL-WATER", "water stain or mold on a ceiling",
         "possible leak above: wet insulation, joists or subfloor of the room above", "high",
         frozenset({"water_stain", "mold"}), frozenset({"ceiling"})),
    Rule("R-WALL-BASE-WATER", "water stain or mold on a wall, starting within 0.3 m of the floor",
         "possible wet wall cavity, rotten sole plate or damp subfloor behind the skirting", "high",
         frozenset({"water_stain", "mold"}), frozenset({"wall"}), max_bottom_m=0.3),
    Rule("R-MOLD-HIDDEN", "mold on any surface",
         "mold on a finish usually continues inside the cavity behind it", "high",
         frozenset({"mold"}), frozenset({"wall", "floor", "ceiling"})),
    Rule("R-PEEL-MOISTURE", "peeling paint on a wall or ceiling",
         "moisture behind the paint layer; check the substrate with a moisture meter", "medium",
         frozenset({"peeling_paint"}), frozenset({"wall", "ceiling"})),
    Rule("R-CRACK-LONG", "a crack at least 1 m long on a wall or ceiling",
         "possible structural movement or settlement behind the finish", "medium",
         frozenset({"crack"}), frozenset({"wall", "ceiling"}), min_length_m=1.0),
    Rule("R-HOLE-SERVICES", "a hole in a wall",
         "wiring or pipes behind the hole may be damaged", "low",
         frozenset({"hole"}), frozenset({"wall"})),
]
RULE_IDS = {r.id for r in RULES}


def _kind(surface_id: str) -> str:
    part = surface_id.split(".")[1]
    return "floor" if part == "FLOOR" else "ceiling" if part == "CEIL" else "wall"


def fires(rule: Rule, region: dict) -> bool:
    if region["class"] not in rule.classes or _kind(region["surface_id"]) not in rule.kinds:
        return False
    if rule.max_bottom_m is not None:
        bottom = region.get("bottom_above_floor_m")
        if bottom is None or bottom["value"] > rule.max_bottom_m:
            return False
    if rule.min_length_m is not None and max(region["width_m"]["value"], region["height_m"]["value"]) < rule.min_length_m:
        return False
    return True


def concealed_flags(damage: list[dict]) -> list[dict]:
    """One flag per rule that fires on a region, on that region's surface."""
    flags = []
    for region in damage:
        for rule in RULES:
            if fires(rule, region):
                flags.append({"id": f"F{len(flags) + 1}", "surface_id": region["surface_id"], "rule_id": rule.id,
                              "description": f"{rule.flag} (because: {rule.when})", "evidence": [region["id"]],
                              "severity": rule.severity})
    return flags
