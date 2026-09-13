"""Damage regions to concealed-damage flags and scope: rules fire on the right evidence, every item is keyed to an
existing surface, and the document still validates."""
import copy
import json
from pathlib import Path

import numpy as np

from cozmo.damage.detect import Region, Surface, place, to_schema
from cozmo.damage.rules import RULE_IDS, concealed_flags
from cozmo.damage.scope import scope_items
from cozmo.export.validate import validate_output

EXAMPLE = json.loads((Path(__file__).resolve().parents[1] / "schema" / "example_output.json").read_text())


def _region(surface_id, cls, box, kind="wall"):
    room = surface_id.split(".")[0]
    return Region(Surface(surface_id, kind, room, np.zeros(2), np.array([1.0, 0.0])), cls, box, 0.8, {1, 2})


def test_rules_fire_on_their_evidence_only():
    damage = to_schema([_region("R1.W1", "water_stain", [0.5, 1.1, 0.05, 0.6]),      # low on a wall
                        _region("R1.W1", "water_stain", [0.5, 1.1, 1.5, 1.9]),       # high on a wall
                        _region("R1.CEIL", "mold", [0.0, 0.4, 0.0, 0.4], "ceiling"),
                        _region("R1.W2", "crack", [0.2, 0.3, 0.5, 1.8])])            # 1.3 m tall crack
    fired = {(f["evidence"][0], f["rule_id"]) for f in concealed_flags(damage)}
    assert fired == {("D1", "R-WALL-BASE-WATER"), ("D3", "R-CEIL-WATER"), ("D3", "R-MOLD-HIDDEN"),
                     ("D4", "R-CRACK-LONG")}


def test_scope_is_keyed_to_surfaces_and_the_document_validates():
    doc = copy.deepcopy(EXAMPLE)
    room = doc["rooms"][0]
    wall_id = room["walls"][0]["id"]
    damage = to_schema([_region(wall_id, "water_stain", [0.5, 1.1, 0.05, 0.6]),
                        _region(f"{room['id']}.CEIL", "hole", [0.0, 0.2, 0.0, 0.2], "ceiling")])
    flags = concealed_flags(damage)
    doc["damage"], doc["concealed_flags"], doc["scope"] = damage, flags, scope_items(doc, damage, flags)
    surfaces = {w["id"] for r in doc["rooms"] for w in r["walls"]} | {f"{r['id']}.{p}" for r in doc["rooms"] for p in ("FLOOR", "CEIL")}
    assert all(item["surface_id"] in surfaces for item in doc["scope"])
    assert all(f["rule_id"] in RULE_IDS for f in flags)
    assert any(item["unit"] == "m2" and "repaint" in item["description"] for item in doc["scope"])
    assert validate_output(doc) == []


def test_tile_points_are_placed_on_the_nearest_wall():
    wall = Surface("R1.W1", "wall", "R1", np.array([0.0, 0.0]), np.array([4.0, 0.0]))
    from shapely.geometry import Polygon
    rooms = {"R1": (Polygon([(0, 0), (4, 0), (4, 3), (0, 3)]), 2.5)}
    surfaces = [wall, Surface("R1.FLOOR", "floor", "R1"), Surface("R1.CEIL", "ceiling", "R1")]
    xyz = np.column_stack([np.linspace(1.0, 1.5, 50), np.linspace(0.8, 1.2, 50), np.full(50, 0.02)])
    s, u, v = place(xyz, 0.0, surfaces, rooms)
    assert s.id == "R1.W1" and 0.9 < u.min() < 1.1 and 0.7 < v.min() < 0.9
