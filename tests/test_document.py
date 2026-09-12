import pytest

from cozmo.export.document import build_document
from cozmo.export.render import render_svg
from cozmo.export.validate import validate_output
from cozmo.geometry.rooms import build_room_map, room_ceilings
from cozmo.geometry.walls import attach_doorways, outline_rooms
from tests.test_rooms import FLOOR, two_room_apartment


def _document():
    points, walk = two_room_apartment(ceiling_b=2.3)
    rooms = build_room_map(points, FLOOR, walk)
    outlines = outline_rooms(rooms, points, FLOOR)
    ceilings = room_ceilings(points, FLOOR, rooms)
    openings = attach_doorways(outlines, rooms)
    info = {"id": "synthetic", "tier": "lidar", "device": None, "input_path": "memory", "pipeline_version": "test"}
    document, _ = build_document(info, FLOOR, rooms, outlines, ceilings, openings, runtime_s=1.0)
    return document


def test_document_passes_the_schema_and_reference_checks():
    assert validate_output(_document()) == []


def test_true_dimensions_fall_inside_the_ranges():
    doc = _document()
    room_a, room_b = doc["rooms"]
    for wall in room_a["walls"]:
        m = wall["length_m"]
        truth = 4.0 if m["value"] > 3.5 else 3.0
        assert m["ci_low"] <= truth <= m["ci_high"]
    assert room_b["ceiling_height_m"]["ci_low"] <= 2.3 <= room_b["ceiling_height_m"]["ci_high"]
    assert room_a["ceiling_height_m"]["observed"] is False  # only room B's ceiling was seen


def test_rooms_are_connected_through_the_doorway():
    doc = _document()
    assert len(doc["plan"]["adjacency"]) == 1
    link = doc["plan"]["adjacency"][0]
    assert (link["room_a"], link["room_b"]) == ("R1", "R2")
    assert len(link["via"]) == 2  # the same doorway, seen from each room
    assert doc["rooms"][0]["openings"][0]["width_m"]["value"] == pytest.approx(0.9, abs=0.15)


def test_plan_drawing_shows_both_rooms():
    svg = render_svg(_document())
    assert svg.startswith("<svg") and "room 1" in svg and "room 2" in svg
