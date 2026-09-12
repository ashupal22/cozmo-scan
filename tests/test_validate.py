import copy
import json

from cozmo.export.validate import validate_output
from tests.conftest import REPO

EXAMPLE = json.loads((REPO / "schema" / "example_output.json").read_text())


def _changed(edit):
    doc = copy.deepcopy(EXAMPLE)
    edit(doc)
    return validate_output(doc)


def test_example_is_valid():
    assert validate_output(EXAMPLE) == []


def test_schema_errors_are_reported_with_location():
    problems = _changed(lambda d: d["capture"].__setitem__("tier", "drone"))
    assert problems and problems[0].startswith("schema: capture/tier")


def test_interval_out_of_order():
    problems = _changed(lambda d: d["rooms"][0]["walls"][0]["length_m"].__setitem__("ci_low", 5.0))
    assert problems == ["interval out of order at rooms/0/walls/0/length_m: ci_low=5.0 value=4.2 ci_high=4.211"]


def test_damage_on_unknown_surface():
    problems = _changed(lambda d: d["damage"][0].__setitem__("surface_id", "R9.W1"))
    assert any("unknown surface R9.W1" in p for p in problems)


def test_opening_on_a_wall_the_room_does_not_have():
    problems = _changed(lambda d: d["rooms"][1]["openings"][0].__setitem__("wall_id", "R2.W9"))
    assert any("R2.W9" in p for p in problems)


def test_flag_citing_unknown_damage():
    problems = _changed(lambda d: d["concealed_flags"][0].__setitem__("evidence", ["D7"]))
    assert any("unknown damage D7" in p for p in problems)
