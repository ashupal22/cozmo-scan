"""summary.md: room extents along their own walls, and every room listed."""
import json
from pathlib import Path

import numpy as np

from cozmo.export.summary import main_dimensions, summary_markdown

EXAMPLE = json.loads((Path(__file__).resolve().parents[1] / "schema" / "example_output.json").read_text())


def test_main_dimensions_of_a_turned_rectangle():
    t = np.radians(30)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    poly = (np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float) @ R.T).tolist()
    wall = lambda v: {"length_m": {"value": v, "ci_low": v - 0.02, "ci_high": v + 0.02, "confidence": 0.9}}  # noqa: E731
    dims, intervals = main_dimensions({"polygon": poly, "walls": [wall(4), wall(3), wall(4), wall(3)]})
    assert np.allclose(dims, [3, 4], atol=1e-6)
    assert np.allclose(intervals, [[2.98, 3.02], [3.98, 4.02]], atol=1e-3)


def test_summary_lists_every_room_with_its_ranges():
    text = summary_markdown(EXAMPLE)
    assert all(room["id"] in text for room in EXAMPLE["rooms"])
    assert "footprint" in text and " to " in text
