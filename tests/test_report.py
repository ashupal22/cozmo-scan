"""report.png and report.pdf: one page, and drawn for plans with damage and with no rooms at all."""
import copy
import json
import re
from pathlib import Path

from cozmo.export.report import render_report

EXAMPLE = json.loads((Path(__file__).resolve().parents[1] / "schema" / "example_output.json").read_text())


def _pages(pdf: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))


def test_report_is_one_page_png_and_pdf(tmp_path):
    png, pdf = render_report(EXAMPLE, tmp_path)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and png.stat().st_size > 20000
    assert pdf.read_bytes()[:4] == b"%PDF" and _pages(pdf) == 1


def test_report_marks_damage_on_its_wall(tmp_path):
    doc = copy.deepcopy(EXAMPLE)
    wall = doc["rooms"][0]["walls"][0]
    m = lambda v: {"value": v, "ci_low": 0.3 * v, "ci_high": 1.3 * v, "confidence": 0.9}  # noqa: E731
    doc["damage"] = [{"id": "D1", "surface_id": wall["id"], "class": "water_stain", "area_m2": m(0.3), "width_m": m(0.6),
                      "height_m": m(0.5), "outline_uv": [[0.2, 0.1], [0.8, 0.1], [0.8, 0.6], [0.2, 0.6]],
                      "detection_confidence": 0.8, "views": 2}]
    _, pdf = render_report(doc, tmp_path)
    assert _pages(pdf) == 1


def test_report_without_rooms(tmp_path):
    doc = copy.deepcopy(EXAMPLE)
    doc["rooms"], doc["plan"]["adjacency"] = [], []
    png, _ = render_report(doc, tmp_path)
    assert png.is_file()
