"""Check an output document: JSON schema, interval order, and internal references."""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "output.schema.json"


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


def validate_output(doc: dict) -> list[str]:
    """Return a list of problems; an empty list means the document is valid."""
    problems = []
    for err in sorted(_validator().iter_errors(doc), key=lambda e: [str(p) for p in e.absolute_path]):
        where = "/".join(str(p) for p in err.absolute_path) or "(root)"
        problems.append(f"schema: {where}: {err.message}")
    if problems:
        return problems  # the checks below assume the structure is valid
    return _interval_problems(doc) + _reference_problems(doc)


def _interval_problems(node, parts=()) -> list[str]:
    problems = []
    if isinstance(node, dict):
        if {"value", "ci_low", "ci_high"} <= node.keys() and not node["ci_low"] <= node["value"] <= node["ci_high"]:
            problems.append(f"interval out of order at {'/'.join(parts)}: "
                            f"ci_low={node['ci_low']} value={node['value']} ci_high={node['ci_high']}")
        for key, child in node.items():
            problems += _interval_problems(child, parts + (str(key),))
    elif isinstance(node, list):
        for i, child in enumerate(node):
            problems += _interval_problems(child, parts + (str(i),))
    return problems


def _duplicates(ids) -> list[str]:
    return [f"duplicate id {x}" for x, n in Counter(ids).items() if n > 1]


def _reference_problems(doc: dict) -> list[str]:
    problems = []
    room_ids = [r["id"] for r in doc["rooms"]]
    problems += _duplicates(room_ids)

    surfaces, openings = set(), set()
    for room in doc["rooms"]:
        rid = room["id"]
        walls = [w["id"] for w in room["walls"]]
        problems += _duplicates(walls)
        problems += [f"wall {w} is listed under room {rid}" for w in walls if not w.startswith(rid + ".")]
        surfaces.update(walls)
        surfaces.update({f"{rid}.FLOOR", f"{rid}.CEIL"})
        for o in room["openings"]:
            if not o["id"].startswith(rid + "."):
                problems.append(f"opening {o['id']} is listed under room {rid}")
            if o["wall_id"] not in walls:
                problems.append(f"opening {o['id']} is on wall {o['wall_id']}, which room {rid} does not have")
            if o["connects_to"] not in (None, "exterior") and o["connects_to"] not in room_ids:
                problems.append(f"opening {o['id']} connects to unknown room {o['connects_to']}")
            openings.add(o["id"])
        problems += _duplicates([o["id"] for o in room["openings"]])

    for link in doc["plan"]["adjacency"]:
        problems += [f"adjacency names unknown room {link[k]}" for k in ("room_a", "room_b") if link[k] not in room_ids]
        problems += [f"adjacency uses unknown opening {o}" for o in link["via"] if o not in openings]

    damage_ids = [d["id"] for d in doc["damage"]]
    problems += _duplicates(damage_ids)
    problems += [f"damage {d['id']} is on unknown surface {d['surface_id']}"
                 for d in doc["damage"] if d["surface_id"] not in surfaces]

    problems += _duplicates([f["id"] for f in doc["concealed_flags"]])
    for flag in doc["concealed_flags"]:
        if flag["surface_id"] not in surfaces:
            problems.append(f"flag {flag['id']} is on unknown surface {flag['surface_id']}")
        problems += [f"flag {flag['id']} cites unknown damage {e}" for e in flag["evidence"] if e not in damage_ids]

    problems += _duplicates([s["id"] for s in doc["scope"]])
    for item in doc["scope"]:
        if item["surface_id"] not in surfaces:
            problems.append(f"scope item {item['id']} is on unknown surface {item['surface_id']}")
        problems += [f"scope item {item['id']} cites unknown damage {d}"
                     for d in item["because"].get("damage_ids", []) if d not in damage_ids]
    return problems
