"""Compare regenerated benchmark results with the committed ones: the same numbers, or the first differences.

    python bench/compare_results.py <regenerated results folder> [--committed bench/results] [--pair NEW=OLD ...]

Every JSON file in the regenerated folder is compared with the committed file of the same name (or the one named by
--pair). Keys that record when or how fast a run happened (code commit, seconds, runtime) are ignored. Numbers must
agree within TOLERANCE (relative) to count as the same.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOLATILE = {"code_commit", "seconds", "runtime_s", "pipeline_version", "pipeline_runtime_s", "wall_s_with_damage"}
TOLERANCE = 1e-6


def differences(a, b, path: str = "", out: list | None = None, limit: int = 8) -> list[str]:
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in VOLATILE:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in the {'committed' if k in a else 'regenerated'} file")
            else:
                differences(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items committed, {len(b)} regenerated")
        for i, (x, y) in enumerate(zip(a, b)):
            differences(x, y, f"{path}[{i}]", out, limit)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if not math.isclose(a, b, rel_tol=TOLERANCE, abs_tol=1e-9):
            out.append(f"{path}: committed {a}, regenerated {b}")
    elif a != b:
        out.append(f"{path}: committed {a!r}, regenerated {b!r}")
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regenerated")
    ap.add_argument("--committed", default=str(ROOT / "bench" / "results"))
    ap.add_argument("--pair", action="append", default=[], help="NEW=OLD: compare regenerated NEW with committed OLD")
    ap.add_argument("--out", help="write the comparison as JSON")
    args = ap.parse_args()
    pairs = dict(p.split("=", 1) for p in args.pair)
    report = {}
    for new in sorted(Path(args.regenerated).glob("*.json")):
        old = Path(args.committed) / pairs.get(new.name, new.name)
        if not old.is_file():
            continue
        diffs = differences(json.loads(old.read_text()), json.loads(new.read_text()))
        report[new.name] = {"committed": str(old.relative_to(Path(args.committed))), "same": not diffs, "differences": diffs}
        print(f"{new.name:45s} {'same' if not diffs else 'DIFFERS'}")
        for d in diffs:
            print(f"    {d}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
