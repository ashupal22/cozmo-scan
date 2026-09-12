"""Command line entry point: `cozmo <command>`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cozmo import __version__
from cozmo.export.validate import validate_output
from cozmo.ingest.detect import VIDEO_EXT, TierError, detect_tier
from cozmo.ingest.photos import photo_rooms, photo_warnings
from cozmo.ingest.stray import CaptureError, StrayCapture


def cmd_inspect(args) -> int:
    tier = detect_tier(args.path)
    if tier == "lidar":
        summary = StrayCapture(args.path).summary()
    elif tier == "photo":
        rooms = photo_rooms(args.path)
        summary = {"tier": "photo", "path": str(args.path),
                   "rooms": {name: len(photos) for name, photos in rooms.items()},
                   "warnings": photo_warnings(rooms)}
    else:
        p = Path(args.path)
        video = p if p.is_file() else next(c for c in sorted(p.iterdir()) if c.suffix.lower() in VIDEO_EXT)
        summary = {"tier": "video", "path": str(video), "size_mb": round(video.stat().st_size / 1e6, 1),
                   "warnings": []}

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    print(args.path)
    for key, value in summary.items():
        if key != "warnings":
            print(f"  {key:26s} {value}")
    for warning in summary["warnings"]:
        print(f"  warning: {warning}")
    return 0


def cmd_validate(args) -> int:
    try:
        doc = json.loads(Path(args.file).read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: cannot read {args.file}: {e}", file=sys.stderr)
        return 2
    problems = validate_output(doc)
    if problems:
        print(f"{args.file}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{args.file}: valid")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cozmo", description="Measured floor plans from iPhone captures.")
    parser.add_argument("--version", action="version", version=f"cozmo-scan {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="detect a capture's tier and print a summary")
    p.add_argument("path", help="LiDAR export folder, video file, or folder of room photo folders")
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("validate", help="check an output JSON file against the schema")
    p.add_argument("file")
    p.set_defaults(func=cmd_validate)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (TierError, CaptureError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
