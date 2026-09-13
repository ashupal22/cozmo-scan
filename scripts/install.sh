#!/usr/bin/env bash
# Clean-machine setup (macOS on Apple silicon or Linux, Python 3.10-3.12). Run from the repo root, inside a venv:
#   python3 -m venv .venv && source .venv/bin/activate && bash scripts/install.sh
# Needs ffmpeg on the PATH for videos (macOS: brew install ffmpeg).
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
# Depth Anything 3 (Apache 2.0), pinned. Its declared extras (xformers, pycolmap, open3d, evo, gsplat) are for training
# and exports we do not use, and some do not build on macOS, so it is installed without them.
python -m pip install --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/depth-anything-3@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
python scripts/fetch_models.py
command -v ffmpeg >/dev/null || echo "warning: ffmpeg not found; install it to run videos (brew install ffmpeg)"
echo "ready: cozmo run <capture>"
