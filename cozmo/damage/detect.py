"""Damage regions on the plan's surfaces (output contract: `damage`).

Detection is zero-shot. CLIP ViT-B/32 (OpenAI, MIT licence) scores tiles of the capture's images against text
prompts for five damage classes, and against the undamaged things a tile usually shows: plain wall, ceiling,
floor, furniture, door, window, picture, shadow, light. A tile counts when the damage prompts together take at
least P_MIN of the probability. It has not been tested on staged damage (we had no damaged room to capture), so
every region keeps its detection confidence and a wide extent interval.

Location and size: the tile's depth pixels are placed in the plan with the capture's (drift-corrected) poses and
assigned to the nearest surface: a wall of a room outline, the floor or the ceiling. They are measured on that
surface's plane: width along it, height (walls) or depth (floor, ceiling), area, and height above the floor.
Tiles of one class on one surface that overlap are merged into one region, and `views` counts the images that
saw it. A region needs MIN_VIEWS images, except at the photo tier, where each room has only a few.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

MODEL = os.environ.get("COZMO_CLIP_MODEL", "openai/clip-vit-base-patch32")
CLASSES = {
    "water_stain": ["a brown water stain on a ceiling", "a yellow water stain on a wall", "water damage on a ceiling"],
    "mold": ["black mold on a wall", "mold spots on a ceiling"],
    "crack": ["a crack in a plaster wall", "a cracked ceiling"],
    "peeling_paint": ["peeling paint on a wall", "flaking paint on a ceiling"],
    "hole": ["a hole in a drywall wall", "a broken hole in a wall"],
}
NEGATIVES = ["a clean painted wall", "a plain white ceiling", "a wooden floor", "a tiled floor", "a carpet",
             "furniture", "a sofa", "a bed", "a kitchen cabinet", "a door", "a window", "a picture on a wall",
             "a shadow on a wall", "a ceiling light", "a curtain", "a shelf with objects", "a person", "a wall corner",
             "a mirror", "a radiator", "a light switch on a wall", "a textured wall"]
P_MIN = 0.6
MIN_VIEWS = 2
MIN_TILE_POINTS = 30
WALL_REACH_M = 0.4
CEILING_BAND_M = 0.25
FLOOR_BAND_M = 0.15
MERGE_GAP_M = 0.1


@dataclass
class View:
    image: np.ndarray          # RGB uint8, same orientation as the depth map
    depth: np.ndarray          # (h, w) metres, 0 = no depth
    K: tuple                   # fx, fy, cx, cy at depth resolution
    R: np.ndarray              # camera (OpenCV) to world
    t: np.ndarray              # camera centre, world
    frame: int
    upright: int | None = None  # cv2 rotate code that turns the image upright, for classification only


@dataclass
class Surface:
    id: str                    # R1.W2, R1.FLOOR, R1.CEIL
    kind: str                  # wall, floor, ceiling
    room: str
    start: np.ndarray | None = None   # walls: plan (x, z) of the wall's start and end
    end: np.ndarray | None = None
    yaw_deg: float = 0.0              # floor and ceiling: room axes, for width and depth


@lru_cache(maxsize=1)
def _clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from cozmo.video.da3 import device
    model = CLIPModel.from_pretrained(MODEL).to(device()).eval()
    processor = CLIPProcessor.from_pretrained(MODEL)
    prompts = [f"a photo of {p}." for ps in CLASSES.values() for p in ps] + [f"a photo of {p}." for p in NEGATIVES]
    with torch.no_grad():
        tokens = processor(text=prompts, return_tensors="pt", padding=True).to(device())
        text = model.get_text_features(**tokens)
        text = text / text.norm(dim=-1, keepdim=True)
    owner = [c for c, ps in CLASSES.items() for _ in ps] + [None] * len(NEGATIVES)
    return model, processor, text, owner


def tiles(shape: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    """Tile boxes as fractions of the image (x0, y0, x1, y1): 4 x 3 on landscape images, 3 x 4 on portrait."""
    h, w = shape
    cols, rows = (4, 3) if w >= h else (3, 4)
    return [(c / cols, r / rows, (c + 1) / cols, (r + 1) / rows) for r in range(rows) for c in range(cols)]


def classify(views: list[View], batch: int = 64) -> list[tuple[int, tuple, str, float]]:
    """(view index, tile box, class, probability) for every tile a damage class wins."""
    import torch
    from PIL import Image
    model, processor, text, owner = _clip()
    crops, where = [], []
    for k, view in enumerate(views):
        h, w = view.image.shape[:2]
        for box in tiles((h, w)):
            crop = view.image[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]
            if view.upright is not None:
                crop = cv2.rotate(crop, view.upright)
            crops.append(Image.fromarray(np.ascontiguousarray(crop)))
            where.append((k, box))
    hits = []
    from cozmo.video.da3 import device
    for s in range(0, len(crops), batch):
        with torch.no_grad():
            pixels = processor(images=crops[s:s + batch], return_tensors="pt").to(device())
            img = model.get_image_features(**pixels)
            img = img / img.norm(dim=-1, keepdim=True)
            prob = (100.0 * img @ text.T).softmax(dim=-1).cpu().numpy()
        for row, p in zip(where[s:s + batch], prob):
            per_class = {c: float(sum(p[j] for j, o in enumerate(owner) if o == c)) for c in CLASSES}
            best = max(per_class, key=per_class.get)
            if per_class[best] >= P_MIN:
                hits.append((row[0], row[1], best, per_class[best]))
    return hits


def tile_points(view: View, box: tuple) -> np.ndarray:
    """World points of the depth pixels inside a tile."""
    h, w = view.depth.shape
    x0, y0, x1, y1 = int(box[0] * w), int(box[1] * h), int(np.ceil(box[2] * w)), int(np.ceil(box[3] * h))
    z = view.depth[y0:y1, x0:x1]
    v, u = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    ok = (z > 0.2) & (z < 8.0)
    fx, fy, cx, cy = view.K
    P = np.stack([(u[ok] - cx) / fx * z[ok], (v[ok] - cy) / fy * z[ok], z[ok]], axis=-1)
    return P @ view.R.T + view.t


def surfaces_of(document: dict, outlines: dict) -> tuple[list[Surface], dict]:
    """Surfaces of the document's rooms, and room id -> (outline polygon, ceiling height above floor)."""
    surfaces, rooms = [], {}
    for room, (rid, outline) in zip(document["rooms"], sorted(outlines.items())):
        name = room["id"]
        rooms[name] = (Polygon(outline.vertices), room["ceiling_height_m"]["value"])
        for k, wall in enumerate(outline.walls):
            surfaces.append(Surface(f"{name}.W{k + 1}", "wall", name, np.asarray(wall.start, float), np.asarray(wall.end, float)))
        surfaces.append(Surface(f"{name}.FLOOR", "floor", name, yaw_deg=outline.yaw_deg))
        surfaces.append(Surface(f"{name}.CEIL", "ceiling", name, yaw_deg=outline.yaw_deg))
    return surfaces, rooms


def _wall_distance(xz: np.ndarray, s: Surface) -> float:
    d = s.end - s.start
    t = np.clip(np.dot(xz - s.start, d) / max(np.dot(d, d), 1e-9), 0, 1)
    return float(np.linalg.norm(xz - (s.start + t * d)))


def place(xyz: np.ndarray, floor_y: float, surfaces: list[Surface], rooms: dict):
    """(surface, u, v) for a tile's points: u along the surface, v up the wall or across the floor or ceiling."""
    height = xyz[:, 1] - floor_y
    xz = xyz[:, [0, 2]].astype(float)
    centre = np.median(xz, axis=0)
    inside = [name for name, (poly, _) in rooms.items() if poly.buffer(0.2).contains(Point(*centre))]
    h = float(np.median(height))
    for name in inside:
        ceiling = rooms[name][1]
        kind = "ceiling" if h > ceiling - CEILING_BAND_M else "floor" if h < FLOOR_BAND_M else None
        if kind:
            s = next(s for s in surfaces if s.room == name and s.kind == kind)
            t = np.radians(s.yaw_deg)
            uv = xz @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            return s, uv[:, 0], uv[:, 1]
    walls = [s for s in surfaces if s.kind == "wall" and (not inside or s.room in inside)]
    if not walls:
        return None
    s = min(walls, key=lambda s: _wall_distance(centre, s))
    if _wall_distance(centre, s) > WALL_REACH_M:
        return None
    d = (s.end - s.start) / max(np.linalg.norm(s.end - s.start), 1e-9)
    return s, (xz - s.start) @ d, height


@dataclass
class Region:
    surface: Surface
    cls: str
    box: list                  # u0, u1, v0, v1 on the surface plane
    prob: float
    frames: set


def detect(views: list[View], floor_y: float, surfaces: list[Surface], rooms: dict, to_plan=None,
           min_views: int = MIN_VIEWS) -> list[Region]:
    """Damage regions on the given surfaces. `to_plan(xyz, frame)` maps camera-world points into the plan frame."""
    regions: list[Region] = []
    for k, box, cls, prob in classify(views):
        xyz = tile_points(views[k], box)
        if len(xyz) < MIN_TILE_POINTS:
            continue
        if to_plan is not None:
            xyz = to_plan(xyz, views[k].frame)
        placed = place(xyz, floor_y, surfaces, rooms)
        if placed is None:
            continue
        s, u, v = placed
        b = [float(np.percentile(u, 10)), float(np.percentile(u, 90)), float(np.percentile(v, 10)), float(np.percentile(v, 90))]
        for r in regions:
            if r.surface.id == s.id and r.cls == cls and b[0] < r.box[1] + MERGE_GAP_M and r.box[0] < b[1] + MERGE_GAP_M \
                    and b[2] < r.box[3] + MERGE_GAP_M and r.box[2] < b[3] + MERGE_GAP_M:
                r.box = [min(r.box[0], b[0]), max(r.box[1], b[1]), min(r.box[2], b[2]), max(r.box[3], b[3])]
                r.prob = max(r.prob, prob)
                r.frames.add(views[k].frame)
                break
        else:
            regions.append(Region(s, cls, b, prob, {views[k].frame}))
    return [r for r in regions if len(r.frames) >= min_views]


def to_schema(regions: list[Region], start: int = 1) -> list[dict]:
    """Damage regions for the JSON. The extent is the tiles' footprint: the damage itself may be smaller, so the
    interval runs from 30% to 130% of it."""
    def extent(value: float) -> dict:
        return {"value": round(value, 3), "ci_low": round(0.3 * value, 3), "ci_high": round(1.3 * value, 3),
                "confidence": 0.9, "method": "footprint of the image tiles that show it, on the surface plane"}
    out = []
    for n, r in enumerate(regions, start):
        u0, u1, v0, v1 = r.box
        width, height = max(u1 - u0, 0.05), max(v1 - v0, 0.05)
        item = {"id": f"D{n}", "surface_id": r.surface.id, "class": r.cls, "area_m2": extent(width * height),
                "width_m": extent(width), "height_m": extent(height),
                "outline_uv": [[round(u0, 3), round(v0, 3)], [round(u1, 3), round(v0, 3)], [round(u1, 3), round(v1, 3)],
                               [round(u0, 3), round(v1, 3)]],
                "detection_confidence": round(r.prob, 3), "views": len(r.frames)}
        if r.surface.kind == "wall":
            item["bottom_above_floor_m"] = {"value": round(max(v0, 0.0), 3), "ci_low": round(max(v0 - 0.15, 0.0), 3),
                                            "ci_high": round(max(v0, 0.0) + 0.15, 3), "confidence": 0.9}
        out.append(item)
    return out


# ---- views of each tier ---------------------------------------------------------------------

def lidar_views(capture, frames_dir, every_s: float = 1.0) -> list[View]:
    """One view per second of a Stray Scanner walk: the video frame and the LiDAR depth of the nearest pose."""
    import subprocess
    from pathlib import Path
    from cozmo.ingest.stray import upright_rotate_code
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(frames_dir.glob("*.jpg"))
    if not files:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(Path(capture.path) / "rgb.mp4"), "-vf",
                        f"fps={1 / every_s},scale=640:-2", "-q:v", "3", str(frames_dir / "%05d.jpg")], check=True)
        files = sorted(frames_dir.glob("*.jpg"))
    t = capture.timestamps - capture.timestamps[0]
    views = []
    for k, f in enumerate(files):
        i = int(np.clip(np.searchsorted(t, k * every_s), 0, len(capture) - 1))
        k_depth = capture.intrinsics(i, "depth")
        image = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        views.append(View(image, capture.depth(i), (k_depth.fx, k_depth.fy, k_depth.cx, k_depth.cy),
                          capture.rotation(i), capture.positions[i], i, upright_rotate_code(capture.rotation(i))))
    return views


def model_views(capture, every: int = 1, loader=None) -> list[View]:
    """Views of a video or photo capture: its own images with their model depth."""
    views = []
    for i in range(0, len(capture), every):
        k = capture.intrinsics(i)
        if loader is not None:
            image = loader(capture.frame_files[i])
        else:
            image = cv2.cvtColor(cv2.imread(str(capture.frame_files[i])), cv2.COLOR_BGR2RGB)
        views.append(View(image, capture.depth(i), (k.fx, k.fy, k.cx, k.cy), capture.rotation(i), capture.positions[i], i))
    return views


# ---- scene conditions the brief names: mirrors, glass, wet-look surfaces, low light ------------------------------

CONDITIONS = {
    "mirror": ["a large mirror on a wall", "a mirrored wardrobe door"],
    "glass": ["a glass door", "a large glass window", "a glass shower screen"],
    "wet-look floor": ["a shiny wet floor", "a glossy reflective floor"],
}
CONDITION_NEGATIVES = ["a plain painted wall", "a room with furniture", "a wooden floor", "a carpet", "a door",
                       "a kitchen", "a ceiling", "a corridor", "a bed", "a sofa"]
CONDITION_P_MIN = 0.5
DARK_MEAN = 0.18            # mean brightness (0-1) below which an image counts as dark
DARK_SHARE = 0.3


def low_light(views: list[View]) -> str | None:
    """A warning when at least DARK_SHARE of the images are dark, else None."""
    brightness = np.array([float(v.image.mean()) / 255.0 for v in views])
    dark = brightness < DARK_MEAN
    if not len(views) or dark.mean() < DARK_SHARE:
        return None
    return (f"low light: {int(dark.sum())} of {len(views)} images are dark (mean brightness {brightness.mean():.2f}); "
            f"depth from images and damage detection are less reliable there. Switch on every light and capture again "
            f"if possible")


def scene_conditions(views: list[View]) -> list[str]:
    """Warnings for conditions that make depth or damage detection unreliable, with how many images show them."""
    import torch
    from PIL import Image
    from cozmo.video.da3 import device
    if not views:
        return []
    low = low_light(views)
    warnings = [low] if low else []
    model, processor, _, _ = _clip()
    prompts = [f"a photo of {p}." for ps in CONDITIONS.values() for p in ps] + [f"a photo of {p}." for p in CONDITION_NEGATIVES]
    owner = [c for c, ps in CONDITIONS.items() for _ in ps] + [None] * len(CONDITION_NEGATIVES)
    seen = {c: [] for c in CONDITIONS}
    with torch.no_grad():
        tokens = processor(text=prompts, return_tensors="pt", padding=True).to(device())
        text = model.get_text_features(**tokens)
        text = text / text.norm(dim=-1, keepdim=True)
        for s in range(0, len(views), 32):
            batch = views[s:s + 32]
            images = [Image.fromarray(np.ascontiguousarray(cv2.rotate(v.image, v.upright) if v.upright is not None else v.image))
                      for v in batch]
            img = model.get_image_features(**processor(images=images, return_tensors="pt").to(device()))
            img = img / img.norm(dim=-1, keepdim=True)
            prob = (100.0 * img @ text.T).softmax(dim=-1).cpu().numpy()
            for v, p in zip(batch, prob):
                for c in CONDITIONS:
                    share = float(sum(p[j] for j, o in enumerate(owner) if o == c))
                    if share >= CONDITION_P_MIN:
                        seen[c].append(v.frame)
    advice = {"mirror": "depth may show a room behind it that is not there; check walls near it",
              "glass": "depth may pass through or reflect; walls and openings near it are less reliable",
              "wet-look floor": "reflections can look like stains, and depth may drop out on it"}
    for c, frames in seen.items():
        if frames:
            warnings.append(f"{c} seen in {len(frames)} of {len(views)} images (first at frame {frames[0]}): {advice[c]}")
    return warnings
