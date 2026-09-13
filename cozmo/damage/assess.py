"""Fill the output contract's damage, concealed-damage flags and scope for one plan."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from cozmo.damage.detect import Surface, detect, lidar_views, model_views, scene_conditions, surfaces_of, to_schema
from cozmo.damage.rules import concealed_flags
from cozmo.damage.scope import scope_items


def _to_plan(correction):
    if correction is None:
        return None
    from cozmo.geometry.fusion import PointSet

    def to_plan(xyz, frame):
        n = len(xyz)
        ps = PointSet(xyz.astype(np.float32), np.zeros((n, 3), np.float32), np.full(n, frame, np.int32),
                      np.zeros(n, np.float32))
        return correction.apply(ps).xyz.astype(float)
    return to_plan


def assess(plan, tier: str, work_dir: Path) -> None:
    """Detect damage on the plan's surfaces, then fire the concealed-damage rules and write the scope."""
    document = plan.document
    regions, conditions = [], []
    if tier in ("lidar", "video"):
        surfaces, rooms = surfaces_of(document, plan.outlines)
        views = lidar_views(plan.capture, Path(work_dir) / "damage_frames") if tier == "lidar" \
            else model_views(plan.capture, every=3)
        regions = detect(views, plan.floor.height, surfaces, rooms, _to_plan(plan.correction))
        conditions = scene_conditions(views)
    else:
        from cozmo.ingest.photos import load_photo
        for room, r in zip(document["rooms"], plan.photo_rooms):
            name = room["id"]
            surfaces = [Surface(f"{name}.W{k + 1}", "wall", name, np.asarray(w.start, float), np.asarray(w.end, float))
                        for k, w in enumerate(r.outline.walls)]
            surfaces += [Surface(f"{name}.FLOOR", "floor", name, yaw_deg=r.outline.yaw_deg),
                         Surface(f"{name}.CEIL", "ceiling", name, yaw_deg=r.outline.yaw_deg)]
            rooms = {name: (Polygon(r.outline.vertices), room["ceiling_height_m"]["value"])}
            views = model_views(r.capture, loader=lambda f: load_photo(f)[0])
            regions += detect(views, r.floor.height, surfaces, rooms, None, min_views=1)
            conditions += [f"{r.name}: {w}" for w in scene_conditions(views)]
    damage = to_schema(regions)
    flags = concealed_flags(damage)
    document["damage"], document["concealed_flags"] = damage, flags
    document["scope"] = scope_items(document, damage, flags)
    warnings = document["quality"]["warnings"]
    warnings[:] = [w for w in warnings if not w.startswith("damage detection, concealed-damage rules and scope")]
    warnings.append(f"damage: {len(damage)} region(s) from a zero-shot image model (CLIP), not yet tested on staged "
                    f"damage; check each against its detection confidence. {len(flags)} concealed-damage flag(s), "
                    f"{len(document['scope'])} scope item(s)")
    warnings += [f"conditions: {w}" for w in conditions]
