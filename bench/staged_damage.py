"""Staged damage synthesised on a real capture (A-DMG-DETECT, A-DMG-AREA): real defect photos placed on real walls at a
known size and height, then found by the shipped damage path.

No damaged room was available to capture. Everything here is real except the damage: the images, LiDAR depth, poses and
room plan of walk c7d28f72c6 (cozmo run, LiDAR tier, drift correction on). Each staged defect is a BD3 photo of a water
stain or a crack (bench/damage_bd3.py). It is mapped onto a wall of the plan at a set size and height above the floor,
and blended into every frame that sees it, darkening the wall where the photo is dark. The LiDAR depth decides what is
in front of it, so furniture hides it. The walls are the ones most frames see.
The photos are the ones the detector classifies best as whole photos. That separates finding and measuring damage in
a room from recognising the texture, which bench/damage_bd3.py measures.

The shipped path then runs unchanged: tiles, CLIP, depth points on surfaces, regions, rules, scope. Scored per staged
defect: found on the right wall with the right class; overlap (IoU) with the staged rectangle on the wall plane; width,
height and area against the staged size; the concealed-damage rule it should fire. The same frames unpainted are run
as a control for false regions.

    COZMO_DATA=/path/to/captures python bench/staged_damage.py
"""
from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from video_vs_lidar import DATA, DERIVED, code_commit  # noqa: E402

import cozmo.damage.detect as dd  # noqa: E402
from cozmo.damage.assess import _to_plan  # noqa: E402
from cozmo.damage.rules import concealed_flags  # noqa: E402
from cozmo.damage.scope import scope_items  # noqa: E402
from cozmo.pipeline import run_lidar  # noqa: E402

OUT = ROOT / "bench" / "results" / "staged_damage.json"
WALK = "c7d28f72c6"
REPO, FILE = "chandrabhuma/building_defect_vqa", "data/test-00000-of-00001.parquet"
# BD3 label, our class, width, height, bottom above floor (m), the rule it should fire
STAGED = [("stain", "water_stain", 0.60, 0.45, 0.10, "R-WALL-BASE-WATER"),
          ("major_crack", "crack", 0.35, 1.20, 0.45, "R-CRACK-LONG")]
OFF_WALL_M = 0.01
FEATHER = 0.12
MIN_PIXELS = 1500
RANGE_M = (0.5, 4.0)
SURFACE_M = 0.10               # the sensor must see a surface this close to the decal: it goes on a wall, never in a doorway
TINTS = {"water_stain": (0.55, 0.70, 0.95), "crack": (0.85, 0.85, 0.85)}   # darkening per R, G, B: brown stain, grey crack
STRENGTH = 0.8                 # the photo's darkest parts darken the wall by 80% of the tint: clearly visible, as staged damage is


def pick_textures() -> dict:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from PIL import Image
    table = pd.read_parquet(hf_hub_download(REPO, FILE, repo_type="dataset"))
    shipped, dd.P_MIN = dd.P_MIN, 0.0
    textures = {}
    for label, cls, *_ in STAGED:
        rows = table[table.answer == label].head(40)
        images = [np.asarray(Image.open(io.BytesIO(r.image["bytes"])).convert("RGB")) for r in rows.itertuples()]
        views = [dd.View(im, np.zeros((4, 4)), (1, 1, 1, 1), np.eye(3), np.zeros(3), k) for k, im in enumerate(images)]
        score = np.zeros(len(images))
        for k, _, c, p in dd.classify(views):
            if c == cls:
                score[k] += p
        best = int(np.argmax(score))
        textures[label] = {"image": images[best], "bd3_row": int(rows.index[best]), "tile_score": round(float(score[best]), 2)}
    dd.P_MIN = shipped
    return textures


def factor_map(image: np.ndarray, cls: str) -> np.ndarray:
    """Multiplicative shading from a defect photo: its bright background -> 1 (wall unchanged), the defect darker in
    the class's tint (the photo's own colours would tint white walls yellow)."""
    lum = image.astype(np.float32).mean(axis=2)
    lo, hi = float(np.percentile(lum, 5)), float(np.percentile(lum, 85))
    dark = np.clip((hi - lum) / max(hi - lo, 1.0), 0.0, 1.0) * STRENGTH      # contrast stretched: a faint stain photo
    return np.clip(1.0 - dark[..., None] * np.array(TINTS[cls], np.float32), 0.0, 1.0)


def plan_camera(capture, correction, i: int):
    """Camera (OpenCV) to plan rotation, and camera centre in the plan, for frame i."""
    R, t = capture.rotation(i), capture.positions[i]
    if correction is None:
        return R, t
    c, s = np.cos(correction.theta[i]), np.sin(correction.theta[i])
    Y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return Y @ R, correction.positions[i]


def decal_on_wall(outline, k: int, u_center: float, width: float, height: float, bottom: float) -> dict:
    wall = outline.walls[k]
    s, e = np.asarray(wall.start, float), np.asarray(wall.end, float)
    d = (e - s) / np.linalg.norm(e - s)
    n = np.array([-d[1], d[0]])
    if (np.asarray(outline.vertices, float).mean(axis=0) - s) @ n < 0:
        n = -n
    return {"start": s, "dir": d, "normal": n, "u0": u_center - width / 2, "u1": u_center + width / 2,
            "v0": bottom, "v1": bottom + height}


def seen_from(decal: dict, view, capture, correction, floor_y: float) -> bool:
    gu, gv = np.meshgrid(np.linspace(decal["u0"], decal["u1"], 5), np.linspace(decal["v0"], decal["v1"], 5))
    xz = decal["start"] + gu.ravel()[:, None] * decal["dir"] + OFF_WALL_M * decal["normal"]
    pts = np.column_stack([xz[:, 0], floor_y + gv.ravel(), xz[:, 1]])
    R, t = plan_camera(capture, correction, view.frame)
    if (t[[0, 2]] - decal["start"]) @ decal["normal"] <= 0.2:
        return False                                   # camera behind the wall
    pc = (pts - t) @ R
    z = pc[:, 2]
    if (z < RANGE_M[0]).any() or (z > RANGE_M[1]).any():
        return False
    fx, fy, cx, cy = view.K
    u, v = fx * pc[:, 0] / z + cx, fy * pc[:, 1] / z + cy
    hd, wd = view.depth.shape
    if not ((u >= 0) & (u < wd) & (v >= 0) & (v < hd)).all():
        return False
    sensor = view.depth[v.astype(int), u.astype(int)]
    return bool(((sensor > 0) & (np.abs(sensor - z) <= SURFACE_M)).mean() >= 0.8)


def paint(view, decal: dict, factor: np.ndarray, capture, correction, floor_y: float):
    """The view with the decal blended in, and the number of visible decal pixels (None when too few)."""
    R, t = plan_camera(capture, correction, view.frame)
    h, w = view.image.shape[:2]
    k = capture.intrinsics(view.frame, "rgb")
    sx, sy = w / capture.rgb_size[0], h / capture.rgb_size[1]
    fx, fy, cx, cy = k.fx * sx, k.fy * sy, k.cx * sx, k.cy * sy
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
    n3 = np.array([decal["normal"][0], 0.0, decal["normal"][1]])
    p0 = np.array([decal["start"][0], 0.0, decal["start"][1]]) + OFF_WALL_M * n3
    n_c, p0_c = R.T @ n3, R.T @ (p0 - t)
    denom = rays @ n_c
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(np.abs(denom) > 1e-6, (p0_c @ n_c) / denom, np.nan)
    X = t + (rays * z[..., None]) @ R.T
    along = (X[..., [0, 2]] - decal["start"]) @ decal["dir"]
    tx = (along - decal["u0"]) / (decal["u1"] - decal["u0"])
    ty = (decal["v1"] - (X[..., 1] - floor_y)) / (decal["v1"] - decal["v0"])
    inside = (z > 0.3) & (tx >= 0) & (tx <= 1) & (ty >= 0) & (ty <= 1)
    depth = cv2.resize(view.depth, (w, h), interpolation=cv2.INTER_NEAREST)
    visible = inside & (depth > 0) & (np.abs(depth - z) <= SURFACE_M)
    if visible.sum() < MIN_PIXELS:
        return None
    th, tw = factor.shape[:2]
    mx = np.nan_to_num(np.clip(tx, 0, 1) * (tw - 1)).astype(np.float32)
    my = np.nan_to_num(np.clip(ty, 0, 1) * (th - 1)).astype(np.float32)
    f = cv2.remap(factor, mx, my, cv2.INTER_LINEAR)
    edge = np.nan_to_num(np.clip(np.minimum.reduce([tx, 1 - tx, ty, 1 - ty]) / FEATHER, 0, 1))
    alpha = np.where(visible, edge, 0.0)[..., None].astype(np.float32)
    out = view.image.astype(np.float32) * (1 - alpha + alpha * f)
    ys, xs = np.nonzero(visible)
    bbox = (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)
    return np.clip(out, 0, 255).astype(np.uint8), int(visible.sum()), bbox


def iou(a: list, b: list) -> float:
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def main():
    plan = run_lidar(DATA / WALK)
    capture, correction, floor_y = plan.capture, plan.correction, plan.floor.height
    views = dd.lidar_views(capture, DERIVED / f"{WALK}_damage_frames")
    surfaces, rooms = dd.surfaces_of(plan.document, plan.outlines)
    to_plan = _to_plan(correction)
    textures = pick_textures()

    # the wall seen by the most frames, for each staged defect (different walls)
    staged, used = [], set()
    for label, cls, width, height, bottom, rule in STAGED:
        best = None
        for rid, outline in sorted(plan.outlines.items()):
            for k, wall in enumerate(outline.walls):
                if wall.support == 0 or wall.length_m < width + 0.4 or (rid, k) in used:
                    continue
                for share in (0.3, 0.5, 0.7):
                    decal = decal_on_wall(outline, k, share * wall.length_m, width, height, bottom)
                    if decal["u0"] < 0.2 or decal["u1"] > wall.length_m - 0.2:
                        continue
                    n = sum(seen_from(decal, v, capture, correction, floor_y) for v in views)
                    if best is None or n > best[0]:
                        best = (n, rid, k, decal)
        n, rid, k, decal = best
        used.add((rid, k))
        staged.append({"label": label, "class": cls, "rule": rule, "surface_id": f"R{rid}.W{k + 1}", "decal": decal,
                       "box": [decal["u0"], decal["u1"], decal["v0"], decal["v1"]], "frames_seeing_fully": n})

    painted, pixels, boxes = [], {s["surface_id"]: [] for s in staged}, {s["surface_id"]: {} for s in staged}
    factors = {s["surface_id"]: factor_map(textures[s["label"]]["image"], s["class"]) for s in staged}
    for view in views:
        image = view.image
        for s in staged:
            got = paint(dd.View(image, view.depth, view.K, view.R, view.t, view.frame, view.upright), s["decal"],
                        factors[s["surface_id"]], capture, correction, floor_y)
            if got:
                image, count, bbox = got
                pixels[s["surface_id"]].append(view.frame)
                boxes[s["surface_id"]][view.frame] = bbox
        painted.append(dd.View(image, view.depth, view.K, view.R, view.t, view.frame, view.upright))

    document = copy.deepcopy(plan.document)
    control = dd.to_schema(dd.detect(views, floor_y, surfaces, rooms, to_plan))
    damage = dd.to_schema(dd.detect(painted, floor_y, surfaces, rooms, to_plan))
    flags = concealed_flags(damage)
    scope = scope_items(document, damage, flags)
    shipped, dd.P_MIN = dd.P_MIN, 0.0             # every tile's best damage class and score, for the diagnosis
    scores = {"painted": dd.classify(painted), "unpainted": dd.classify(views)}
    dd.P_MIN = shipped
    index = {v.frame: k for k, v in enumerate(views)}

    def tile_scores(surface_id: str) -> dict:
        out = {}
        for name, hits in scores.items():
            best = (0.0, None)
            for k, box, cls, prob in hits:
                for frame, b in boxes[surface_id].items():
                    if index[frame] == k and box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]:
                        best = max(best, (prob, cls), key=lambda x: x[0])
            out[name] = {"max_damage_score": round(best[0], 3), "class": best[1]}
        return out
    results, matched = [], set()
    for s in staged:
        truth = s["box"]
        on_wall = [r for r in damage if r["surface_id"] == s["surface_id"] and r["class"] == s["class"]]
        best = max(on_wall, key=lambda r: iou([r["outline_uv"][0][0], r["outline_uv"][2][0], r["outline_uv"][0][1],
                                                r["outline_uv"][2][1]], truth), default=None)
        entry = {"staged": {"class": s["class"], "surface_id": s["surface_id"], "bd3_row": textures[s["label"]]["bd3_row"],
                            "width_m": round(truth[1] - truth[0], 3), "height_m": round(truth[3] - truth[2], 3),
                            "bottom_m": round(truth[2], 3), "area_m2": round((truth[1] - truth[0]) * (truth[3] - truth[2]), 3),
                            "frames_painted": len(pixels[s["surface_id"]]), "frames_seeing_fully": s["frames_seeing_fully"]},
                 "found": best is not None, "tiles_covering_it": tile_scores(s["surface_id"])}
        if best:
            matched.add(best["id"])
            box = [best["outline_uv"][0][0], best["outline_uv"][2][0], best["outline_uv"][0][1], best["outline_uv"][2][1]]
            area_true = entry["staged"]["area_m2"]
            entry.update({"region": best["id"], "iou": round(iou(box, truth), 3), "width_m": best["width_m"]["value"],
                          "height_m": best["height_m"]["value"], "area_m2": best["area_m2"]["value"],
                          "area_error_pct": round(100 * (best["area_m2"]["value"] / area_true - 1), 1),
                          "area_interval_holds": bool(best["area_m2"]["ci_low"] <= area_true <= best["area_m2"]["ci_high"]),
                          "views": best["views"], "detection_confidence": best["detection_confidence"],
                          "rule_fired": any(f["rule_id"] == s["rule"] and best["id"] in f["evidence"] for f in flags),
                          "scope_items": [i["description"] for i in scope if i["surface_id"] == s["surface_id"]]})
        else:
            entry["regions_on_that_wall"] = [(r["class"], r["detection_confidence"]) for r in damage if r["surface_id"] == s["surface_id"]]
        results.append(entry)
        print(json.dumps(entry), flush=True)
    false = [r for r in damage if r["id"] not in matched]
    report = {"benchmark": "staged_damage", "code_commit": code_commit(), "walk": WALK, "p_min": dd.P_MIN,
              "views": len(views), "staged": results,
              "summary": {"found_right_class_right_wall": f"{sum(r['found'] for r in results)}/{len(results)}",
                          "iou_at_least_0.5": f"{sum(r.get('iou', 0) >= 0.5 for r in results)}/{len(results)}",
                          "area_within_25pct": f"{sum(abs(r.get('area_error_pct', 1e9)) <= 25 for r in results)}/{len(results)}",
                          "rules_fired": f"{sum(r.get('rule_fired', False) for r in results)}/{len(results)}",
                          "false_regions": len(false), "false_regions_detail": [(r["surface_id"], r["class"]) for r in false],
                          "control_regions_unpainted": len(control)}}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("summary", json.dumps(report["summary"]))
    sample = [v for v in painted if v.frame in pixels[staged[0]["surface_id"]]][:1] + \
             [v for v in painted if v.frame in pixels[staged[-1]["surface_id"]]][:1]
    for n, v in enumerate(sample):
        img = cv2.rotate(v.image, v.upright) if v.upright is not None else v.image
        cv2.imwrite(str(Path(sys.argv[1]) / f"staged_{n}.jpg") if len(sys.argv) > 1 else f"/tmp/staged_{n}.jpg",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
