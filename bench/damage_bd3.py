"""The damage detector on real defect photos: BD3 (Kottari and Arjunan, 2024), via its CC-BY-4.0 re-release
`chandrabhuma/building_defect_vqa` on Hugging Face (test split). Phone photos of walls, about 1 m away, one label each:
algae, major_crack, minor_crack, peeling, plain, spalling, stain. The images are used locally and never redistributed.

The detector runs exactly as shipped (cozmo/damage/detect.py: 12 tiles per image, CLIP ViT-B/32, P_MIN). An image
counts as detected when any tile passes; its class is the best tile's class. BD3 classes map to ours as:
stain -> water_stain, algae -> mold, major/minor crack -> crack, peeling -> peeling_paint, spalling -> hole (the
nearest of our five). `plain` walls measure false alarms. A threshold sweep shows the recall and false-alarm trade.

    python bench/damage_bd3.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cozmo.damage.detect as dd  # noqa: E402

OUT = ROOT / "bench" / "results" / "damage_bd3.json"
REPO, FILE = "chandrabhuma/building_defect_vqa", "data/test-00000-of-00001.parquet"
MAPPING = {"stain": "water_stain", "algae": "mold", "major_crack": "crack", "minor_crack": "crack",
           "peeling": "peeling_paint", "spalling": "hole", "plain": None}
SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7)


def main():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from PIL import Image
    table = pd.read_parquet(hf_hub_download(REPO, FILE, repo_type="dataset"))
    shipped = dd.P_MIN
    dd.P_MIN = 0.0                                   # keep every tile's best class and probability
    best = []                                        # per image: (label, best class, best probability)
    for s in range(0, len(table), 64):
        rows = table.iloc[s:s + 64]
        views = []
        for k, row in enumerate(rows.itertuples()):
            img = np.asarray(Image.open(io.BytesIO(row.image["bytes"])).convert("RGB"))
            views.append(dd.View(img, np.zeros((4, 4)), (1, 1, 1, 1), np.eye(3), np.zeros(3), k))
        top = {}
        for k, _, cls, p in dd.classify(views):
            if p > top.get(k, (None, -1))[1]:
                top[k] = (cls, p)
        for k, row in enumerate(rows.itertuples()):
            cls, p = top.get(k, (None, 0.0))
            best.append((row.answer, cls, p))
        print(f"{min(s + 64, len(table))}/{len(table)}", flush=True)
    report = {"benchmark": "damage_bd3", "dataset": f"{REPO} ({FILE}), derived from BD3", "images": len(best),
              "p_min_shipped": shipped, "mapping": MAPPING, "by_threshold": {}}
    for t in SWEEP:
        per = {}
        for label in sorted({b[0] for b in best}):
            rows = [b for b in best if b[0] == label]
            hit = [b for b in rows if b[2] >= t]
            entry = {"images": len(rows), "flagged": len(hit), "flagged_share": round(len(hit) / len(rows), 3)}
            if MAPPING.get(label):
                entry["right_class"] = sum(b[1] == MAPPING[label] for b in hit)
            per[label] = entry
        damaged = [b for b in best if MAPPING.get(b[0])]
        plain = [b for b in best if b[0] == "plain"]
        report["by_threshold"][str(t)] = {
            "recall_any_damage": round(sum(b[2] >= t for b in damaged) / len(damaged), 3),
            "right_class_share_of_damaged": round(sum(b[2] >= t and b[1] == MAPPING[b[0]] for b in damaged) / len(damaged), 3),
            "false_alarm_share_of_plain": round(sum(b[2] >= t for b in plain) / max(len(plain), 1), 3),
            "per_class": per}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for t, r in report["by_threshold"].items():
        print(t, {k: r[k] for k in ("recall_any_damage", "right_class_share_of_damaged", "false_alarm_share_of_plain")})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
