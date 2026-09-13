# Damage classes, concealed-damage rules and scope

The code is `cozmo/damage/`. Every concealed-damage flag in an output names one of these rules in `rule_id` (gate A-FLAG-RULE). Every scope item names its surface and the damage regions or rules behind it (A-SCOPE-KEY).

## Detection

CLIP ViT-B/32 scores 12 tiles of each image against prompts for 5 damage classes (water_stain, mold, crack, peeling_paint, hole) and 22 undamaged things (plain wall, ceiling, floor, furniture, door, window, picture, shadow, light and others). A tile is damage when the damage prompts together take at least 0.6 of the probability. The tile's depth pixels are placed on the plan and measured on the nearest surface. A region needs 2 images (1 at the photo tier). This is zero-shot and **not tested on real damage**: `bench/damage_sanity.py` measures false alarms on our undamaged walks and recall on painted stains only.

## Concealed-damage rules

A rule fires on one damage region when its class, surface and position match.

| Rule | Fires on | Flag: what may be hidden | Severity |
|---|---|---|---|
| `R-CEIL-WATER` | water stain or mold on a ceiling | possible leak above: wet insulation, joists or subfloor of the room above | high |
| `R-WALL-BASE-WATER` | water stain or mold on a wall, starting within 0.3 m of the floor | possible wet wall cavity, rotten sole plate or damp subfloor behind the skirting | high |
| `R-MOLD-HIDDEN` | mold on any surface | mold on a finish usually continues inside the cavity behind it | high |
| `R-PEEL-MOISTURE` | peeling paint on a wall or ceiling | moisture behind the paint layer; check the substrate with a moisture meter | medium |
| `R-CRACK-LONG` | a crack at least 1 m long on a wall or ceiling | possible structural movement or settlement behind the finish | medium |
| `R-HOLE-SERVICES` | a hole in a wall | wiring or pipes behind the hole may be damaged | low |

## Scope

Each damage region adds a repair on its surface. Each damaged surface is repainted whole, because paint is applied corner to corner. Each concealed-damage flag adds an inspection of the cavity. Quantities come from the plan and carry its intervals: a wall's area is its length times the room's ceiling height, and a ceiling's area is the room's floor area.

| Damage class | Repair on its surface | Unit |
|---|---|---|
| water_stain | seal the stain with stain-blocking primer | m2 |
| mold | clean and treat mold (surface remediation), 0.3 m beyond the visible edge | m2 |
| crack | tape, fill and sand the crack | m |
| peeling_paint | scrape loose paint and prime | m2 |
| hole | patch the hole in the plasterboard | each |
