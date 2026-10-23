"""V220: matched origin route ablation with no 3-D map depth, virtual scans kept."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_origin_route_v152 as route
from run import ROOT, sha
from run_rgb_floor_target_v213 import floor_entry


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v220"
FRAME_ROOT = OUT / "rgb-and-range-only-origin"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V213 = ROOT / "artifacts/ape-closed-loop-cove50-v213/rgb-floor-target-live-audit.json"
V215 = ROOT / "artifacts/ape-closed-loop-cove50-v215/range-limited-depth-live-audit.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    limited = {r["seed"]: r for r in json.loads(V215.read_text())["selected_routes"]}
    rows = []
    for scene in scenes:
        if scene["label"] != "generous_kinematic":
            continue
        seed = scene["seed"]
        entry = next(r for r in entries if r["seed"] == seed and
                     not r["absent"] and r["light_multiplier"] == 1.)
        floor = floor_entry(entry)
        branch = FRAME_ROOT / f"seed{seed}-light1.00-present"
        branch.mkdir(parents=True, exist_ok=True)
        hashes = []
        for view in entry["rows"]:
            yaw = float(view["yaw_command_rad"])
            name = f"lower-angle-{yaw:+.1f}.npz"
            original = (ROOT / "artifacts/ape-closed-loop-cove50-v204" /
                        f"seed{seed}-light1.00-present" / name)
            with np.load(original) as z:
                values = {key: z[key].copy() for key in z.files}
            values["depth_m"] = np.full_like(values["depth_m"], np.nan)
            output = branch / name
            np.savez_compressed(output, **values)
            hashes.append({"original_sha256": sha(original),
                           "no_depth_sha256": sha(output)})
        positions = scene["audits"]["0.065"]["positions_xy_m"]
        variant = {"variant": f"seed{seed}-rgb-floor-no-map-depth",
                   "cream_cheese_xy_m": positions["CreamCheese"],
                   "parmesan_xy_m": positions["Parmesan"]}
        selected = route.screen(seed, variant, floor,
                                frame_root=FRAME_ROOT, light_multiplier=1.)
        rows.append({"seed": seed,
                     "limited_depth_status": limited[seed]["status"],
                     "limited_depth_path_cells": limited[seed].get("selected_path_cells"),
                     "frame_pairs": hashes,
                     "no_depth_route": selected})
        print(seed, selected["status"], selected.get("selected_kind"),
              len(selected.get("selected_path_cells", [])), flush=True)
    result = {"scope": "V220 revealed matched V203 generous-room initial route ablation. Retain V204 owner RGB boxes and V213 RGB floor-ray target positions, selected body/pose, 4/8/11cm virtual central scans, and unchanged V152 planner; replace only all actor 3-D map depth pixels with missing. Evaluator path score after actor choice. Tests whether these initial information routes need depth beyond scans; no live motion/stop, physical scan layout or new threshold tuning.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v213_sha256": sha(V213),
              "v215_sha256": sha(V215), "rows": rows}
    (OUT / "no-map-depth-route-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
