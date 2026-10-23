"""V225: ideal short-range lower-head depth as separate 3-D map source."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_origin_route_v152 as route
from audit_chest_short_depth_v223 import ideal_mask
from run import ROOT, sha
from run_rgb_floor_target_v213 import floor_entry


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v225"
FRAME_ROOT = OUT / "short-head-depth-origin"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V215 = ROOT / "artifacts/ape-closed-loop-cove50-v215/range-limited-depth-live-audit.json"
V220 = ROOT / "artifacts/ape-closed-loop-cove50-v220/no-map-depth-route-audit.json"
SHORT = (.07, .50, 84., 58.)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    long = {r["seed"]: r for r in json.loads(V215.read_text())["selected_routes"]}
    none = {r["seed"]: r["no_depth_route"] for r in json.loads(V220.read_text())["rows"]}
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
        counts = []
        for view in entry["rows"]:
            yaw = float(view["yaw_command_rad"])
            name = f"lower-angle-{yaw:+.1f}.npz"
            original = (ROOT / "artifacts/ape-closed-loop-cove50-v204" /
                        f"seed{seed}-light1.00-present" / name)
            with np.load(original) as z:
                values = {key: z[key].copy() for key in z.files}
            dense = values["depth_m"]
            values["depth_m"] = ideal_mask(dense, SHORT)
            output = branch / name
            np.savez_compressed(output, **values)
            counts.append({"yaw_rad": yaw,
                           "dense_valid_pixels": int(np.isfinite(dense).sum()),
                           "short_valid_pixels": int(np.isfinite(values["depth_m"]).sum()),
                           "original_sha256": sha(original),
                           "short_sha256": sha(output)})
        positions = scene["audits"]["0.065"]["positions_xy_m"]
        variant = {"variant": f"seed{seed}-head-rgb-floor-short-depth",
                   "cream_cheese_xy_m": positions["CreamCheese"],
                   "parmesan_xy_m": positions["Parmesan"]}
        selected = route.screen(seed, variant, floor,
                                frame_root=FRAME_ROOT, light_multiplier=1.)
        rows.append({"seed": seed, "frame_counts": counts,
                     "long_depth_status": long[seed]["status"],
                     "long_depth_path_cells": long[seed].get("selected_path_cells"),
                     "no_depth_status": none[seed]["status"],
                     "short_depth_route": selected})
        print(seed, selected["status"], selected.get("selected_kind"),
              len(selected.get("selected_path_cells", [])), flush=True)
    result = {"scope": "V225 revealed initial-route paired sensor role screen: same V203 rooms, V204 lower-head RGB owner boxes, V213 known-floor target ray, virtual central4/8/11cm scans and frozen route code; 3-D map from an ideal separate co-aligned lower-head 7-50cm,84x58deg short depth observation (D401-class range/FOV reference). Exact simulator depth masked before actor map; evaluator route score after choice. No real stereo holes, module/processor fit, timing, physical low range or live loop. This differs from V211's rejection of short depth for long-range can *location* because RGB floor geometry now handles that specific role.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v215_sha256": sha(V215),
              "v220_sha256": sha(V220), "short_range_fov": SHORT,
              "rows": rows}
    (OUT / "head-short-depth-route-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
