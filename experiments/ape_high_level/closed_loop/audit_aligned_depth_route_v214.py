"""V214: initial actor route with ideal aligned D430-class range and field mask."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import audit_origin_route_v152 as route
from run import H, ROOT, W, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v214"
FRAME_ROOT = OUT / "range-limited-origin-frames"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V205 = ROOT / "artifacts/ape-closed-loop-cove50-v205/fourth-split-route-audit.json"
RGB_FOVY_DEG = 67.
DEPTH_FOVX_DEG = 87.
DEPTH_FOVY_DEG = 58.
MIN_DEPTH_M = .30
MAX_DEPTH_M = 3.


def mask_depth(depth: np.ndarray) -> np.ndarray:
    # Ideal upper bound: center-aligned depth optics and no real stereo holes,
    # noise, motion blur, occlusion baseline, latency or calibration error.
    focal = H / (2 * math.tan(math.radians(RGB_FOVY_DEG) / 2))
    ys, xs = np.indices((H, W))
    horizontal = np.abs(np.arctan((xs - W / 2) / focal))
    vertical = np.abs(np.arctan((ys - H / 2) / focal))
    valid = ((depth >= MIN_DEPTH_M) & (depth <= MAX_DEPTH_M) &
             (horizontal <= math.radians(DEPTH_FOVX_DEG / 2)) &
             (vertical <= math.radians(DEPTH_FOVY_DEG / 2)))
    result = depth.copy()
    result[~valid] = np.nan
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    previous = {row["seed"]: row for row in json.loads(V205.read_text())["rows"]}
    rows = []
    for scene in scenes:
        if scene["label"] != "generous_kinematic":
            continue
        seed = scene["seed"]
        entry = next(row for row in entries if row["seed"] == seed and
                     not row["absent"] and row["light_multiplier"] == 1.)
        branch = FRAME_ROOT / f"seed{seed}-light1.00-present"
        branch.mkdir(parents=True, exist_ok=True)
        pixel_counts = []
        for view in entry["rows"]:
            yaw = float(view["yaw_command_rad"])
            name = f"lower-angle-{yaw:+.1f}.npz"
            original = (ROOT / "artifacts/ape-closed-loop-cove50-v204" /
                        f"seed{seed}-light1.00-present" / name)
            with np.load(original) as z:
                values = {key: z[key].copy() for key in z.files}
            unmasked = values["depth_m"]
            values["depth_m"] = mask_depth(unmasked)
            np.savez_compressed(branch / name, **values)
            pixel_counts.append({"yaw_rad": yaw,
                                 "unmasked_finite": int(np.isfinite(unmasked).sum()),
                                 "masked_finite": int(np.isfinite(values["depth_m"]).sum()),
                                 "source_sha256": sha(original),
                                 "masked_sha256": sha(branch / name)})
        positions = scene["audits"]["0.065"]["positions_xy_m"]
        variant = {"variant": f"seed{seed}-aligned-ideal-depth-mask",
                   "cream_cheese_xy_m": positions["CreamCheese"],
                   "parmesan_xy_m": positions["Parmesan"]}
        screened = route.screen(seed, variant, entry,
                                frame_root=FRAME_ROOT, light_multiplier=1.)
        baseline = previous[seed]
        rows.append({"seed": seed, "baseline_status": baseline["status"],
                     "baseline_kind": baseline.get("selected_kind"),
                     "baseline_selected_path_cells": baseline.get("selected_path_cells"),
                     "masked_frame_counts": pixel_counts,
                     "masked_initial_route": screened})
        print(seed, screened["status"], screened.get("selected_kind"),
              "steps", len(screened.get("selected_path_cells", [])), flush=True)
    result = {"scope": "V214 paired actor initial-route screen on the three revealed V203 generous rooms. V204 accepted RGB identity and dense-depth target positions unchanged; only lower-camera map depth is ideally masked to 0.3-3m and a center-aligned 87x58deg D430-class field on the existing 640x480 67deg RGB view. Actor still has virtual central 4/8/11cm scans; evaluator path clearance scored afterward. This is an optimistic optical bound, not a packaged D430 or live stop result.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v205_sha256": sha(V205),
              "depth_range_m": [MIN_DEPTH_M, MAX_DEPTH_M],
              "depth_fov_degrees_hv": [DEPTH_FOVX_DEG, DEPTH_FOVY_DEG],
              "rows": rows}
    (OUT / "aligned-depth-route-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
