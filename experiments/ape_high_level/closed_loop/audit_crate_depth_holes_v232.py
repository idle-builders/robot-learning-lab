"""V232: adversarial missing-crate depth in the ideal short-head map."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_origin_route_v152 as route
from audit_head_short_depth_v225 import FRAME_ROOT as BASE_FRAMES
from depth_voxel import DepthVoxelMap
from run import H, ROOT, W, sha
from run_moved_props_v134 import MovedPropRoom
from run_rgb_floor_target_v213 import floor_entry
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v232"
FRAME_ROOT = OUT / "crate-hole-origin"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V225 = ROOT / "artifacts/ape-closed-loop-cove50-v225/head-short-depth-route-audit.json"


def crate_mask(room: MovedPropRoom) -> np.ndarray:
    room.renderer.enable_segmentation_rendering()
    room.renderer.update_scene(room.data, camera="lower_rgb")
    labels = room.renderer.render().copy()
    room.renderer.disable_segmentation_rendering()
    gid = room.model.geom("loop_crate_geom").id
    return ((labels[:, :, 0] == gid) &
            (labels[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    baseline = {r["seed"]: r for r in json.loads(V225.read_text())["rows"]}
    rows = []
    for scene in scenes:
        if scene["label"] != "generous_kinematic":
            continue
        seed = scene["seed"]
        positions = scene["audits"]["0.065"]["positions_xy_m"]
        entry = next(r for r in entries if r["seed"] == seed and
                     not r["absent"] and r["light_multiplier"] == 1.)
        room = MovedPropRoom({"CreamCheese": tuple(positions["CreamCheese"]),
                              "Parmesan": tuple(positions["Parmesan"])})
        try:
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
            room.set_table_height(.37)
            room._forward()
            branch = FRAME_ROOT / f"seed{seed}-light1.00-present"
            branch.mkdir(parents=True, exist_ok=True)
            counts = []
            for view in entry["rows"]:
                yaw = float(view["yaw_command_rad"])
                name = f"lower-angle-{yaw:+.1f}.npz"
                original = BASE_FRAMES / branch.name / name
                with np.load(original) as z:
                    values = {key: z[key].copy() for key in z.files}
                room.data.qpos[:] = values["actor_qpos"]
                room.actual[:] = 0.
                room.head = float(room.data.qpos[room.head_addr])
                room._forward()
                mask = crate_mask(room)
                depth = values["depth_m"]
                removed = int((mask & np.isfinite(depth)).sum())
                values["depth_m"] = depth.copy()
                values["depth_m"][mask] = np.nan
                output = branch / name
                np.savez_compressed(output, **values)
                counts.append({"yaw_rad": yaw,
                               "visible_crate_pixels_evaluator_only": int(mask.sum()),
                               "short_depth_crate_pixels_removed": removed,
                               "short_depth_valid_after": int(np.isfinite(values["depth_m"]).sum()),
                               "source_sha256": sha(original), "output_sha256": sha(output)})
            variant = {"variant": f"seed{seed}-short-depth-no-crate-returns",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            selected = route.screen(seed, variant, floor_entry(entry),
                                    frame_root=FRAME_ROOT, light_multiplier=1.)
            voxel = DepthVoxelMap()
            self_mask = SelfDepthMask(room.model, camera="lower_rgb",
                                      width=W, height=H)
            try:
                stamp = 0.
                for view in entry["rows"]:
                    name = f"lower-angle-{float(view['yaw_command_rad']):+.1f}.npz"
                    with np.load(branch / name) as z:
                        filtered, _ = self_mask.apply(z["depth_m"], z["actor_qpos"])
                        stamp = float(z["capture_time_s"])
                        voxel.observe(filtered,
                                      (z["camera_origin_m"], z["camera_rotation"]),
                                      stamp, fovy_deg=67., self_masked=True)
            finally:
                self_mask.close()
            coverage = []
            for index, (x, y) in enumerate(selected.get("selected_path_cells", [])):
                sample = voxel.buffered_upper_coverage(
                    np.array([.04 * x, .04 * y, 0.]), stamp)
                coverage.append({"path_index": index, **sample})
            row = {"seed": seed, "frame_counts": counts,
                   "baseline_status": baseline[seed]["short_depth_route"]["status"],
                   "baseline_path_cells": baseline[seed]["short_depth_route"].get("selected_path_cells"),
                   "crate_hole_route": selected,
                   "actor_map_upper_volume_coverage": coverage}
            rows.append(row)
            print(seed, selected["status"], selected.get("selected_kind"),
                  len(selected.get("selected_path_cells", [])), flush=True)
        finally:
            room.close()
    result = {"scope": "V232 revealed three V203 initial-route stress screens on V225 ideal short 7-50cm lower-head depth. Simulator evaluator segmentation deletes every visible crate depth return before actor mapping; RGB owner boxes, known-floor target ray, virtual 4/8/11cm scans, poses and frozen planner unchanged. This is an adversarial correlated depth-hole bound, not a calibrated D401/D405 material/stereo failure model. Actor never sees crate labels; evaluator path clearance scored after route choice. No live loop, physical package or stop qualification.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v225_sha256": sha(V225),
              "rows": rows}
    (OUT / "crate-depth-hole-route-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
