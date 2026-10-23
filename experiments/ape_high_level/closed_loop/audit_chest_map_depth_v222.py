"""V222: separate chest depth optical role with head RGB floor target."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_origin_route_v152 as route
from audit_aligned_depth_route_v214 import mask_depth
from run import ROOT, Room, sha
from run_rgb_floor_target_v213 import floor_entry
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v222"
FRAME_ROOT = OUT / "chest-depth-origin-frames"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V215 = ROOT / "artifacts/ape-closed-loop-cove50-v215/range-limited-depth-live-audit.json"
CAMERA = "chest_low50_x45"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    prior = {r["seed"]: r for r in json.loads(V215.read_text())["selected_routes"]}
    rows = []
    room = Room()
    original_mask = route.SelfDepthMask
    def chest_mask(model, *, camera, width, height):
        assert camera == "lower_rgb"
        return SelfDepthMask(model, camera=CAMERA, width=width, height=height)
    try:
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
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            room.model.body_pos[room.model.body("loop_table").id] = (
                .40, (1 if seed % 2 else -1) * .46, 0.)
            room.set_table_height(.37)
            room.model.light_diffuse[0] *= 1.
            for view in entry["rows"]:
                yaw = float(view["yaw_command_rad"])
                name = f"lower-angle-{yaw:+.1f}.npz"
                original = (ROOT / "artifacts/ape-closed-loop-cove50-v204" /
                            f"seed{seed}-light1.00-present" / name)
                with np.load(original) as z:
                    qpos = z["actor_qpos"].copy()
                    estimate = z["estimated_pose_xyyaw"].copy()
                    stamp = float(z["capture_time_s"])
                room.data.qpos[:] = qpos
                room.actual[:] = (0., 0., 0.)
                room.estimated[:] = estimate
                room.head = float(qpos[room.head_addr])
                room.time_s = stamp
                room._forward()
                rgb, dense = room.capture(CAMERA, depth=True)
                assert dense is not None
                limited = mask_depth(dense)
                pose = room.camera_pose_estimate(CAMERA)
                output = branch / name
                np.savez_compressed(output, rgb=np.asarray(rgb), depth_m=limited,
                                    camera_origin_m=pose[0], camera_rotation=pose[1],
                                    actor_qpos=room.pose_data.qpos.copy(),
                                    estimated_pose_xyyaw=estimate,
                                    capture_time_s=stamp)
                counts.append({"yaw_rad": yaw,
                               "dense_valid_pixels": int(np.isfinite(dense).sum()),
                               "limited_valid_pixels": int(np.isfinite(limited).sum()),
                               "actor_frame_sha256": sha(output)})
            positions = scene["audits"]["0.065"]["positions_xy_m"]
            variant = {"variant": f"seed{seed}-head-rgb-chest-limited-depth",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            route.SelfDepthMask = chest_mask
            try:
                selected = route.screen(seed, variant, floor,
                                        frame_root=FRAME_ROOT, light_multiplier=1.)
            finally:
                route.SelfDepthMask = original_mask
            rows.append({"seed": seed, "chest_frame_counts": counts,
                         "limited_head_status": prior[seed]["status"],
                         "limited_head_path_cells": prior[seed].get("selected_path_cells"),
                         "chest_depth_route": selected})
            print(seed, selected["status"], selected.get("selected_kind"),
                  "steps", len(selected.get("selected_path_cells", [])), flush=True)
    finally:
        route.SelfDepthMask = original_mask
        room.close()
    result = {"scope": "V222 revealed initial-route optical screen: same V203 generous room, V204 lower-head RGB owner identity, V213 RGB known-floor target positions, selected pose and virtual central 4/8/11cm scans; replace 3-D map images with separate hypothetical fixed torso/chest-low50 depth viewpoint, ideally center-aligned 0.3-3m/87x58deg D430-class range/FOV mask. Self mask uses chest model; no real stereo holes, timing, extrinsics, module/processor package or native action. Evaluator route clearance only after actor route choice. Tests whether a torso sensor could supply the map role rather than squeezing a third module into the current two-RGB head.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v215_sha256": sha(V215),
              "rows": rows}
    (OUT / "chest-depth-origin-route-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
