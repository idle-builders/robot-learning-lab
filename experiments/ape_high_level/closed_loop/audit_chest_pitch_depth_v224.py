"""V224: fixed chest pitch alternatives for full and ideal short depth maps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_origin_route_v152 as route
from audit_chest_short_depth_v223 import CONFIGS, ideal_mask
from run import ROOT, Room, sha
from run_rgb_floor_target_v213 import floor_entry
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v224"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V223 = ROOT / "artifacts/ape-closed-loop-cove50-v223/chest-depth-range-audit.json"
CAMERAS = ("chest_low30_x45", "chest_low70_x45")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    prior = {r["seed"]: r for r in json.loads(V223.read_text())["rows"]}
    room = Room()
    old_mask = route.SelfDepthMask
    rows = []
    try:
        for camera in CAMERAS:
            def camera_mask(model, *, camera: str, width: int, height: int):
                assert camera == "lower_rgb"
                return SelfDepthMask(model, camera=current_camera,
                                     width=width, height=height)
            current_camera = camera
            for scene in scenes:
                if scene["label"] != "generous_kinematic":
                    continue
                seed = scene["seed"]
                entry = next(r for r in entries if r["seed"] == seed and
                             not r["absent"] and r["light_multiplier"] == 1.)
                floor = floor_entry(entry)
                room.reset()
                room.layout(seed, absent=False, furnished=True)
                room.model.body_pos[room.model.body("loop_table").id] = (
                    .40, (1 if seed % 2 else -1) * .46, 0.)
                room.set_table_height(.37)
                counts = []
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
                    rgb, dense = room.capture(camera, depth=True)
                    assert dense is not None
                    pose = room.camera_pose_estimate(camera)
                    count = {"yaw_rad": yaw, "dense_valid_pixels": int(np.isfinite(dense).sum())}
                    for label, spec in CONFIGS.items():
                        branch = OUT / camera / label / f"seed{seed}-light1.00-present"
                        branch.mkdir(parents=True, exist_ok=True)
                        masked = ideal_mask(dense, spec)
                        path = branch / name
                        np.savez_compressed(path, rgb=np.asarray(rgb), depth_m=masked,
                                            camera_origin_m=pose[0], camera_rotation=pose[1],
                                            actor_qpos=room.pose_data.qpos.copy(),
                                            estimated_pose_xyyaw=estimate,
                                            capture_time_s=stamp)
                        count[f"{label}_valid_pixels"] = int(np.isfinite(masked).sum())
                        count[f"{label}_sha256"] = sha(path)
                    counts.append(count)
                positions = scene["audits"]["0.065"]["positions_xy_m"]
                variant = {"variant": f"seed{seed}-{camera}-depth-range-compare",
                           "cream_cheese_xy_m": positions["CreamCheese"],
                           "parmesan_xy_m": positions["Parmesan"]}
                selected = {}
                route.SelfDepthMask = camera_mask
                try:
                    for label in CONFIGS:
                        selected[label] = route.screen(seed, variant, floor,
                            frame_root=OUT / camera / label, light_multiplier=1.)
                finally:
                    route.SelfDepthMask = old_mask
                rows.append({"camera": camera, "seed": seed,
                             "frame_counts": counts,
                             "chest_50_baseline": {
                                 key: value["status"] for key, value in
                                 prior[seed]["selected_routes"].items()},
                             "selected_routes": selected})
                print(camera, seed,
                      {key: value["status"] for key, value in selected.items()},
                      flush=True)
    finally:
        route.SelfDepthMask = old_mask
        room.close()
    result = {"scope": "V224 revealed initial-route fixed chest pitch screen following V223 50deg failure. Same V203/V204 origin, lower RGB floor target, virtual central4/8/11cm scans, actor planner, chest self mask and evaluator-only selected-path clearance. Test hypothetical torso camera pitch-down30 and pitch-down70 with ideal full exact depth and 7-50cm/84x58deg short depth; V223 covers pitch-down50. No physical camera mount, stereo quality, native skill or live loop.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v223_sha256": sha(V223),
              "rows": rows}
    (OUT / "chest-pitch-depth-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
