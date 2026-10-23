"""V223: chest depth range role, full versus 7-50cm ideal short depth."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_origin_route_v152 as route
from run import H, ROOT, W, Room, sha
from run_rgb_floor_target_v213 import floor_entry
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v223"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V222 = ROOT / "artifacts/ape-closed-loop-cove50-v222/chest-depth-origin-route-audit.json"
CAMERA = "chest_low50_x45"
CONFIGS = {"full": (0., 2., None, None),
           "short_07_50": (.07, .50, 84., 58.)}


def ideal_mask(depth: np.ndarray, spec: tuple) -> np.ndarray:
    minimum, maximum, fovx, fovy = spec
    result = depth.copy()
    valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    if fovx is not None:
        focal = H / (2 * math.tan(math.radians(67.) / 2))
        ys, xs = np.indices((H, W))
        valid &= (np.abs(np.arctan((xs - W / 2) / focal)) <=
                  math.radians(fovx / 2))
        valid &= (np.abs(np.arctan((ys - H / 2) / focal)) <=
                  math.radians(fovy / 2))
    result[~valid] = np.nan
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    previous = {r["seed"]: r for r in json.loads(V222.read_text())["rows"]}
    room = Room()
    old_mask = route.SelfDepthMask
    def chest_mask(model, *, camera, width, height):
        assert camera == "lower_rgb"
        return SelfDepthMask(model, camera=CAMERA, width=width, height=height)
    rows = []
    try:
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
            crate_geom = room.model.geom("loop_crate_geom").id
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
                rgb, dense = room.capture(CAMERA, depth=True)
                assert dense is not None
                pose = room.camera_pose_estimate(CAMERA)
                room.renderer.enable_segmentation_rendering()
                room.renderer.update_scene(room.data, camera=CAMERA)
                segments = room.renderer.render().copy()
                room.renderer.disable_segmentation_rendering()
                crate = ((segments[:, :, 0] == crate_geom) &
                         (segments[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
                crate_ranges = dense[crate]
                count = {"yaw_rad": yaw, "crate_visible_pixels_evaluator_only": int(crate.sum()),
                         "crate_depth_quantiles_m_evaluator_only":
                             np.round(np.quantile(crate_ranges, [.05, .5, .95]), 4).tolist()
                             if len(crate_ranges) else None}
                for label, spec in CONFIGS.items():
                    branch = OUT / label / f"seed{seed}-light1.00-present"
                    branch.mkdir(parents=True, exist_ok=True)
                    masked = ideal_mask(dense, spec)
                    output = branch / name
                    np.savez_compressed(output, rgb=np.asarray(rgb), depth_m=masked,
                                        camera_origin_m=pose[0], camera_rotation=pose[1],
                                        actor_qpos=room.pose_data.qpos.copy(),
                                        estimated_pose_xyyaw=estimate,
                                        capture_time_s=stamp)
                    count[f"{label}_valid_pixels"] = int(np.isfinite(masked).sum())
                    count[f"{label}_crate_valid_pixels_evaluator_only"] = int(
                        np.isfinite(masked[crate]).sum())
                    count[f"{label}_sha256"] = sha(output)
                counts.append(count)
            positions = scene["audits"]["0.065"]["positions_xy_m"]
            variant = {"variant": f"seed{seed}-chest-depth-range-compare",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            selected = {}
            route.SelfDepthMask = chest_mask
            try:
                for label in CONFIGS:
                    selected[label] = route.screen(seed, variant, floor,
                                                   frame_root=OUT / label,
                                                   light_multiplier=1.)
            finally:
                route.SelfDepthMask = old_mask
            rows.append({"seed": seed, "counts": counts,
                         "d430_class_chest_status": previous[seed]["chest_depth_route"]["status"],
                         "selected_routes": selected})
            print(seed, {key: value["status"] for key, value in selected.items()},
                  flush=True)
    finally:
        route.SelfDepthMask = old_mask
        room.close()
    result = {"scope": "V223 revealed same-scene chest-depth range role screen after V222 D430-class chest failed. On V203 generous rooms and V204 saved origin poses, keep lower-head RGB identity, V213 floor-ray target, virtual central4/8/11cm scans and frozen route code. Compare exact simulator chest depth with an ideal 7-50cm,84x58deg short-range mask at the same hypothetical fixed chest-low50 optical center; chest self mask for both. Evaluator crate segmentation/depth quantiles and path clearance are only diagnostics after actor observation/route choice. Not a real D401/D405 package, stereo quality, physical placement or live loop.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v222_sha256": sha(V222),
              "configurations": CONFIGS, "rows": rows}
    (OUT / "chest-depth-range-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
