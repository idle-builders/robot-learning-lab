"""V94: render a small near-arm occluder and test actor self-depth masking."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from depth_voxel import DepthVoxelMap, VOXEL_M
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, sensed_range, sha
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
V90 = ROOT / "artifacts/ape-closed-loop-cove50-v90"
MASK_CODE = SOURCE.with_name("self_depth_mask.py")
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v94"


def reproject(pixel_row: int, pixel_col: int, depth_m: float,
              origin: np.ndarray, rotation: np.ndarray,
              fovy_deg: float) -> np.ndarray:
    focal = H / (2 * math.tan(math.radians(fovy_deg) / 2))
    ray = np.array([(pixel_col - W / 2) / focal,
                    -(pixel_row - H / 2) / focal, -1.])
    return origin + rotation @ (ray * depth_m)


def occupied_upper(voxel: DepthVoxelMap, stamp: float) -> set[tuple[int, int, int]]:
    return {cell for cell, hit_s in voxel.occupied.items()
            if hit_s >= voxel.free.get(cell, -math.inf)
            and hit_s >= stamp - 1.6
            and .18 <= (cell[2] + .5) * VOXEL_M <= .43}


def replay(seed: int, room: Room, robot: RobotGeometry,
           mask: SelfDepthMask) -> dict:
    branch = V90 / f"seed{seed}-actor"
    trace_path = branch / "trace.json"
    frame_path = branch / "frame-13.npz"
    trace = json.loads(trace_path.read_text())
    with np.load(frame_path) as z:
        qpos = z["actor_qpos"].copy()
        pose = z["estimated_pose_xyyaw"].copy()
        stamp = float(z["capture_time_s"])
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = qpos
    room.actual[:] = np.asarray(
        trace["final_actual_pose_xyyaw_evaluator_only"])
    room.estimated[:] = pose
    room.head = float(qpos[room.head_addr])
    room.time_s = stamp
    room._forward()
    camera_id = room.model.camera("lower_rgb").id
    camera_origin = room.data.cam_xpos[camera_id].copy()
    camera_rotation = room.data.cam_xmat[camera_id].reshape(3, 3).copy()
    fovy = float(room.model.cam_fovy[camera_id])
    baseline_rgb, baseline_depth = room.capture("lower_rgb", depth=True)
    baseline_filtered, baseline_mask = mask.apply(baseline_depth, qpos)
    baseline_scans = {str(height): sensed_range(
        room, seed, 194000 + int(height * 100), height_m=height)
        for height in (.04, .08, .11)}

    # A robot-only render nominates a camera ray for the *evaluator* scene
    # edit. The actor later receives only the changed measured depth image.
    mask.data.qpos[:] = qpos
    mujoco.mj_forward(room.model, mask.data)
    mask.renderer.enable_depth_rendering()
    mask.renderer.update_scene(mask.data, camera="lower_rgb",
                               scene_option=mask.option)
    predicted_self = mask.renderer.render().copy()
    mask.renderer.disable_depth_rendering()
    candidate = []
    for row in range(8, H - 8, 4):
        for col in range(8, W - 8, 4):
            self_depth = float(predicted_self[row, col])
            if not .155 <= self_depth <= .19:
                continue
            point = reproject(row, col, self_depth - .04,
                              camera_origin, camera_rotation, fovy)
            if not .18 <= point[2] <= .43:
                continue
            # Prefer the center of a visible self patch so a small box
            # renders several closer pixels rather than grazing one edge.
            patch = predicted_self[row - 4:row + 5, col - 4:col + 5]
            support = int(np.count_nonzero(np.abs(patch - self_depth) < .025))
            candidate.append((-support, row, col, self_depth, point))
    if not candidate:
        return {"seed": seed, "status": "no_self_ray_candidate"}
    _, row, col, self_depth, center = min(candidate,
                                           key=lambda item: item[:3])
    box = room.model.geom("loop_box_geom")
    box.size[:] = (.012, .012, .012)
    room.model.body_pos[room.model.body("loop_box").id] = center
    room._forward()
    changed_rgb, changed_depth = room.capture("lower_rgb", depth=True)
    filtered, changed_mask = mask.apply(changed_depth, qpos)
    robot.set_state(qpos)
    solids = robot.primitives()
    baseline_voxel = DepthVoxelMap()
    changed_voxel = DepthVoxelMap()
    camera_pose = room.camera_pose_estimate("lower_rgb")
    for map_, depth in ((baseline_voxel, baseline_filtered),
                        (changed_voxel, filtered)):
        map_.observe(depth, camera_pose, stamp, fovy_deg=fovy,
                     self_hit=None)
    old_upper = occupied_upper(baseline_voxel, stamp)
    new_upper = occupied_upper(changed_voxel, stamp)
    extra = new_upper - old_upper
    near_box = {cell for cell in extra if np.linalg.norm(
        (np.asarray(cell[:2], dtype=float) + .5) * VOXEL_M -
        center[:2]) <= .12}
    scan_counts = {}
    scan_changed = {}
    for height in (.04, .08, .11):
        a = sensed_range(room, seed, 194000 + int(height * 100),
                         height_m=height)
        scan_counts[str(height)] = sum(value is not None for value in a)
        before = baseline_scans[str(height)]
        scan_changed[str(height)] = sum(
            (left is None) != (right is None) or
            (left is not None and right is not None and
             abs(left - right) >= .01)
            for left, right in zip(before, a))
    branch_out = OUT / f"seed{seed}"
    branch_out.mkdir(parents=True, exist_ok=True)
    baseline_rgb.save(branch_out / "before.png")
    changed_rgb.save(branch_out / "near-arm-occluder.png")
    np.savez_compressed(branch_out / "near-arm-actor-input.npz",
                        rgb=np.asarray(changed_rgb),
                        depth_m=changed_depth,
                        filtered_depth_m=filtered,
                        actor_qpos=qpos,
                        estimated_pose_xyyaw=pose,
                        camera_origin_m=camera_pose[0],
                        camera_rotation=camera_pose[1],
                        capture_time_s=stamp)
    return {"seed": seed, "status": "rendered",
            "v90_trace_sha256": sha(trace_path),
            "v90_endpoint_frame_sha256": sha(frame_path),
            "selected_self_pixel_rc": [row, col],
            "predicted_self_depth_m": round(self_depth, 4),
            "occluder_center_evaluator_only_m": center.tolist(),
            "before_mask": baseline_mask,
            "after_mask": changed_mask,
            "extra_actor_upper_occupied_cells": len(extra),
            "extra_upper_cells_near_new_box": len(near_box),
            "evaluator_selected_model_clearance_m": round(
                room._current_obstacle_clearance()[0], 4),
            "range_nonmissing_beams_by_height": scan_counts,
            "range_changed_beams_by_height": scan_changed,
            "before_rgb_sha256": sha(branch_out / "before.png"),
            "after_rgb_sha256": sha(branch_out / "near-arm-occluder.png"),
            "actor_input_sha256": sha(branch_out / "near-arm-actor-input.npz")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    mask = SelfDepthMask(room.model, camera="lower_rgb",
                         width=W, height=H)
    try:
        robot = RobotGeometry(room.model)
        rows = [replay(seed, room, robot, mask) for seed in (85, 88)]
    finally:
        mask.close()
        room.close()
    gate = all(row["status"] == "rendered" and
               row["after_mask"]["closer_surface_pixels_preserved"] > 0 and
               row["extra_upper_cells_near_new_box"] > 0
               for row in rows)
    result = {"scope": "Rendered near-arm 2.4cm cube placed about4cm in front of a predicted self ray at each V90 endpoint. Actor mask uses only selected CAD/shell and joint/head/camera state; scene edit/clearance evaluator-only. Tests closer-than-self observation preservation, not natural dynamic hand behavior or motion.",
              "source_sha256": sha(SOURCE),
              "mask_code_sha256": sha(MASK_CODE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows,
              "near_arm_occluder_preserved_gate_pass": gate}
    (OUT / "near-arm-occluder-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_self_occluder_v94_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "self_depth_mask_imported_source.py").write_bytes(
        MASK_CODE.read_bytes())
    for row in rows:
        print(row["seed"], row["status"],
              row.get("after_mask"), "extra upper",
              row.get("extra_actor_upper_occupied_cells"), "near",
              row.get("extra_upper_cells_near_new_box"), flush=True)
    print("near arm occluder gate:", gate, flush=True)


if __name__ == "__main__":
    main()
