"""V101: actor-only first annulus step from two fresh learned-object views."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap
from goal_map_route import GoalMapRoute
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, sensed_range, sha
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance


SOURCE = Path(__file__)
V96 = ROOT / "artifacts/ape-closed-loop-cove50-v96"
V97 = ROOT / "artifacts/ape-closed-loop-cove50-v97"
V100 = ROOT / "artifacts/ape-closed-loop-cove50-v100"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v101"


def accepted_source(seed: int) -> tuple[np.ndarray, np.ndarray,
                                        list[Path], dict, Path]:
    if seed == 85:
        audit_path = V100 / "live-side-extension-audit.json"
        audit = json.loads(audit_path.read_text())
        view_rows = audit["views"]["views"]
        accepted = next(row for row in view_rows
                        if row["present_accepted_correct_evaluator_only"])
        proposal = next(item for item in accepted["present_proposals"]
                        if item["actor_gate_accept"])
        target = np.asarray(proposal["depth_world_xyz_m"][:2], dtype=float)
        actual = np.asarray(audit["base"]["evaluator_final_actual_xyyaw"])
        paths = [V100 / f"seed85-endpoint-views/view-{index}-actor-input.npz"
                 for index in range(3)]
        source = {"accepted_similarity": proposal["similarity"],
                  "accepted_box": proposal["box"],
                  "source_branch": "V100 information view"}
    else:
        audit_path = V97 / "post-detour-reacquisition-audit.json"
        audit = json.loads(audit_path.read_text())
        branch = next(row for row in audit["rows"] if row["seed"] == seed)
        accepted = next(row for row in branch["views"]
                        if row["present"]["evaluator_accept_is_target"])
        target = np.asarray(accepted["present"]["accepted_world_xyz_m"][:2],
                            dtype=float)
        actual = np.asarray(json.loads((V96 / f"seed{seed}-repair/trace.json")
                            .read_text())["evaluator_final_actual_xyyaw"])
        paths = [V97 / f"seed{seed}/view-{index}-actor-input.npz"
                 for index in range(3)]
        source = {"accepted_similarity": accepted["present"][
                      "accepted_similarity"],
                  "accepted_box": accepted["present"]["accepted_box"],
                  "source_branch": "V97 post-detour view"}
    return target, actual, paths, source, audit_path


def replay(seed: int, room: Room, robot: RobotGeometry,
           mask: SelfDepthMask) -> dict:
    target, actual, paths, accepted, audit_path = accepted_source(seed)
    voxel = DepthVoxelMap()
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    qpos = None
    pose = None
    stamp = None
    inputs = []
    for path in paths:
        with np.load(path) as z:
            qpos = z["actor_qpos"].copy()
            pose = z["estimated_pose_xyyaw"].copy()
            depth = z["depth_m"].copy()
            camera = (z["camera_origin_m"].copy(),
                      z["camera_rotation"].copy())
            stamp = float(z["capture_time_s"])
        filtered, _ = mask.apply(depth, qpos)
        voxel.observe(filtered, camera, stamp, fovy_deg=fovy,
                      self_masked=True)
        inputs.append(sha(path))
    assert qpos is not None and pose is not None and stamp is not None
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = qpos
    room.actual[:] = actual
    room.estimated[:] = pose
    room.head = float(qpos[room.head_addr])
    room.time_s = stamp
    room._forward()
    scan11 = sensed_range(room, seed, 1010011, height_m=.11)
    scan8 = sensed_range(room, seed, 1010008, height_m=.08)
    scan4 = sensed_range(room, seed, 1010004, height_m=.04)
    point = TwoHeightClearance(robot, qpos, pose, scan11, scan4,
                               voxel, stamp, range_8=scan8)
    planner = GoalMapRoute()
    command = planner.plan(scan11, pose, target, None, stamp,
                           searching=False,
                           edge_clearance=point.edge_gap)
    true_first = None
    step_delta = None
    if command is not None:
        heading, _, _, detail = command
        step_delta = detail["step_distance_m"] * np.asarray([
            math.cos(heading), math.sin(heading)])
        samples = []
        for index in range(21):
            room.actual[:] = actual
            room.actual[:2] += .88 * step_delta * index / 20
            room._forward()
            samples.append(float(room._current_obstacle_clearance()[0]))
        true_first = min(samples)
    return {"seed": seed,
            "accepted_source_audit_sha256": sha(audit_path),
            "head_actor_input_sha256": inputs,
            "accepted_actor_target_xy_m": target.tolist(),
            "accepted_source": accepted,
            "actor_pose_xyyaw": pose.tolist(),
            "actor_upper_voxel_occupied": len(voxel.occupied),
            "actor_range_source_counts": point.source_counts(),
            "planner_command": (None if command is None else {
                "heading_rad": round(command[0], 4),
                "heading_offset_rad": round(command[1], 4),
                "actor_first_edge_point_gap_m": round(command[2], 4),
                "detail": command[3]}),
            "planner_debug": planner.last_debug,
            "first_command_delta_xy_m": (None if step_delta is None
                                         else step_delta.tolist()),
            "first_edge_evaluator_min_clearance_m": (
                None if true_first is None else round(true_first, 4))}


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
    gate = all(row["planner_command"] is not None and
               row["planner_command"]["detail"]["goal_kind"] == "annulus" and
               row["first_edge_evaluator_min_clearance_m"] >= .05
               for row in rows)
    result = {"scope": "Actor-only first known-free GoalMapRoute annulus command from fresh V100/V97 learned target RGB-D proposals, latest 4/8/11cm virtual range and V95 self-masked near head-depth. Frozen6.5cm selected-body first-edge point check. Evaluator geometry labels only after plan; no motion executed.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows,
              "fresh_target_first_step_screen_pass": gate}
    (OUT / "fresh-target-first-step-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_fresh_target_approach_v101_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], row["planner_command"], "true first",
              row["first_edge_evaluator_min_clearance_m"], flush=True)
    print("fresh target first step screen:", gate, flush=True)


if __name__ == "__main__":
    main()
