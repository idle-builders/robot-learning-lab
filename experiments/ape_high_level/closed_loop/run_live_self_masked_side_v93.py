"""V93: live receding-horizon outward motion from V90 actor side look."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, sha
from run_live_local_bypass_v86 import run
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
RUNNER = SOURCE.with_name("run_live_local_bypass_v86.py")
MASK_CODE = SOURCE.with_name("self_depth_mask.py")
V90 = ROOT / "artifacts/ape-closed-loop-cove50-v90"
V92 = ROOT / "artifacts/ape-closed-loop-cove50-v92/self-depth-mask-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v93"


def load_initial(seed: int, room: Room, robot: RobotGeometry,
                 mask: SelfDepthMask, *,
                 allow_near_masked_depth: bool = False) -> tuple[dict, dict]:
    branch = V90 / f"seed{seed}-actor"
    trace_path = branch / "trace.json"
    trace = json.loads(trace_path.read_text())
    voxel = DepthVoxelMap()
    frame_hashes = []
    mask_totals = {"predicted_self_pixels": 0,
                   "masked_self_pixels": 0,
                   "closer_surface_pixels_preserved": 0}
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    final_qpos = None
    final_stamp = None
    for index in range(1, 14):
        path = branch / f"frame-{index}.npz"
        with np.load(path) as z:
            qpos = z["actor_qpos"].copy()
            depth = z["depth_m"].copy()
            camera = (z["camera_origin_m"].copy(),
                      z["camera_rotation"].copy())
            stamp = float(z["capture_time_s"])
        filtered, info = mask.apply(depth, qpos)
        for key in mask_totals:
            mask_totals[key] += info[key]
        robot.set_state(qpos)
        solids = robot.primitives()
        voxel.observe(filtered, camera, stamp, fovy_deg=fovy,
                      self_hit=None,
                      self_masked=allow_near_masked_depth)
        final_qpos, final_stamp = qpos, stamp
        frame_hashes.append(sha(path))
    assert final_qpos is not None and final_stamp is not None
    initial = {"voxel": voxel,
               "actor_qpos": final_qpos,
               "estimated_pose_xyyaw": np.asarray(
                   trace["final_estimated_pose_xyyaw"], dtype=float),
               "actual_pose_xyyaw_evaluator_only": np.asarray(
                   trace["final_actual_pose_xyyaw_evaluator_only"],
                   dtype=float),
               "capture_time_s": final_stamp,
               "visited_cells": trace["actor_graph_path_cells"],
               "provenance": {"v90_trace_sha256": sha(trace_path),
                              "v90_actor_frame_sha256": frame_hashes,
                              "initial_self_depth_mask": mask_totals}}
    return initial, trace


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    goals = {int(row["seed"]): row for row in
             json.loads(V92.read_text())["rows"]}
    room = Room()
    mask = SelfDepthMask(room.model, camera="lower_rgb",
                         width=W, height=H)
    try:
        robot = RobotGeometry(room.model)
        rows = []
        for seed in (85, 88):
            initial, prior = load_initial(seed, room, robot, mask)
            goal = goals[seed]
            row = run(seed, {"actor_goal_cell": goal["actor_goal_cell"],
                             "blocked_cell": None}, room, robot,
                      output_root=OUT, video_label="V93",
                      goal_audit_path=V92, initial=initial,
                      self_mask=mask)
            target = np.asarray(prior["target_estimate_xy_m"], dtype=float)
            side = np.asarray(goal["actor_side_unit_xy"], dtype=float)
            start = initial["estimated_pose_xyyaw"][:2]
            final = np.asarray(row["final_estimated_pose_xyyaw"][:2])
            row["actor_target_estimate_xy_m"] = target.tolist()
            row["start_target_gap_m"] = round(float(np.linalg.norm(
                target - start)), 4)
            row["final_target_gap_m"] = round(float(np.linalg.norm(
                target - final)), 4)
            row["actor_side_progress_m"] = round(float((final - start) @ side), 4)
            (OUT / f"seed{seed}-repair/trace.json").write_text(
                json.dumps(row, indent=2) + "\n")
            rows.append(row)
    finally:
        mask.close()
        room.close()
    gate = all(row["status"] == "local_goal_reached" and
               row["evaluator_min_path_clearance_m"] >= .05 and
               not row["evaluator_model_overlap"] for row in rows)
    result = {"scope": "Live actor-chosen outward local goals from V90 non-oracle side looks, V92 CAD/shell-predicted self depth before 3D map fusion, fresh 4/8/11cm point prechecks, 10Hz independent4/11cm monitor, scan registration, head RGB-D and synchronized video. Kinematic, not owner target reach/native stop qualification.",
              "source_sha256": sha(SOURCE),
              "runner_source_sha256": sha(RUNNER),
              "mask_code_sha256": sha(MASK_CODE),
              "v92_goal_audit_sha256": sha(V92),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": [{key: value for key, value in row.items()
                        if key != "events"} for row in rows],
              "live_self_masked_side_gate_pass": gate}
    (OUT / "live-self-masked-side-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "run_live_self_masked_side_v93_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "run_live_local_bypass_v86_imported_source.py").write_bytes(
        RUNNER.read_bytes())
    (OUT / "self_depth_mask_imported_source.py").write_bytes(
        MASK_CODE.read_bytes())
    for row in rows:
        print(row["seed"], row["status"], "moves",
              row["executed_moves"], "clearance",
              row["evaluator_min_path_clearance_m"], "side progress",
              row["actor_side_progress_m"], "target gap",
              row["start_target_gap_m"], "->", row["final_target_gap_m"],
              flush=True)
    print("live self-masked side gate:", gate, flush=True)


if __name__ == "__main__":
    main()
