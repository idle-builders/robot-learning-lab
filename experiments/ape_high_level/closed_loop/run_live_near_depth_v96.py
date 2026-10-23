"""V96: live V93 outward detour using V95 near self-masked depth rays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, sha
from run_live_local_bypass_v86 import run
from run_live_self_masked_side_v93 import load_initial
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
RUNNER = SOURCE.with_name("run_live_local_bypass_v86.py")
LOADER = SOURCE.with_name("run_live_self_masked_side_v93.py")
MASK_CODE = SOURCE.with_name("self_depth_mask.py")
VOXEL_CODE = SOURCE.with_name("depth_voxel.py")
V95 = ROOT / "artifacts/ape-closed-loop-cove50-v95/near-depth-after-self-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v96"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    goals = {int(row["seed"]): row for row in
             json.loads(V95.read_text())["ordinary_route_rows"]}
    room = Room()
    mask = SelfDepthMask(room.model, camera="lower_rgb",
                         width=W, height=H)
    try:
        robot = RobotGeometry(room.model)
        rows = []
        for seed in (85, 88):
            initial, prior = load_initial(seed, room, robot, mask,
                                          allow_near_masked_depth=True)
            goal = goals[seed]
            row = run(seed, {"actor_goal_cell": goal["actor_goal_cell"],
                             "blocked_cell": None}, room, robot,
                      output_root=OUT, video_label="V96",
                      goal_audit_path=V95, initial=initial,
                      self_mask=mask, allow_near_masked_depth=True)
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
    result = {"scope": "Live actor-side outward V95 goals from V90 viewpoints with CAD/shell-predicted pixel self mask and valid near-camera depth voxels. Fresh4/8/11cm point prechecks, independent10Hz4/11cm guard, scan registration, depth reobservation. V94 cube tested separately; kinematic, no owner reach/native stop qualification.",
              "source_sha256": sha(SOURCE),
              "runner_source_sha256": sha(RUNNER),
              "loader_source_sha256": sha(LOADER),
              "mask_code_sha256": sha(MASK_CODE),
              "voxel_code_sha256": sha(VOXEL_CODE),
              "v95_audit_sha256": sha(V95),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": [{key: value for key, value in row.items()
                        if key != "events"} for row in rows],
              "live_near_depth_side_gate_pass": gate}
    (OUT / "live-near-depth-side-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "run_live_near_depth_v96_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "run_live_local_bypass_v86_imported_source.py").write_bytes(
        RUNNER.read_bytes())
    (OUT / "run_live_self_masked_side_v93_loader_source.py").write_bytes(
        LOADER.read_bytes())
    (OUT / "self_depth_mask_imported_source.py").write_bytes(
        MASK_CODE.read_bytes())
    (OUT / "depth_voxel_imported_source.py").write_bytes(VOXEL_CODE.read_bytes())
    for row in rows:
        print(row["seed"], row["status"], "moves",
              row["executed_moves"], "clearance",
              row["evaluator_min_path_clearance_m"], "side progress",
              row["actor_side_progress_m"], "target gap",
              row["start_target_gap_m"], "->", row["final_target_gap_m"],
              flush=True)
    print("live near-depth side gate:", gate, flush=True)


if __name__ == "__main__":
    main()
