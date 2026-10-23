"""V88: execute V87 actor-target local bypasses under fresh observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robot_geometry import RobotGeometry
from run import ROOT, Room, sha
from run_live_local_bypass_v86 import run


SOURCE = Path(__file__)
RUNNER = SOURCE.with_name("run_live_local_bypass_v86.py")
V87 = ROOT / "artifacts/ape-closed-loop-cove50-v87/actor-target-bypass-audit.json"
V82 = ROOT / "artifacts/ape-closed-loop-cove50-v82"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v88"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    goals = {int(row["seed"]): row for row in json.loads(V87.read_text())["rows"]}
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = []
        for seed in (85, 88):
            goal = goals[seed]
            trace = json.loads((V82 / f"seed{seed}-oracle/trace.json").read_text())
            target = np.asarray(trace["target_estimate_xy_m"], dtype=float)
            start_xy = np.asarray(goal["actor_stop_pose_xyyaw"][:2],
                                  dtype=float)
            row = run(seed, {"actor_goal_cell": goal["actor_goal_cell"],
                             "blocked_cell": goal["excluded_rejected_cell"]},
                      room, robot, output_root=OUT, video_label="V88",
                      goal_audit_path=V87)
            row["actor_target_estimate_xy_m"] = target.tolist()
            row["start_target_gap_m"] = round(float(np.linalg.norm(
                target - start_xy)), 4)
            row["final_target_gap_m"] = round(float(np.linalg.norm(
                target - np.asarray(row[
                    "final_estimated_pose_xyyaw"][:2]))), 4)
            (OUT / f"seed{seed}-repair/trace.json").write_text(
                json.dumps(row, indent=2) + "\n")
            rows.append(row)
    finally:
        room.close()
    gate = all(row["status"] == "local_goal_reached" and
               row["evaluator_min_path_clearance_m"] >= .05 and
               not row["evaluator_model_overlap"] and
               not row["blocked_cell_traversed"] and
               row["final_target_gap_m"] < row["start_target_gap_m"]
               for row in rows)
    result = {"scope": "Live local receding-horizon execution from exact V82 stops of V87 actor-target-direction goals. Still inherits V82 oracle-produced history and rejected cell; not an oracle-free end-to-end route. Target-area reach and native stop not claimed.",
              "source_sha256": sha(SOURCE),
              "runner_source_sha256": sha(RUNNER),
              "v87_audit_sha256": sha(V87),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": [{key: value for key, value in row.items()
                        if key != "events"} for row in rows],
              "actor_target_local_bypass_gate_pass": gate}
    (OUT / "actor-target-live-bypass-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "run_actor_target_bypass_v88_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "run_live_local_bypass_v86_imported_source.py").write_bytes(
        RUNNER.read_bytes())
    for row in rows:
        print(row["seed"], row["status"], "moves",
              row["executed_moves"], "clearance",
              row["evaluator_min_path_clearance_m"], "target gap",
              row["start_target_gap_m"], "->", row["final_target_gap_m"],
              flush=True)
    print("actor target local bypass gate:", gate, flush=True)


if __name__ == "__main__":
    main()
