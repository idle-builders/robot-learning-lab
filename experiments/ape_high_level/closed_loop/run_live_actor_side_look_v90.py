"""V90: live V89 side choice joined to V80 guarded observation motion."""

from __future__ import annotations

import json
from pathlib import Path

from robot_geometry import RobotGeometry
from run import ROOT, Room, sha
from run_guarded_side_look_v80 import run_branch


SOURCE = Path(__file__)
RUNNER = SOURCE.with_name("run_guarded_side_look_v80.py")
SIDE_CODE = SOURCE.with_name("upper_obstacle_side.py")
ROUTES = ROOT / "artifacts/ape-closed-loop-cove50-v77/active-route-coverage-audit.json"
V89 = ROOT / "artifacts/ape-closed-loop-cove50-v89/upper-obstacle-side-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v90"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    route_rows = {int(row["seed"]): row for row in
                  json.loads(ROUTES.read_text())["rows"]}
    references = {int(row["seed"]): row[
        "independent_reference_side_evaluator_only"] for row in
        json.loads(V89.read_text())["rows"]}
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [run_branch(seed, None, route_rows[seed], room, robot,
                           output_root=OUT) for seed in (85, 88)]
    finally:
        room.close()
    for row in rows:
        row["independent_reference_side_evaluator_only"] = references[
            row["seed"]]
        row["selected_side_matches_reference"] = (
            row["side"] == references[row["seed"]])
        (OUT / f"seed{row['seed']}-actor/trace.json").write_text(
            json.dumps(row, indent=2) + "\n")
    gate = all(row["selected_side_matches_reference"] and
               row["status"] == "viewpoint_reached" and
               row["evaluator_min_path_clearance_m"] >= .05 and
               not row["evaluator_model_overlap"] for row in rows)
    result = {"scope": "V89 actor upper-depth side cue selected live V80 guarded local side observation from V72 actor stops. 4/8/11cm point prechecks, 10Hz independent4/11cm monitor, scan localization and fresh depth; V77 route only post-action evaluation. Kinematic base/head, no owner target reach/native stopping claim.",
              "source_sha256": sha(SOURCE),
              "runner_source_sha256": sha(RUNNER),
              "side_code_sha256": sha(SIDE_CODE),
              "v77_evaluator_audit_sha256": sha(ROUTES),
              "v89_audit_sha256": sha(V89),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": [{key: value for key, value in row.items()
                        if key != "events"} for row in rows],
              "live_actor_side_look_gate_pass": gate}
    (OUT / "live-actor-side-look-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "run_live_actor_side_look_v90_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "run_guarded_side_look_v80_imported_source.py").write_bytes(
        RUNNER.read_bytes())
    (OUT / "upper_obstacle_side_imported_source.py").write_bytes(
        SIDE_CODE.read_bytes())
    for row in rows:
        print(row["seed"], row["side"], row["status"],
              "steps", row["executed_base_steps"],
              "min", row["evaluator_min_path_clearance_m"],
              "matches", row["selected_side_matches_reference"],
              flush=True)
    print("live actor side look gate:", gate, flush=True)


if __name__ == "__main__":
    main()
