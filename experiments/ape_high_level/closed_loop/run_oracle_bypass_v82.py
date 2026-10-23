"""V82: execute V77 truth route only as a guarded-motion upper bound."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robot_geometry import RobotGeometry
from run import ROOT, Room, sha
from run_guarded_side_look_v80 import GRID_M, run_branch


SOURCE = Path(__file__)
HARNESS = Path(__file__).with_name("run_guarded_side_look_v80.py")
ROUTES = ROOT / "artifacts/ape-closed-loop-cove50-v77/active-route-coverage-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v82"


def oracle_cells(route_row: dict) -> list[tuple[int, int]]:
    route = np.asarray(route_row["reference_route"]["path_xy_m"], dtype=float)
    actual = np.asarray(route_row["reference_start_actual_xyyaw"])
    estimated = np.asarray(route_row["actor_estimated_xyyaw"])
    actor_points = estimated[:2] + route - actual[:2]
    cells = []
    for point in actor_points:
        node = tuple(int(round(float(value) / GRID_M)) for value in point)
        if not cells or node != cells[-1]:
            cells.append(node)
    if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
           for a, b in zip(cells[:-1], cells[1:])):
        raise AssertionError("Oracle route projection skipped a grid cell")
    return cells


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads(ROUTES.read_text())
    routes = {int(row["seed"]): row for row in audit["rows"]}
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = []
        for seed in (85, 88):
            cells = oracle_cells(routes[seed])
            print(seed, "oracle cells", len(cells), flush=True)
            rows.append(run_branch(
                seed, 0, routes[seed], room, robot,
                oracle_path_cells=cells, output_root=OUT))
    finally:
        room.close()
    gate = all(row["status"] == "viewpoint_reached" and
               row["evaluator_true_target_gap_m"] <= .34 and
               row["evaluator_min_path_clearance_m"] >= .05 and
               not row["evaluator_model_overlap"]
               for row in rows)
    result = {"scope": "Evaluator V77 path cells supplied to V80 guarded sensor/motion harness as an explicit oracle-route upper bound from exact V72 corridor stops. The 4/8/11cm precheck, 10Hz 4/11cm alarm and scan localizer still use only actor sensors; no actor planner claim or native gait/stop qualification.",
              "source_sha256": sha(SOURCE),
              "motion_harness_sha256": sha(HARNESS),
              "v77_audit_sha256": sha(ROUTES),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": [{key: value for key, value in row.items()
                        if key != "events"} for row in rows],
              "oracle_route_execution_gate_pass": gate}
    (OUT / "oracle-bypass-execution-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "run_oracle_bypass_v82_source.py").write_bytes(SOURCE.read_bytes())
    (OUT / "run_guarded_side_look_v80_source.py").write_bytes(
        HARNESS.read_bytes())
    print("oracle route execution gate:", gate)


if __name__ == "__main__":
    main()
