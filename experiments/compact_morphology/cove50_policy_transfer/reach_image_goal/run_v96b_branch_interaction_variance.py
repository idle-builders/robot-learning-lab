"""Fresh variation of specialist and rear-correction branch interaction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v96b"
# id, toy X/Y (m), rail (V), delay (ms), physical gaze yaw (rad),
# expected alternate specialist, expected older rear correction.
CONDITIONS = (
    ("auto-x-plus012", .00612, .00603, 11, 10, -1.3, True, False),
    ("auto-x-plus018", .00618, .00597, 11, 10, -1.3, True, False),
    ("auto-x-plus028", .00628, .00604, 11, 10, -1.3, True, False),
    ("auto-y-plus015", .00622, .00615, 11, 10, -1.3, True, False),
    ("auto-y-minus015", .00625, .00585, 11, 10, -1.3, True, False),
    ("auto-motor10", .00624, .00602, 10, 0, -1.3, True, False),
    ("auto-motor12", .00627, .00598, 12, 5, -1.3, True, False),
    ("natural-minus015-11v", .00585, .00600, 11, 10, -1.3, False, False),
    ("natural-minus012-12v", .00588, .00595, 12, 10, -1.3, False, False),
    ("rear-yaw-minus1p5-plus022", .00622, .00601, 11, 10, -1.5, False, True),
    ("rear-yaw-minus1p5-plus027", .00627, .00598, 11, 10, -1.5, False, True),
    ("rear-yaw-minus1p5-plus032", .00632, .00603, 11, 10, -1.5, False, True),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_matrix(plan_path: Path, dest: Path, cases: list[dict], rows: dict) -> dict:
    ordered = [rows[case["id"]] for case in cases if case["id"] in rows]
    by_id = {case["id"]: case for case in cases}
    complete = len(ordered) == len(cases)
    active = [row for row in ordered if row["automatic_alternate_gaze_specialist_applied"]]
    natural = [row for row in ordered if row["selection_mode"] == "v16_specialist_locked" and
               not row["automatic_alternate_gaze_specialist_applied"] and
               not by_id[row["id"]]["expected_rear_contact_bias"]]
    motor_exposure = len({(by_id[row["id"]]["rail_v"], by_id[row["id"]]["delay_ms"])
                          for row in active})
    toy_exposure = len({(by_id[row["id"]]["toy_offset_x_m"],
                         by_id[row["id"]]["toy_offset_y_m"]) for row in active})
    rear_bias = sum(row["automatic_rear_contact_bias_applied"] for row in ordered)
    expected_exposures = all(
        row["automatic_alternate_gaze_specialist_applied"] ==
        by_id[row["id"]]["expected_alternate_gaze_specialist"] and
        row["automatic_rear_contact_bias_applied"] ==
        by_id[row["id"]]["expected_rear_contact_bias"] for row in ordered)
    gate = (complete and all(row["pass"] and row["contact_s"] >= 1.2 for row in ordered)
            and len(active) >= 6 and len(natural) >= 2 and
            motor_exposure >= 2 and toy_exposure >= 3 and rear_bias >= 2 and
            expected_exposures)
    matrix = {"schema": "robot-learning-lab.cove50-v96b-branch-interaction-variance.v1",
              "role": "automatic_specialist_and_older_rear_rule_targeted_variation",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "plan_sha256": sha(plan_path), "complete": complete,
              "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
              "automatic_specialist_use_count": len(active),
              "natural_specialist_count": len(natural),
              "automatic_motor_condition_count": motor_exposure,
              "automatic_toy_condition_count": toy_exposure,
              "rear_bias_use_count": rear_bias,
              "expected_exposures_met": expected_exposures,
              "development_gate_met": gate, "cases": ordered}
    temp = dest / "matrix.json.tmp"
    temp.write_text(json.dumps(matrix, indent=2) + "\n")
    os.replace(temp, dest / "matrix.json")
    return matrix


def main() -> None:
    pilot_plan_path = BASE / "v95/two-region-pilot-plan.json"
    pilot_matrix_path = BASE / "v95/native-two-region-pilot/matrix.json"
    broad_plan_path = BASE / "v95/new-seed-variance-plan.json"
    broad_matrix_path = BASE / "v95/native-new-seed-variance/matrix.json"
    targeted_plan_path = BASE / "v96/targeted-region-variance-plan.json"
    targeted_matrix_path = BASE / "v96/native-targeted-region-variance/matrix.json"
    pilot_plan = json.loads(pilot_plan_path.read_text())
    pilot = json.loads(pilot_matrix_path.read_text())
    broad = json.loads(broad_matrix_path.read_text())
    targeted = json.loads(targeted_matrix_path.read_text())
    if (not pilot["complete"] or not pilot["development_gate_met"] or
            pilot["plan_sha256"] != sha(pilot_plan_path) or
            not broad["complete"] or broad["plan_sha256"] != sha(broad_plan_path) or
            not targeted["complete"] or targeted["plan_sha256"] != sha(targeted_plan_path) or
            pilot_plan["source_sha256"]["evaluator"] != sha(native.EVALUATOR) or
            pilot_plan["source_sha256"]["pointer"] !=
            sha(ROOT / "robot/ape-current-design.json")):
        raise ValueError("Need pilot, broad unused-seed study and completed V96 diagnosis")
    anchor = next(case for case in pilot_plan["cases"]
                  if case["id"] == "rear-gaze-minus1p3-zero")
    cases = []
    for ident, toy_x, toy_y, rail, delay, gaze, alternate, rear in CONDITIONS:
        case = {key: value for key, value in anchor.items()
                if not key.startswith("expected_") and key not in ("role", "id")}
        case.update(id=ident, toy_offset_x_m=toy_x, toy_offset_y_m=toy_y,
                    rail_v=rail, delay_ms=delay, gaze_head_yaw_rad=gaze,
                    expected_alternate_gaze_specialist=alternate,
                    expected_rear_contact_bias=rear)
        cases.append(case)
    plan = {
        "schema": "robot-learning-lab.cove50-v96b-branch-interaction-variance-plan.v1",
        "role": "automatic_specialist_and_older_rear_rule_targeted_variation",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": pilot_plan["source_sha256"],
        "pilot_plan_sha256": sha(pilot_plan_path),
        "pilot_matrix_sha256": sha(pilot_matrix_path),
        "broad_new_seed_plan_sha256": sha(broad_plan_path),
        "broad_new_seed_matrix_sha256": sha(broad_matrix_path),
        "broad_new_seed_gate_met": broad["development_gate_met"],
        "v96_targeted_plan_sha256": sha(targeted_plan_path),
        "v96_targeted_matrix_sha256": sha(targeted_matrix_path),
        "v96_targeted_gate_met": targeted["development_gate_met"],
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "variation": ("Twelve new source-frozen uninterrupted native visits near the "
                      "opened 10433 rear-camera boundary. Seven perturb the new "
                      "specialist region, two target natural specialist choices and "
                      "three change gaze to the older rear-correction region. Toy X/Y, "
                      "10–12 V, 0–10 ms delay and −1.3/−1.5 rad gaze all vary. "
                      "Routing is automatic from captured RGB and camera pose."),
        "gate": ("All twelve visits pass full native contact, RGB, support, rated motor "
                 "and collision checks with >=1.2 s contact; exact predeclared "
                 "alternate and rear branch exposures; at least six new specialist "
                 "uses, two natural specialist controls, two rear-correction uses, "
                 "two motor settings and three toy positions among new-rule uses. "
                 "The failed V95 broad and V96 targeted gates remain unchanged."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "branch-interaction-variance-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V96b plan/source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    dest = OUT / "native-branch-interaction-variance"
    dest.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        iterator = iter(cases)
        pending = {}
        for _ in range(4):
            case = next(iterator, None)
            if case:
                pending[pool.submit(native.run_case, case, plan, dest)] = case
        while pending:
            future = next(as_completed(tuple(pending)))
            case = pending.pop(future)
            row = future.result()
            rows[case["id"]] = row
            write_matrix(plan_path, dest, cases, rows)
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "selection_mode",
                               "automatic_alternate_gaze_specialist_applied", "faults")}),
                  flush=True)
            next_case = next(iterator, None)
            if next_case:
                pending[pool.submit(native.run_case, next_case, plan, dest)] = next_case
    matrix = write_matrix(plan_path, dest, cases, rows)
    print(json.dumps({key: matrix[key] for key in
                      ("case_count", "pass_count", "automatic_specialist_use_count",
                       "natural_specialist_count", "automatic_motor_condition_count",
                       "automatic_toy_condition_count", "development_gate_met")}),
          flush=True)


if __name__ == "__main__":
    main()
