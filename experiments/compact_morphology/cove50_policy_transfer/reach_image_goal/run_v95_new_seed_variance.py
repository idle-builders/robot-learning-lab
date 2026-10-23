"""Frozen new-seed toy, motor and gaze challenge for the automatic V95 reach rule."""

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
OUT = BASE / "v95"
# seed, toy X/Y (m), rail (V), delay (ms), physical gaze yaw (rad)
CONDITIONS = (
    (17001, .00615, .00600, 11, 10, -1.3),
    (17002, .00630, .00600, 10, 0, -1.3),
    (17003, .00645, .00600, 12, 10, -1.3),
    (17004, .00585, .00600, 11, 10, -1.3),
    (17005, .00615, .00620, 11, 10, -1.3),
    (17006, .00630, .00580, 11, 10, -1.3),
    (17007, .00615, .00600, 11, 10, -1.5),
    (17008, .00630, .00600, 11, 10, -1.5),
    (17009, .00615, .00600, 12, 0, -1.3),
    (17010, .00630, .00600, 10, 10, -1.3),
    (17011, .00585, .00620, 10, 0, -1.3),
    (17012, .00645, .00580, 12, 10, -1.3),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_matrix(plan_path: Path, dest: Path, cases: list[dict], rows: dict) -> dict:
    ordered = [rows[case["id"]] for case in cases if case["id"] in rows]
    complete = len(ordered) == len(cases)
    automatic = sum(row["automatic_alternate_gaze_specialist_applied"] for row in ordered)
    natural_specialist = sum(row["selection_mode"] == "v16_specialist_locked" and
                             not row["automatic_alternate_gaze_specialist_applied"]
                             for row in ordered)
    rear_bias = sum(row["automatic_rear_contact_bias_applied"] for row in ordered)
    gate = (complete and all(row["pass"] and row["contact_s"] >= 1.2 for row in ordered)
            and automatic >= 3 and natural_specialist >= 2 and rear_bias == 0)
    matrix = {"schema": "robot-learning-lab.cove50-v95-new-seed-variance.v1",
              "role": "automatic_observed_reach_new_seed_development",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "plan_sha256": sha(plan_path), "complete": complete,
              "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
              "automatic_specialist_use_count": automatic,
              "natural_specialist_count": natural_specialist,
              "rear_bias_use_count": rear_bias,
              "development_gate_met": gate, "cases": ordered}
    temp = dest / "matrix.json.tmp"
    temp.write_text(json.dumps(matrix, indent=2) + "\n")
    os.replace(temp, dest / "matrix.json")
    return matrix


def main() -> None:
    pilot_plan_path = OUT / "two-region-pilot-plan.json"
    pilot_matrix_path = OUT / "native-two-region-pilot/matrix.json"
    pilot_plan = json.loads(pilot_plan_path.read_text())
    pilot = json.loads(pilot_matrix_path.read_text())
    if (not pilot["complete"] or not pilot["development_gate_met"] or
            pilot["plan_sha256"] != sha(pilot_plan_path) or
            pilot_plan["source_sha256"]["evaluator"] != sha(native.EVALUATOR) or
            pilot_plan["source_sha256"]["pointer"] !=
            sha(ROOT / "robot/ape-current-design.json")):
        raise ValueError("Need passing source-matched V95 pilot and unchanged body pointer")
    anchor = next(case for case in pilot_plan["cases"]
                  if case["id"] == "rear-gaze-minus1p3-zero")
    cases = []
    for seed, toy_x, toy_y, rail, delay, gaze in CONDITIONS:
        case = {key: value for key, value in anchor.items()
                if not key.startswith("expected_") and key not in ("role", "id")}
        case.update(id=f"new-seed-{seed}", seed=seed,
                    toy_offset_x_m=toy_x, toy_offset_y_m=toy_y,
                    rail_v=rail, delay_ms=delay,
                    gaze_head_yaw_rad=gaze,
                    expected_rear_contact_bias=False)
        cases.append(case)
    plan = {"schema": "robot-learning-lab.cove50-v95-new-seed-variance-plan.v1",
            "role": "automatic_observed_reach_new_seed_development",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "source_sha256": pilot_plan["source_sha256"],
            "pilot_plan_sha256": sha(pilot_plan_path),
            "pilot_matrix_sha256": sha(pilot_matrix_path),
            "native_case_runner_sha256": sha(Path(native.__file__)),
            "runner_sha256": sha(Path(__file__)),
            "case_count": len(cases), "cases": cases,
            "variation": ("Twelve previously unused simulator seeds 17001–17012; "
                          "toy X 5.85–6.45 mm, toy Y 5.80–6.20 mm, 10–12 V, "
                          "0–10 ms delay and physical head yaw −1.3/−1.5 rad. "
                          "All route choices arise from the captured RGB and camera pose."),
            "gate": ("All twelve native visits pass full contact, RGB, two-foot support, "
                     "rated motor, collision and action checks with >=1.2 s contact; "
                     "at least three automatic alternate-gaze specialist uses and two "
                     "naturally selected specialist controls; no unintended V93 rear "
                     "goal correction. This is development on a rejected body."),
            }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "new-seed-variance-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V95 new-seed plan/source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    dest = OUT / "native-new-seed-variance"
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
                       "natural_specialist_count", "development_gate_met")}), flush=True)


if __name__ == "__main__":
    main()
