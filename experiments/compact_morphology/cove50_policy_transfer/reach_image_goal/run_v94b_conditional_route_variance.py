"""Condition prescribed specialist visits on the actually observed scoped route."""

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
    run_v94_native_case as native,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v94b"
CONDITIONS = (
    ("toy-plus015mm", .00615, 11, 10),
    ("toy-plus030mm", .00630, 11, 10),
    ("toy-plus045mm", .00645, 11, 10),
    ("toy-minus015mm", .00585, 11, 10),
    ("rail10-delay0", .006, 10, 0),
    ("rail12-delay10", .006, 12, 10),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_matrix(plan_path: Path, dest: Path, baselines: dict, forced: dict,
                 complete: bool) -> dict:
    pairs = []
    for name, *_ in CONDITIONS:
        baseline = baselines.get(name)
        specialist = forced.get(name)
        if baseline:
            pairs.append({"condition": name,
                          "baseline_route": baseline["selection_mode"],
                          "baseline_contact_s": baseline["contact_s"],
                          "baseline_pass": baseline["pass"],
                          "forced_attempted": specialist is not None,
                          "forced_route": None if specialist is None else specialist["selection_mode"],
                          "forced_contact_s": None if specialist is None else specialist["contact_s"],
                          "forced_pass": None if specialist is None else specialist["pass"]})
    rows = [row for name, *_ in CONDITIONS
            for row in (baselines.get(name), forced.get(name)) if row is not None]
    scoped = [p for p in pairs if p["baseline_route"] == "v21_scoped_y_locked"]
    natural = [p for p in pairs if p["baseline_route"] != "v21_scoped_y_locked"]
    rescues = sum(not p["baseline_pass"] and bool(p["forced_pass"]) and
                  p["forced_contact_s"] >= 1.2 for p in scoped)
    gate = (complete and len(pairs) == len(CONDITIONS) and len(scoped) >= 3 and
            rescues >= 3 and
            all(p["forced_attempted"] and p["forced_route"] == "v16_specialist_locked" and
                p["forced_pass"] and p["forced_contact_s"] >= 1.2 for p in scoped) and
            any(p["baseline_route"] == "v16_specialist_locked" and p["baseline_pass"]
                for p in natural) and
            all(p["baseline_pass"] and p["baseline_contact_s"] >= 1.2 for p in natural))
    matrix = {"schema": "robot-learning-lab.cove50-v94b-conditional-route-variance.v1",
              "role": "conditional_prescribed_specialist_teacher_not_automatic_control",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "plan_sha256": sha(plan_path), "complete": complete,
              "condition_count": len(pairs), "case_count": len(rows),
              "pass_count": sum(row["pass"] for row in rows),
              "scoped_pair_count": len(scoped),
              "natural_specialist_count": sum(p["baseline_route"] == "v16_specialist_locked"
                                              for p in natural),
              "paired_rescue_count": rescues,
              "diagnostic_gate_met": gate, "pairs": pairs, "cases": rows}
    temp = dest / "matrix.json.tmp"
    temp.write_text(json.dumps(matrix, indent=2) + "\n")
    os.replace(temp, dest / "matrix.json")
    return matrix


def main() -> None:
    teacher_plan_path = BASE / "v94/alternate-gaze-route-plan.json"
    teacher_matrix_path = BASE / "v94/native-alternate-gaze-route/matrix.json"
    stop_path = BASE / "v94/gaze-route-variance-stop.json"
    teacher_plan = json.loads(teacher_plan_path.read_text())
    teacher = json.loads(teacher_matrix_path.read_text())
    stop = json.loads(stop_path.read_text())
    if (not teacher["complete"] or not teacher["teacher_found"] or
            stop["diagnostic_gate_met"] or
            teacher["plan_sha256"] != sha(teacher_plan_path) or
            teacher_plan["source_sha256"]["evaluator"] != sha(native.EVALUATOR)):
        raise ValueError("Need completed V94 teacher and stopped overstrict paired plan")
    anchor = next(case for case in teacher_plan["cases"] if case["id"] == "scoped-zero")
    baseline_cases = [{**anchor, "id": name + "-baseline", "condition": name,
                       "toy_offset_x_m": toy_x, "rail_v": rail, "delay_ms": delay,
                       "diagnostic_force_specialist": False,
                       "diagnostic_goal_shift_body_y_m": 0.0,
                       "expected_selection_mode": None}
                      for name, toy_x, rail, delay in CONDITIONS]
    plan = {
        "schema": "robot-learning-lab.cove50-v94b-conditional-route-variance-plan.v1",
        "role": "conditional_prescribed_specialist_teacher_not_automatic_control",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": teacher_plan["source_sha256"],
        "teacher_plan_sha256": sha(teacher_plan_path),
        "teacher_matrix_sha256": sha(teacher_matrix_path),
        "stopped_v94_plan_sha256": stop["plan_sha256"],
        "stopped_v94_record_sha256": sha(stop_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "condition_count": len(CONDITIONS),
        "maximum_visit_count": len(CONDITIONS) * 2,
        "baseline_cases": baseline_cases,
        "conditional_visit_rule": ("After every natural-route baseline, replay an otherwise "
                                   "identical prescribed specialist visit only when the "
                                   "baseline selected v21_scoped_y_locked. Record natural "
                                   "specialist successes without forcing again."),
        "gate": ("Complete six natural baselines and every eligible forced replay. "
                 "Require at least three actual scoped failures rescued to full native "
                 "passes with >=1.2 s contact, all scoped forced visits passing, "
                 "at least one natural specialist pass, and every non-scoped baseline "
                 "passing. This is teacher variation only."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "conditional-route-variance-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V94b plan/source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    dest = OUT / "native-conditional-route-variance"
    dest.mkdir(parents=True, exist_ok=True)
    baselines = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(native.run_case, case, plan, dest): case
                   for case in baseline_cases}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            baselines[case["condition"]] = row
            write_matrix(plan_path, dest, baselines, {}, False)
            print(json.dumps({"condition": case["condition"], "visit": "baseline",
                              "pass": row["pass"], "contact_s": row["contact_s"],
                              "selection_mode": row["selection_mode"]}), flush=True)
    forced_cases = []
    for case in baseline_cases:
        if baselines[case["condition"]]["selection_mode"] == "v21_scoped_y_locked":
            forced_cases.append({**case, "id": case["condition"] + "-forced-specialist",
                                 "diagnostic_force_specialist": True,
                                 "expected_selection_mode": "v16_specialist_locked"})
    forced = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(native.run_case, case, plan, dest): case
                   for case in forced_cases}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            forced[case["condition"]] = row
            write_matrix(plan_path, dest, baselines, forced, False)
            print(json.dumps({"condition": case["condition"], "visit": "forced-specialist",
                              "pass": row["pass"], "contact_s": row["contact_s"],
                              "selection_mode": row["selection_mode"]}), flush=True)
    matrix = write_matrix(plan_path, dest, baselines, forced, True)
    print(json.dumps({key: matrix[key] for key in
                      ("condition_count", "case_count", "scoped_pair_count",
                       "natural_specialist_count", "paired_rescue_count",
                       "diagnostic_gate_met")}), flush=True)


if __name__ == "__main__":
    main()
