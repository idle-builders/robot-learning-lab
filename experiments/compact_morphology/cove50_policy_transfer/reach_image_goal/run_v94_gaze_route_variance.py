"""Pair scoped and prescribed specialist routes near the alternate-gaze miss."""

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
OUT = BASE / "v94"
VARIATIONS = (
    ("toy-x-plus015mm", .00615, 11, 10),
    ("toy-x-minus015mm", .00585, 11, 10),
    ("rail10-delay0", .006, 10, 0),
    ("rail12-delay10", .006, 12, 10),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    teacher_plan_path = OUT / "alternate-gaze-route-plan.json"
    teacher_matrix_path = OUT / "native-alternate-gaze-route/matrix.json"
    teacher_plan = json.loads(teacher_plan_path.read_text())
    teacher = json.loads(teacher_matrix_path.read_text())
    if (not teacher["complete"] or not teacher["teacher_found"] or
            teacher["plan_sha256"] != sha(teacher_plan_path) or
            not any(row["id"] == "specialist-zero" and row["pass"]
                    for row in teacher["cases"])):
        raise ValueError("Need completed V94 specialist-route contact teacher")
    anchor = next(case for case in teacher_plan["cases"] if case["id"] == "scoped-zero")
    if teacher_plan["source_sha256"]["evaluator"] != sha(native.EVALUATOR):
        raise ValueError("V94 teacher evaluator source changed")
    cases = []
    for name, toy_x, rail, delay in VARIATIONS:
        for forced, label in ((False, "scoped"), (True, "specialist")):
            cases.append({**anchor, "id": f"{name}-{label}",
                          "toy_offset_x_m": toy_x, "rail_v": rail, "delay_ms": delay,
                          "diagnostic_force_specialist": forced,
                          "diagnostic_goal_shift_body_y_m": 0.0,
                          "expected_selection_mode": ("v16_specialist_locked" if forced else
                                                      "v21_scoped_y_locked")})
    plan = {
        "schema": "robot-learning-lab.cove50-v94-gaze-route-variance-plan.v1",
        "role": "paired_prescribed_specialist_route_variation_not_automatic_control",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": teacher_plan["source_sha256"],
        "teacher_plan_sha256": sha(teacher_plan_path),
        "teacher_matrix_sha256": sha(teacher_matrix_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "pair_count": len(VARIATIONS), "cases": cases,
        "hypothesis": ("V94 forced specialist route restores the alternate-gaze miss "
                       "without changing goal Y. Small toy and motor variations should "
                       "test whether this is a regional contact repair."),
        "change": ("Pair natural scoped versus prescribed specialist routes on four "
                   "same-gaze conditions: ±0.15 mm toy X and 10/12 V with 0/10 ms delay."),
        "gate": ("Complete all eight fresh native visits. Require four actual scoped-to-"
                 "specialist paired route changes with at least three baseline-failure "
                 "to full-gate specialist rescues, all specialist visits >=1.2 s "
                 "continuous contact, and no passing baseline lost. This is teacher "
                 "development, not an automatic controller."),
    }
    plan_path = OUT / "gaze-route-variance-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V94 variation plan or source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    dest = OUT / "native-gaze-route-variance"
    dest.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        iterator = iter(cases)
        pending = {}
        for _ in range(2):
            case = next(iterator, None)
            if case is not None:
                pending[pool.submit(native.run_case, case, plan, dest)] = case
        while pending:
            future = next(as_completed(tuple(pending)))
            case = pending.pop(future)
            row = future.result()
            if row["selection_mode"] != case["expected_selection_mode"]:
                row["faults"].append("observed route differs from frozen expectation")
                row["pass"] = False
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in cases if item["id"] in rows]
            pairs = []
            for name, *_ in VARIATIONS:
                scoped = rows.get(f"{name}-scoped")
                specialist = rows.get(f"{name}-specialist")
                if scoped and specialist:
                    pairs.append({"condition": name,
                                  "scoped_route": scoped["selection_mode"],
                                  "specialist_route": specialist["selection_mode"],
                                  "scoped_contact_s": scoped["contact_s"],
                                  "specialist_contact_s": specialist["contact_s"],
                                  "scoped_pass": scoped["pass"],
                                  "specialist_pass": specialist["pass"]})
            complete = len(ordered) == len(cases)
            rescues = sum(not p["scoped_pass"] and p["specialist_pass"] for p in pairs)
            gate = (complete and len(pairs) == len(VARIATIONS) and rescues >= 3 and
                    all(p["scoped_route"] == "v21_scoped_y_locked" and
                        p["specialist_route"] == "v16_specialist_locked" and
                        p["specialist_pass"] and p["specialist_contact_s"] >= 1.2 and
                        (not p["scoped_pass"] or p["specialist_pass"])
                        for p in pairs))
            matrix = {"schema": "robot-learning-lab.cove50-v94-gaze-route-variance.v1",
                      "role": plan["role"], "body_role": plan["body_role"],
                      "plan_sha256": sha(plan_path), "complete": complete,
                      "case_count": len(ordered),
                      "pass_count": sum(item["pass"] for item in ordered),
                      "pair_count": len(pairs), "paired_rescue_count": rescues,
                      "diagnostic_gate_met": gate, "pairs": pairs, "cases": ordered}
            temp = dest / "matrix.json.tmp"
            temp.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temp, dest / "matrix.json")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "selection_mode",
                               "diagnostic_force_specialist", "faults")}), flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, dest)] = next_case


if __name__ == "__main__":
    main()
