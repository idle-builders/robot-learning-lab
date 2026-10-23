"""Separate specialist-route and goal-shift teachers on the alternate-gaze miss."""

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
PROBES = (("scoped-zero", False, 0.0),
          ("scoped-minus2mm", False, -.002),
          ("specialist-zero", True, 0.0),
          ("specialist-minus1mm", True, -.001))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    v92_plan_path = BASE / "v92/rear-teacher-variance-plan.json"
    v92_matrix_path = BASE / "v92/native-rear-teacher-variance/matrix.json"
    v93_pilot_path = BASE / "v93/native-observed-bias-pilot/matrix.json"
    v92_plan = json.loads(v92_plan_path.read_text())
    v92 = json.loads(v92_matrix_path.read_text())
    v93 = json.loads(v93_pilot_path.read_text())
    if (not v92["complete"] or not v93["complete"] or
            v92["plan_sha256"] != sha(v92_plan_path) or
            not any(row["id"] == "rear-gaze-minus1p3-zero" and not row["pass"]
                    for row in v93["cases"])):
        raise ValueError("Need the retained V92/V93 alternate-gaze failure")
    anchor = next(case for case in v92_plan["cases"]
                  if case["id"] == "rear-gaze-minus1p3-zero")
    cases = [{**anchor, "id": label, "diagnostic_force_specialist": forced,
              "diagnostic_goal_shift_body_y_m": shift,
              "expected_selection_mode": ("v16_specialist_locked" if forced else
                                          "v21_scoped_y_locked")}
             for label, forced, shift in PROBES]
    hashes = dict(v92_plan["source_sha256"])
    hashes.update({"evaluator": sha(native.EVALUATOR),
                   "runner": sha(Path(native.__file__))})
    if hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json"):
        raise ValueError("Selected mechanical pointer changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v94-alternate-gaze-diagnostic-plan.v1",
        "role": "prescribed_route_and_goal_teacher_not_automatic_control",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": hashes,
        "v92_plan_sha256": sha(v92_plan_path),
        "v92_matrix_sha256": sha(v92_matrix_path),
        "v93_pilot_matrix_sha256": sha(v93_pilot_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "hypothesis": ("The alternate gaze improves image-goal error but selects the "
                       "scoped-Y actor; neither zero nor −1 mm added shift contacted. "
                       "The miss may be actor route rather than goal localization."),
        "change": ("Replay the same observed camera, toy, motor and entry with the "
                   "scoped actor at zero/−2 mm additional goal shift and an explicitly "
                   "forced specialist actor at zero/−1 mm additional shift."),
        "gate": ("Complete four fresh native visits with route and source identities "
                 "checked. Report full RGB, support, motor, collision and fingertip "
                 "contact gates. A passing teacher only motivates new observed-rule "
                 "development and never qualifies automatic control."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "alternate-gaze-route-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V94 plan or source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    dest = OUT / "native-alternate-gaze-route"
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
                row["faults"].append("observed route differs from frozen diagnostic expectation")
                row["pass"] = False
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in cases if item["id"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v94-alternate-gaze-diagnostic.v1",
                      "role": plan["role"], "body_role": plan["body_role"],
                      "plan_sha256": sha(plan_path),
                      "complete": len(ordered) == len(cases),
                      "case_count": len(ordered),
                      "pass_count": sum(item["pass"] for item in ordered),
                      "teacher_found": any(item["pass"] and item["contact_s"] >= 1.2
                                           for item in ordered),
                      "cases": ordered}
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
