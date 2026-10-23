"""Pilot the observed rear-view contact correction on native paired evidence."""

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
    run_v93_native_case as native,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v93"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    v90_plan_path = BASE / "v90/goal-shift-sweep-plan.json"
    v92_plan_path = BASE / "v92/rear-teacher-variance-plan.json"
    v92_matrix_path = BASE / "v92/native-rear-teacher-variance/matrix.json"
    v89_plan_path = BASE / "v89/rear-boundary-probe-plan.json"
    v45_control_path = BASE / "v45/combined-screen-plan.json"
    v90 = json.loads(v90_plan_path.read_text())
    v92 = json.loads(v92_plan_path.read_text())
    matrix = json.loads(v92_matrix_path.read_text())
    v89 = json.loads(v89_plan_path.read_text())
    if (not matrix["complete"] or matrix["diagnostic_gate_met"] or
            matrix["plan_sha256"] != sha(v92_plan_path)):
        raise ValueError("Need completed, source-matched failed V92 diagnostic")
    pairs = {pair["condition"]: pair for pair in matrix["pairs"]}
    in_region = ("toy-x-plus015mm", "toy-x-minus015mm", "rail10-delay0", "rail12-delay10")
    if any(pairs[name]["zero_pass"] or not pairs[name]["shift_pass"] for name in in_region):
        raise ValueError("V93 region needs the four observed paired rescues")
    anchor = next(case for case in v90["cases"] if case["id"] == "prescribed-y-zero")
    anchor = {key: value for key, value in anchor.items()
              if key != "diagnostic_goal_shift_body_y_m"}
    cases = [{**anchor, "id": "rear-anchor-10433", "role": "opened_rear_failure",
              "expected_rear_contact_bias": True}]
    for name in in_region:
        earlier = next(case for case in v92["cases"] if case["id"] == f"rear-{name}-zero")
        case = {key: value for key, value in earlier.items()
                if key != "diagnostic_goal_shift_body_y_m"}
        cases.append({**case, "role": "opened_paired_rear_variation",
                      "expected_rear_contact_bias": True})
    control = next(case for case in v89["cases"] if case["seed"] == 10514)
    cases.append({**control, "id": "prior-control-10514", "role": "passing_prior_control",
                  "expected_rear_contact_bias": False})
    control = next(case for case in json.loads(v45_control_path.read_text())["cases"]
                   if case["seed"] == 10831)
    # The V45 passing control was run on the same center route as the anchor.
    cases.append({**{key: value for key, value in anchor.items()
                     if key != "expected_selection_mode"}, **control,
                  "id": "prior-control-10831", "role": "passing_prior_control",
                  "expected_rear_contact_bias": False})
    gaze = next(case for case in v92["cases"] if case["id"] == "rear-gaze-minus1p3-zero")
    cases.append({**{key: value for key, value in gaze.items()
                     if key != "diagnostic_goal_shift_body_y_m"},
                  "role": "known_out_of_region_failure", "expected_rear_contact_bias": False,
                  "expected_selection_mode": "v21_scoped_y_locked"})
    sources = native.v83.v16.SOURCES
    hashes = dict(v90["source_sha256"])
    hashes.update({"evaluator": sha(native.EVALUATOR),
                   "body": sha(sources["body"]),
                   "actor": sha(native.v83.v57.ACTOR),
                   "selector": sha(native.v83.SELECTOR),
                   "template": sha(native.v83.screen.core.TEMPLATE),
                   "pointer": sha(ROOT / "robot/ape-current-design.json"),
                   "runner": sha(Path(native.__file__))})
    plan = {
        "schema": "robot-learning-lab.cove50-v93-observed-bias-pilot-plan.v1",
        "role": "automatic_observed_rear_contact_bias_development",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": hashes,
        "v90_plan_sha256": sha(v90_plan_path),
        "v92_plan_sha256": sha(v92_plan_path),
        "v92_matrix_sha256": sha(v92_matrix_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "hypothesis": ("The V90 and four V92 paired rear-camera rescues support a bounded "
                       "observable -1 mm goal correction. The alternate gaze enters a different "
                       "route and is a retained out-of-region failure."),
        "change": ("Apply -1 mm body-Y automatically only when the selected specialist "
                   "route has entry RGB error X [0,1.5) mm, Y [5.8,6.5) mm, rear "
                   "camera X [-50,0) mm and Y below -80 mm, and at least 40 orange pixels."),
        "gate": ("All five in-region targets and both passing controls must clear the "
                 "full native gate with >=1.2 s contact. Bias exposure must match the "
                 "frozen expected values for all eight cases. Retain the alternate-gaze "
                 "out-of-region outcome even if it fails. This pilot is not body qualification."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "observed-bias-pilot-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V93 plan or source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    dest = OUT / "native-observed-bias-pilot"
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
            row["role"] = case["role"]
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in cases if item["id"] in rows]
            complete = len(ordered) == len(cases)
            gate = (complete and all(item["automatic_rear_contact_bias_applied"] ==
                                     item["expected_rear_contact_bias"] for item in ordered) and
                    all(item["pass"] and item["contact_s"] >= 1.2 for item in ordered
                        if item["role"] != "known_out_of_region_failure"))
            matrix_out = {"schema": "robot-learning-lab.cove50-v93-observed-bias-pilot.v1",
                          "role": plan["role"], "body_role": plan["body_role"],
                          "plan_sha256": sha(plan_path), "complete": complete,
                          "case_count": len(ordered),
                          "pass_count": sum(item["pass"] for item in ordered),
                          "in_region_pass_count": sum(item["pass"] for item in ordered
                                                      if item["role"] != "known_out_of_region_failure"),
                          "development_gate_met": gate, "cases": ordered}
            temp = dest / "matrix.json.tmp"
            temp.write_text(json.dumps(matrix_out, indent=2) + "\n")
            os.replace(temp, dest / "matrix.json")
            print(json.dumps({key: row[key] for key in
                              ("id", "role", "pass", "contact_s", "selection_mode",
                               "automatic_rear_contact_bias_applied", "faults")}), flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, dest)] = next_case


if __name__ == "__main__":
    main()
