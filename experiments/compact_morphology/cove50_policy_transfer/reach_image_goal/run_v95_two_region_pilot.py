"""Pilot both observed rear-camera reach corrections in fresh native replay."""

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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    v93_plan_path = BASE / "v93/observed-bias-pilot-plan.json"
    v93_matrix_path = BASE / "v93/native-observed-bias-pilot/matrix.json"
    v94_plan_path = BASE / "v94b/conditional-route-variance-plan.json"
    v94_matrix_path = BASE / "v94b/native-conditional-route-variance/matrix.json"
    v93_plan = json.loads(v93_plan_path.read_text())
    v93 = json.loads(v93_matrix_path.read_text())
    v94_plan = json.loads(v94_plan_path.read_text())
    v94 = json.loads(v94_matrix_path.read_text())
    scoped = [pair for pair in v94["pairs"]
              if pair["baseline_route"] == "v21_scoped_y_locked"]
    natural = [pair for pair in v94["pairs"]
               if pair["baseline_route"] != "v21_scoped_y_locked"]
    teacher_support = (v94["paired_rescue_count"] >= 2 and
                       all(pair["forced_attempted"] and pair["forced_pass"] and
                           pair["forced_contact_s"] >= 1.2 for pair in scoped) and
                       all(pair["baseline_pass"] and pair["baseline_contact_s"] >= 1.2
                           for pair in natural))
    if (not v93["complete"] or not v93["development_gate_met"] or
            not v94["complete"] or not teacher_support or
            v93["plan_sha256"] != sha(v93_plan_path) or
            v94["plan_sha256"] != sha(v94_plan_path)):
        raise ValueError("Need source-matched V93 gate and complete V94b paired teacher evidence")
    cases = []
    for earlier in v93_plan["cases"]:
        case = dict(earlier)
        alternate = case["role"] == "known_out_of_region_failure"
        case["expected_alternate_gaze_specialist"] = alternate
        if alternate:
            case["expected_selection_mode"] = "v16_specialist_locked"
            case["role"] = "opened_alternate_gaze_rescue"
        cases.append(case)
    nearby = {pair["condition"]: pair for pair in v94["pairs"]}
    for earlier in v94_plan["baseline_cases"]:
        observed = nearby[earlier["condition"]]
        scoped = observed["baseline_route"] == "v21_scoped_y_locked"
        case = {key: value for key, value in earlier.items()
                if key not in ("diagnostic_force_specialist",
                               "diagnostic_goal_shift_body_y_m")}
        case.update(id="alternate-" + earlier["id"],
                    role=("opened_alternate_gaze_variation" if scoped else
                          "natural_specialist_boundary_control" if
                          observed["baseline_route"] == "v16_specialist_locked" else
                          "nearby_off_region_control"),
                    expected_selection_mode=("v16_specialist_locked" if scoped else
                                             observed["baseline_route"]),
                    expected_rear_contact_bias=False,
                    expected_alternate_gaze_specialist=scoped)
        cases.append(case)
    if len(cases) != 14:
        raise ValueError("Expected eight V93 cases and six V94b varied conditions")
    hashes = dict(v93_plan["source_sha256"])
    hashes.update({"evaluator": sha(native.EVALUATOR),
                   "runner": sha(Path(native.__file__))})
    if hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json"):
        raise ValueError("Mechanical pointer changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v95-two-region-pilot-plan.v1",
        "role": "automatic_observed_rear_two_region_development",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": hashes,
        "v93_plan_sha256": sha(v93_plan_path),
        "v93_matrix_sha256": sha(v93_matrix_path),
        "v94_plan_sha256": sha(v94_plan_path),
        "v94_matrix_sha256": sha(v94_matrix_path),
        "v94_diagnostic_gate_met": v94["diagnostic_gate_met"],
        "v94_paired_rescue_count": v94["paired_rescue_count"],
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "expected_alternate_route_use_count": 1 + v94["scoped_pair_count"],
        "hypothesis": ("The V93 observed −1 mm rear-contact correction and the V94b "
                       "specialist-route teacher repair different camera/image-error "
                       "regions without altering actor weights or sensor input. V94b failed "
                       "its frozen three-rescue diagnostic because only two natural baselines "
                       "missed; this trial therefore requires every scoped forced visit to "
                       "pass, including the already passing controls, and scores the new "
                       "automatic rule independently."),
        "change": ("Add an observed specialist route for entry RGB error X [1.5,2.0) mm, "
                   "Y [4.5,5.5) mm, camera X [−50,0) mm, Y below −80 mm and at least "
                   "40 orange pixels. Preserve the V93 high-Y goal correction."),
        "gate": ("All fourteen native visits must pass full contact, RGB, support, "
                 "motor and collision gates with >=1.2 s contact. Five high-Y "
                 "targets must activate only the V93 goal correction; the original "
                 "alternate-gaze miss and every naturally scoped V94b neighbor must "
                 "activate only the new specialist route. Naturally specialist "
                 "neighbors and two prior passing controls must activate neither. "
                 "This is opened development on the rejected historical body."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "two-region-pilot-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V95 plan or source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    dest = OUT / "native-two-region-pilot"
    dest.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        iterator = iter(cases)
        pending = {}
        for _ in range(4):
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
            gate = (complete and
                    all(item["pass"] and item["contact_s"] >= 1.2 for item in ordered) and
                    sum(item["automatic_rear_contact_bias_applied"] for item in ordered) == 5 and
                    sum(item["automatic_alternate_gaze_specialist_applied"]
                        for item in ordered) == plan["expected_alternate_route_use_count"] and
                    not any(item["automatic_rear_contact_bias_applied"] and
                            item["automatic_alternate_gaze_specialist_applied"]
                            for item in ordered))
            matrix = {"schema": "robot-learning-lab.cove50-v95-two-region-pilot.v1",
                      "role": plan["role"], "body_role": plan["body_role"],
                      "plan_sha256": sha(plan_path), "complete": complete,
                      "case_count": len(ordered),
                      "pass_count": sum(item["pass"] for item in ordered),
                      "rear_bias_use_count": sum(item["automatic_rear_contact_bias_applied"]
                                                 for item in ordered),
                      "alternate_route_use_count": sum(
                          item["automatic_alternate_gaze_specialist_applied"]
                          for item in ordered),
                      "development_gate_met": gate, "cases": ordered}
            temp = dest / "matrix.json.tmp"
            temp.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temp, dest / "matrix.json")
            print(json.dumps({key: row[key] for key in
                              ("id", "role", "pass", "contact_s", "selection_mode",
                               "automatic_rear_contact_bias_applied",
                               "automatic_alternate_gaze_specialist_applied", "faults")}),
                  flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, dest)] = next_case


if __name__ == "__main__":
    main()
