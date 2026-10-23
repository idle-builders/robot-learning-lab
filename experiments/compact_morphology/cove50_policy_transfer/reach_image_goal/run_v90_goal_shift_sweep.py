"""Opened native teacher sweep after V89 specialist routing still missed 10433."""

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
    run_v90_native_case as native,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v90"
SHIFTS_M = (0.0, .001, .002, .003, -.001, -.002, -.003)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_path = BASE / "v89/rear-boundary-probe-plan.json"
    failure_path = BASE / "v89/native-rear-boundary-probe/seed-10433-prior-10433.json"
    source = json.loads(source_path.read_text())
    failure = json.loads(failure_path.read_text())
    if (failure["selection_mode"] != "v16_specialist_locked" or
            failure["phases"]["reach"]["continuous_finger_contact_s"] != 0.0):
        raise ValueError("Need the observed V89 specialist contact failure")
    anchor = next(case for case in source["cases"] if case["seed"] == 10433)
    cases = []
    for value in SHIFTS_M:
        label = "zero" if value == 0 else f"{'plus' if value > 0 else 'minus'}{abs(value)*1000:g}mm"
        cases.append({**anchor, "id": f"prescribed-y-{label}",
                      "baseline_contact_s": 0.0,
                      "diagnostic_goal_shift_body_y_m": value})
    hashes = {**source["source_sha256"], "evaluator": sha(native.EVALUATOR)}
    if hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json"):
        raise ValueError("Mechanical design pointer changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v90-goal-shift-sweep-plan.v1",
        "role": "prescribed_body_y_goal_teacher_not_automatic_control",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": hashes,
        "v89_probe_plan_sha256": sha(source_path),
        "v89_10433_failure_report_sha256": sha(failure_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "hypothesis": "On 10433 the specialist actor tracks its commanded goal, but the "
                      "fingertip misses by a narrow physical margin after the newer "
                      "preposition. A small body-Y goal shift may restore contact.",
        "change": "Prescribe 0, ±1, ±2 and ±3 mm body-Y RGB goal shifts after the same "
                  "V89 observed route choice; keep actor, RGB observation, toy, motor "
                  "and body fixed. The shift is an unavailable teacher parameter, not "
                  "an automatic rule or deployable policy.",
        "gate": "Complete all seven exact native visits, retain full RGB/contact/safety "
                "reports and traces, and identify whether any shifted visit holds "
                "at least 1.2 s contact without a safety fault. A single rescue is "
                "only teacher evidence; neighboring starts must still be checked.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "goal-shift-sweep-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V90 teacher plan/source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    dest = OUT / "native-goal-shift-sweep"
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
            row["diagnostic_goal_shift_body_y_m"] = case["diagnostic_goal_shift_body_y_m"]
            if row["selection_mode"] != "v16_specialist_locked":
                row["faults"].append("specialist arm route changed during teacher sweep")
                row["pass"] = False
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in cases if item["id"] in rows]
            complete = len(ordered) == len(cases)
            matrix = {
                "schema": "robot-learning-lab.cove50-v90-goal-shift-sweep.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(plan_path), "complete": complete,
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "teacher_rescue_found": (complete and any(
                    item["pass"] and item["margin_1p2_s_met"] and
                    item["diagnostic_goal_shift_body_y_m"] != 0 for item in ordered)),
                "cases": ordered,
            }
            temp = dest / "matrix.json.tmp"
            temp.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temp, dest / "matrix.json")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "selection_mode",
                               "diagnostic_goal_shift_body_y_m", "faults")}),
                  flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, dest)] = next_case


if __name__ == "__main__":
    main()
