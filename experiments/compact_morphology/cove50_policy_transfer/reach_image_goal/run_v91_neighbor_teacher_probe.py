"""Paired native neighborhood probe for V90's opened -1 mm contact teacher."""

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
OUT = BASE / "v91"
NEIGHBORS = (
    (16001, .0060, .0065, 10, 0, -1.3),
    (16002, .0060, .0055, 12, 10, -1.5),
    (16003, .0065, .0060, 11, 0, -1.5),
    (16004, .0055, .0060, 10, 10, -1.3),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    teacher_plan_path = BASE / "v90/goal-shift-sweep-plan.json"
    teacher_matrix_path = BASE / "v90/native-goal-shift-sweep/matrix.json"
    teacher_plan = json.loads(teacher_plan_path.read_text())
    teacher_matrix = json.loads(teacher_matrix_path.read_text())
    if (not teacher_matrix["complete"] or
            not any(row["id"] == "prescribed-y-minus1mm" and row["pass"] and
                    row["margin_1p2_s_met"] for row in teacher_matrix["cases"])):
        raise ValueError("Need a completed V90 sweep with the -1 mm full-gate teacher")
    anchor = next(case for case in teacher_plan["cases"] if case["id"] == "prescribed-y-zero")
    source_hashes = teacher_plan["source_sha256"]
    if (source_hashes["evaluator"] != sha(native.EVALUATOR) or
            source_hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json")):
        raise ValueError("Frozen evaluator or mechanical design pointer changed")
    cases = []
    for seed, toy_x, toy_y, rail, delay, gaze in NEIGHBORS:
        for shift, label in ((0.0, "baseline"), (-.001, "minus1mm")):
            cases.append({**anchor, "id": f"neighbor-{seed}-{label}", "seed": seed,
                          "toy_offset_x_m": toy_x, "toy_offset_y_m": toy_y,
                          "rail_v": rail, "delay_ms": delay,
                          "gaze_head_yaw_rad": gaze,
                          "prior_role": "opened_new_neighbor",
                          "baseline_contact_s": None,
                          "diagnostic_goal_shift_body_y_m": shift})
    plan = {
        "schema": "robot-learning-lab.cove50-v91-neighbor-teacher-plan.v1",
        "role": "paired_prescribed_goal_teacher_variation_not_automatic_control",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": source_hashes,
        "v90_plan_sha256": sha(teacher_plan_path),
        "v90_matrix_sha256": sha(teacher_matrix_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "pair_count": len(NEIGHBORS), "cases": cases,
        "hypothesis": ("The -1 mm body-Y teacher repairs a narrow fingertip contact gap "
                       "across nearby visible toy positions and motor/gaze starts, not just seed 10433."),
        "change": ("Pair zero and -1 mm prescribed body-Y shifts on four new development "
                   "seeds with ±0.5 mm toy XY, 10-12 V rail, 0-10 ms delay and two "
                   "physical gaze settings. Hold all other source, actor and RGB code fixed."),
        "gate": ("Complete all eight native visits. Each -1 mm teacher must pass the full "
                 "gate with at least 1.2 s contact and preserve its baseline branch. "
                 "Require at least two zero-to-pass paired rescues, at least two actual "
                 "specialist-route uses, and no baseline passing visit lost. This is a "
                 "development teacher gate; no automatic controller or held-out score."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "neighbor-teacher-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V91 plan or source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    dest = OUT / "native-neighbor-teacher"
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
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in cases if item["id"] in rows]
            pairs = []
            for seed, *_ in NEIGHBORS:
                prior = rows.get(f"neighbor-{seed}-baseline")
                teacher = rows.get(f"neighbor-{seed}-minus1mm")
                if prior and teacher:
                    pairs.append({"seed": seed, "baseline_contact_s": prior["contact_s"],
                                  "teacher_contact_s": teacher["contact_s"],
                                  "baseline_route": prior["selection_mode"],
                                  "teacher_route": teacher["selection_mode"],
                                  "baseline_pass": prior["pass"],
                                  "teacher_pass": teacher["pass"]})
            complete = len(ordered) == len(cases)
            diagnostic_gate_met = (complete and len(pairs) == len(NEIGHBORS) and
                                   all(pair["teacher_pass"] and pair["teacher_contact_s"] >= 1.2 and
                                       pair["baseline_route"] == pair["teacher_route"] and
                                       (not pair["baseline_pass"] or pair["teacher_pass"])
                                       for pair in pairs) and
                                   sum(not pair["baseline_pass"] and pair["teacher_pass"]
                                       for pair in pairs) >= 2 and
                                   sum(pair["teacher_route"] == "v16_specialist_locked"
                                       for pair in pairs) >= 2)
            matrix = {"schema": "robot-learning-lab.cove50-v91-neighbor-teacher.v1",
                      "role": plan["role"], "body_role": plan["body_role"],
                      "plan_sha256": sha(plan_path), "complete": complete,
                      "case_count": len(ordered), "pass_count": sum(item["pass"] for item in ordered),
                      "pair_count": len(pairs), "diagnostic_gate_met": diagnostic_gate_met,
                      "pairs": pairs, "cases": ordered}
            temporary = dest / "matrix.json.tmp"
            temporary.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temporary, dest / "matrix.json")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "selection_mode",
                               "diagnostic_goal_shift_body_y_m", "faults")}), flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, dest)] = next_case


if __name__ == "__main__":
    main()
