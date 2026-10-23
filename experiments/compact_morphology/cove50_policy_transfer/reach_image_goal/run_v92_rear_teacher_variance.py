"""Test the V90 contact teacher around its actual rear-view failure region."""

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
OUT = BASE / "v92"
VARIATIONS = (
    ("toy-x-plus015mm", .00615, .006, 11, 10, -1.5),
    ("toy-x-minus015mm", .00585, .006, 11, 10, -1.5),
    ("rail10-delay0", .006, .006, 10, 0, -1.5),
    ("rail12-delay10", .006, .006, 12, 10, -1.5),
    ("gaze-minus1p3", .006, .006, 11, 10, -1.3),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rear_high_y_exposed(report_path: Path) -> bool:
    features = json.loads(report_path.read_text())["selection_features"]
    x, y = features["selected_image_error_xy_m"]
    camera_x, camera_y, _ = features["entry_camera_world_xyz_m"]
    return (0 <= x < .0015 and .006 <= y < .0065 and
            -.05 <= camera_x < 0 and camera_y < -.08)


def main() -> None:
    teacher_plan_path = BASE / "v90/goal-shift-sweep-plan.json"
    teacher_matrix_path = BASE / "v90/native-goal-shift-sweep/matrix.json"
    broad_matrix_path = BASE / "v91/native-neighbor-teacher/matrix.json"
    teacher_plan = json.loads(teacher_plan_path.read_text())
    teacher_matrix = json.loads(teacher_matrix_path.read_text())
    broad_matrix = json.loads(broad_matrix_path.read_text())
    if (not teacher_matrix["complete"] or not broad_matrix["complete"] or
            broad_matrix["diagnostic_gate_met"] or
            not any(row["id"] == "prescribed-y-minus1mm" and row["pass"]
                    for row in teacher_matrix["cases"])):
        raise ValueError("Need completed V90 teacher and failed broad-neighbor V91 diagnostic")
    anchor = next(case for case in teacher_plan["cases"] if case["id"] == "prescribed-y-zero")
    hashes = teacher_plan["source_sha256"]
    if hashes["evaluator"] != sha(native.EVALUATOR) or hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json"):
        raise ValueError("Frozen evaluator or mechanical pointer changed")
    cases = []
    for name, toy_x, toy_y, rail, delay, gaze in VARIATIONS:
        for shift, label in ((0.0, "zero"), (-.001, "minus1mm")):
            cases.append({**anchor, "id": f"rear-{name}-{label}",
                          "toy_offset_x_m": toy_x, "toy_offset_y_m": toy_y,
                          "rail_v": rail, "delay_ms": delay, "gaze_head_yaw_rad": gaze,
                          "prior_role": "opened_rear_view_teacher_variation",
                          "baseline_contact_s": None,
                          "diagnostic_goal_shift_body_y_m": shift})
    plan = {
        "schema": "robot-learning-lab.cove50-v92-rear-teacher-variance-plan.v1",
        "role": "paired_prescribed_rear_view_teacher_not_automatic_control",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": hashes,
        "v90_plan_sha256": sha(teacher_plan_path),
        "v90_matrix_sha256": sha(teacher_matrix_path),
        "v91_matrix_sha256": sha(broad_matrix_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "pair_count": len(VARIATIONS), "cases": cases,
        "hypothesis": ("V91 new seeds mostly entered other camera/route regions, so they did "
                       "not test the V90 rear-camera high-Y failure. Small toy and motor "
                       "variations around that view should reveal whether -1 mm generalizes."),
        "change": ("Pair zero versus prescribed -1 mm body-Y goal shift on five opened "
                   "conditions from the same 10433 start: ±0.15 mm toy X, 10/12 V with "
                   "0/10 ms delay, and a second physical gaze. Preserve evaluator and actor."),
        "rear_view_observation_gate": ("Selected RGB error X in [0,1.5) mm, Y in [6,6.5) mm, "
                                       "camera world X in [-50,0) mm and Y below -80 mm."),
        "gate": ("Complete all ten native visits; require all shifted visits full-gate "
                 "pass with >=1.2 s contact and no baseline pass lost. At least two "
                 "zero-to-pass rescues must occur on paired visits where the same "
                 "observed rear-high-Y branch activates. This is teacher development only."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "rear-teacher-variance-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V92 plan or source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    dest = OUT / "native-rear-teacher-variance"
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
            row["observed_rear_high_y"] = rear_high_y_exposed(ROOT / row["result"])
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in cases if item["id"] in rows]
            pairs = []
            for name, *_ in VARIATIONS:
                zero = rows.get(f"rear-{name}-zero")
                shifted = rows.get(f"rear-{name}-minus1mm")
                if zero and shifted:
                    pairs.append({"condition": name, "zero_contact_s": zero["contact_s"],
                                  "shift_contact_s": shifted["contact_s"],
                                  "zero_pass": zero["pass"], "shift_pass": shifted["pass"],
                                  "zero_route": zero["selection_mode"],
                                  "shift_route": shifted["selection_mode"],
                                  "zero_rear_high_y": zero["observed_rear_high_y"],
                                  "shift_rear_high_y": shifted["observed_rear_high_y"]})
            complete = len(ordered) == len(cases)
            rescues = sum(not pair["zero_pass"] and pair["shift_pass"] and
                          pair["zero_rear_high_y"] and pair["shift_rear_high_y"] and
                          pair["zero_route"] == pair["shift_route"] == "v16_specialist_locked"
                          for pair in pairs)
            gate = (complete and len(pairs) == len(VARIATIONS) and rescues >= 2 and
                    all(pair["shift_pass"] and pair["shift_contact_s"] >= 1.2 and
                        (not pair["zero_pass"] or pair["shift_pass"])
                        for pair in pairs))
            matrix = {"schema": "robot-learning-lab.cove50-v92-rear-teacher-variance.v1",
                      "role": plan["role"], "body_role": plan["body_role"],
                      "plan_sha256": sha(plan_path), "complete": complete,
                      "case_count": len(ordered), "pass_count": sum(item["pass"] for item in ordered),
                      "pair_count": len(pairs), "rear_view_rescue_count": rescues,
                      "diagnostic_gate_met": gate, "pairs": pairs, "cases": ordered}
            temp = dest / "matrix.json.tmp"
            temp.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temp, dest / "matrix.json")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "selection_mode",
                               "diagnostic_goal_shift_body_y_m", "observed_rear_high_y", "faults")}), flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, dest)] = next_case


if __name__ == "__main__":
    main()
