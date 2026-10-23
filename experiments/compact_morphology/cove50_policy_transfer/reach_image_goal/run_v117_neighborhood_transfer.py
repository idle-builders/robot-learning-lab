"""Test a local contact teacher across five different observed routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v108_native_case as shifted,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V113_PLAN = BASE / "v113/varied-paired-contact-data-plan.json"
V115_PLAN = BASE / "v115/opened-contact-slope-plan.json"
V115_MATRIX = BASE / "v115/native-opened-contact-slope/matrix.json"
OUT = BASE / "v117"
PLAN = OUT / "neighborhood-transfer-plan.json"
DEST = OUT / "native-neighborhood-transfer"
ROUTES = ("center", "pitch-low", "pitch-high", "roll-low", "roll-high")
SHIFT_M = -.003


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def freeze() -> dict:
    if not 0 < abs(SHIFT_M) <= .003:
        raise ValueError("Evaluator accepts only nonzero offsets within ±3 mm")
    source = json.loads(V113_PLAN.read_text())
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("V117 applies only to the historical rejected sandbox")
    if source["source_sha256"]["shift_evaluator"] != sha(shifted.EVALUATOR):
        raise ValueError("Source target evaluator changed")
    slope_plan = json.loads(V115_PLAN.read_text())
    slope_matrix = json.loads(V115_MATRIX.read_text())
    if slope_matrix["plan_sha256"] != sha(V115_PLAN):
        raise ValueError("V115 teacher source does not match its plan")
    teacher = next((row for row in slope_matrix["cases"]
                    if row["seed"] == 19036 and row["id"] == "slope-19036--3mm"), None)
    if not teacher or not teacher["pass"] or not teacher["margin_1p2_s_met"]:
        raise ValueError("V115 −3 mm contact teacher is not retained")
    anchor = next(case for case in source["cases"] if case["seed"] == 19036)
    ax, ay = anchor["source_entry_camera_xy_mm"]
    choices = []
    for route in ROUTES:
        candidates = [case for case in source["cases"]
                      if case["seed"] != 19036 and case["route_variant"] == route]
        if not candidates:
            raise ValueError(f"No source entry for route {route}")
        candidate = min(candidates, key=lambda case: (
            math.hypot(case["source_entry_camera_xy_mm"][0] - ax,
                       case["source_entry_camera_xy_mm"][1] - ay),
            case["seed"]))
        distance = math.hypot(candidate["source_entry_camera_xy_mm"][0] - ax,
                              candidate["source_entry_camera_xy_mm"][1] - ay)
        if distance > 35:
            raise ValueError("Selected route is outside 35 mm camera neighborhood")
        choices.append({**candidate,
                        "id": f"neighbor-{candidate['seed']}-neg3",
                        "diagnostic_goal_shift_body_y_m": SHIFT_M,
                        "camera_distance_from_19036_mm": round(distance, 3)})
    if len({case["seed"] for case in choices}) != 5:
        raise ValueError("Neighborhood routes reused a physical start")
    entry_refs = {ref["id"]: ref for ref in source["v105_entry_reports"]}
    refs = [entry_refs[case["source_entry_id"]] for case in choices]
    for ref in refs:
        if sha(ROOT / ref["path"]) != ref["sha256"]:
            raise ValueError("V105 observed entry source changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v117-neighborhood-transfer-plan.v1",
        "role": "opened_neighborhood_contact_teacher_transfer_not_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**source["source_sha256"],
                          "evaluator": sha(shifted.EVALUATOR),
                          "runner": sha(Path(__file__))},
        "v113_training_plan_sha256": sha(V113_PLAN),
        "v115_teacher_plan_sha256": sha(V115_PLAN),
        "v115_teacher_report": teacher["result"],
        "v115_teacher_report_sha256": sha(ROOT / teacher["result"]),
        "selected_entry_reports": refs,
        "selection": ("After V115 opened 19036 at −3 mm passed, take the "
                      "closest V113 preselected RGB entry camera position "
                      "to 19036 for each of five distinct route commands, "
                      "excluding 19036. Selection sees only frozen V113 "
                      "plan entry camera coordinates and route commands, "
                      "never these five starts' contact results."),
        "method": ("Replay only the prescribed −3 mm post-route body-Y "
                   "target on each exact V113 planned physical start. "
                   "The independent V113 zero and −2 mm visits remain "
                   "paired references when completed. Retain full native "
                   "RGB, fingertip contact, support, rated-effort, action "
                   "and collision evidence."),
        "case_count": 5,
        "cases": choices,
        "qualification": ("Five opened neighborhood visits can test whether "
                          "the local teacher transfers across route and "
                          "motor variance. It does not train a selector, "
                          "replace the sealed V116 contact holdout or qualify "
                          "the new mechanical body."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V117 plan or source changed")
    PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    print(f"frozen {PLAN} sha256={sha(PLAN)}", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "matrix.json"
    existing = json.loads(path.read_text()) if path.exists() else None
    if existing and existing["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V117 matrix uses another plan")
    rows = {row["id"]: row for row in existing["cases"]} if existing else {}
    run_plan = {"source_sha256": plan["source_sha256"]}
    for case in plan["cases"]:
        if case["id"] in rows:
            continue
        rows[case["id"]] = shifted.run_case(case, run_plan, DEST)
        ordered = [rows[c["id"]] for c in plan["cases"] if c["id"] in rows]
        matrix = {
            "schema": "robot-learning-lab.cove50-v117-neighborhood-transfer.v1",
            "role": plan["role"], "body_role": plan["body_role"],
            "plan_sha256": sha(PLAN), "case_count": len(ordered),
            "complete": len(ordered) == len(plan["cases"]),
            "pass_count": sum(row["pass"] for row in ordered),
            "margin_1p2_s_count": sum(row["margin_1p2_s_met"] for row in ordered),
            "cases": ordered,
        }
        atomic_json(path, matrix)
        row = rows[case["id"]]
        print(json.dumps({"id": row["id"], "contact_s": row["contact_s"],
                          "pass": row["pass"],
                          "margin_1p2_s_met": row["margin_1p2_s_met"]}), flush=True)


if __name__ == "__main__":
    main()
