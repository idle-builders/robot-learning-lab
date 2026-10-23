"""Test whether opened contact teachers transfer across varied RGB entries."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as automatic,
    run_v108_native_case as shifted,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v109"
PLAN = OUT / "varied-specialist-transfer-plan.json"
DEST = OUT / "native-varied-specialist-transfer"
V101_PLAN = BASE / "v101/disjoint-reach-plan.json"
V105_PLAN = BASE / "v105/varied-entry-exposure-plan.json"
V105_REPORTS = {
    19011: "e9d1dafa7590abef7f6c8406ce32ba40447baf51910e4001ea2a0431c5700cfb",
    19020: "6630d43b9c4a228675e404435e2c7565d3bbd8ad495a7d28e474d7e7b1091e33",
}
SHIFTS = (0.0, -.001, -.002, -.003)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    score = json.loads(V101_PLAN.read_text())
    entry = json.loads(V105_PLAN.read_text())
    if (sha(V105_PLAN) !=
            "a60172d1b927351f88f740bba70cf1e6439f193f49974f05cc1dcfae1577fadb" or
            score["source_sha256"]["evaluator"] != sha(automatic.EVALUATOR)):
        raise ValueError("Expected unchanged V95 evaluator and V105 entry plan")
    sources = {}
    for label, runner in (("automatic", automatic), ("shifted", shifted)):
        sources[label] = {**score["source_sha256"],
                          "evaluator": sha(runner.EVALUATOR),
                          "runner": sha(Path(runner.__file__))}
    entry_cases = {case["seed"]: case for case in entry["cases"]}
    entry_refs = []
    cases = []
    for seed, expected_hash in V105_REPORTS.items():
        report_path = BASE / f"v105/native-varied-entry-exposure/seed-{seed}-entry-{seed}.json"
        if sha(report_path) != expected_hash:
            raise ValueError(f"V105 entry report changed for {seed}")
        report = json.loads(report_path.read_text())
        if (not report["preflight_only"] or
                report["selection_mode"] != "v16_specialist_locked" or
                report["selected_pointer_sha256"] != sources["automatic"]["pointer"]):
            raise ValueError(f"Expected opened natural-specialist entry {seed}")
        entry_refs.append({"seed": seed, "path": str(report_path.relative_to(ROOT)),
                           "sha256": expected_hash,
                           "entry_error_xy_m": report["selection_features"]["entry_error_xy_m"],
                           "camera_xyz_m": report["selection_features"]["entry_camera_world_xyz_m"]})
        base = {key: value for key, value in entry_cases[seed].items()
                if key != "id" and not key.startswith("expected_")}
        for shift in SHIFTS:
            label = "zero" if shift == 0 else f"neg{abs(round(shift * 1000))}"
            cases.append({**base, "id": f"transfer-{seed}-{label}",
                          "diagnostic_goal_shift_body_y_m": shift,
                          "expected_selection_mode": "v16_specialist_locked",
                          "entry_report_sha256": expected_hash})
    plan = {
        "schema": "robot-learning-lab.cove50-v109-specialist-transfer-plan.v1",
        "role": "opened_varied_full_native_teacher_transfer_not_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v101_source_plan_sha256": sha(V101_PLAN),
        "v105_entry_plan_sha256": sha(V105_PLAN),
        "opened_entry_reports": entry_refs,
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "method": ("Use two V105-opened natural-specialist RGB entries, one "
                   "pitch-low front-camera and one roll-high rear-camera start. "
                   "For each physical seed and condition, compare the unchanged "
                   "V95 automatic full native route with -1, -2 and -3 mm "
                   "post-selection body-Y target shifts. Require the same observed "
                   "specialist route and full RGB, contact, support, motor, action "
                   "and collision gates. These are development, not score starts."),
        "decision_gate": ("A useful teacher must rescue a failed natural baseline "
                          "with at least 1.2 s contact while retaining success "
                          "across both varied camera sides. Otherwise refine the "
                          "region/data hypothesis before training. All new actor "
                          "or router candidates need retention and new disjoint "
                          "full native scoring."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V109 plan or source changed")
    PLAN.write_bytes(encoded)
    return plan


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    result = {
        "schema": "robot-learning-lab.cove50-v109-varied-specialist-transfer.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
        "case_count": len(ordered),
        "pass_count": sum(row["pass"] for row in ordered),
        "cases": ordered,
    }
    temporary = DEST / "matrix.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, DEST / "matrix.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        raise ValueError("Use one or two native evaluators")
    plan = freeze()
    print(f"frozen {PLAN} sha256={sha(PLAN)}", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    old = DEST / "matrix.json"
    existing = json.loads(old.read_text()) if old.exists() else None
    if existing and existing["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V109 matrix belongs to another plan")
    rows = {row["id"]: row for row in existing["cases"]} if existing else {}

    def run_case(case: dict) -> dict:
        diagnostic = case["diagnostic_goal_shift_body_y_m"] != 0
        runner = shifted if diagnostic else automatic
        source = plan["source_sha256"]["shifted" if diagnostic else "automatic"]
        row = runner.run_case(case, {"source_sha256": source}, DEST)
        report = json.loads((ROOT / row["result"]).read_text())
        if row["selection_mode"] != "v16_specialist_locked":
            raise ValueError("V109 route differs from opened entry")
        row.update({"diagnostic_goal_shift_body_y_m":
                    case["diagnostic_goal_shift_body_y_m"],
                    "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
                    "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
                    "toy_offset_x_m": case["toy_offset_x_m"],
                    "toy_offset_y_m": case["toy_offset_y_m"],
                    "entry_error_xy_m": report["selection_features"]["entry_error_xy_m"]})
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case): case for case in plan["cases"]
                   if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            write_matrix(plan, rows)
            print(json.dumps({key: row[key] for key in
                              ("id", "contact_s", "pass", "faults")}), flush=True)
    matrix = write_matrix(plan, rows)
    print(json.dumps({key: matrix[key] for key in
                      ("case_count", "pass_count")}), flush=True)


if __name__ == "__main__":
    main()
