"""Collect supported-route opened variance with both target directions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as baseline,
    run_v108_native_case as shifted,
    run_v139_reserved_contact as score_runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
SCORE_PLAN = BASE / "v139/reserved-contact-score-plan.json"
SCORE_MATRIX = BASE / "v139/native-reserved-contact/matrix.json"
DIAG = BASE / "v141/native-opened-failure-shifts/matrix.json"
DIRECTION = BASE / "v142/valid-visits-and-launch-failure.json"
REJECTED_PLAN = BASE / "v143/opened-opposite-direction-variance-plan.json"
REJECTION = BASE / "v143/preflight-rejection.json"
OUT = BASE / "v144"
PLAN = OUT / "supported-opposite-direction-variance-plan.json"
DEST = OUT / "native-supported-opposite-direction-variance"
ANCHORS = (24111, 24119)
SHIFTS = (-.002, .002)
PERTURBATIONS = (
    (-.0004, .0, 12, 10, -1.5, .0, .0),
    (+.0004, .0, 11, 5, -1.3, .0, .0),
    (.0, -.0004, 10, 10, -1.5, .0, .0),
    (.0, +.0004, 12, 0, -1.3, .0, .0),
    (-.0002, +.0002, 11, 10, -1.3, +.005, -.005),
    (+.0002, -.0002, 10, 5, -1.5, -.005, +.005),
    (-.0004, -.0004, 12, 5, -1.3, +.01, +.01),
    (+.0004, +.0004, 11, 0, -1.5, -.01, -.01),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict, *, frozen: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2) + "\n"
    if frozen and path.exists() and path.read_text() != encoded:
        raise ValueError(f"Frozen V144 plan changed: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(encoded)
    os.replace(temp, path)


def freeze() -> dict:
    pointer = ROOT / "robot/ape-current-design.json"
    if json.loads(pointer.read_text())["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical rejected control sandbox only")
    source_plan = json.loads(SCORE_PLAN.read_text())
    score = json.loads(SCORE_MATRIX.read_text())
    diag = json.loads(DIAG.read_text())
    direction = json.loads(DIRECTION.read_text())
    rejection = json.loads(REJECTION.read_text())
    if (not score["complete"] or score["score_gate_met"] or
            score["plan_sha256"] != sha(SCORE_PLAN) or
            not diag["complete"] or diag["case_count"] != 2 or
            direction["status"] != "stopped_evaluator_range_rejection" or
            direction["valid_supported_count"] != 5 or
            rejection["status"] != "rejected_before_contact" or
            rejection["plan_sha256"] != sha(REJECTED_PLAN) or
            rejection["invalid_case_count"] != 4 or
            source_plan["source_sha256"]["body"] !=
            sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml") or
            source_plan["source_sha256"]["pointer"] != sha(pointer) or
            source_plan["source_sha256"]["evaluator"] != sha(baseline.EVALUATOR) or
            source_plan["source_sha256"]["shift_evaluator"] != sha(shifted.EVALUATOR)):
        raise ValueError("Failed score, opened diagnosis or native source changed")
    score_rows = {row["seed"]: row for row in score["cases"]}
    original = {case["seed"]: case for case in source_plan["cases"]}
    if (score_rows[24111]["candidate_contact_s"] != .58 or
            score_rows[24119]["candidate_contact_s"] != 1.0 or
            {row["seed"] for row in diag["cases"]} != set(ANCHORS) or
            not any(row["shift_body_y_m"] == .002 and row["contact_s"] == 2.54
                    for row in direction["valid_visits"])):
        raise ValueError("Opened failure anchors differ from saved evidence")
    cases = []
    for seed in ANCHORS:
        anchor = original[seed]
        for index, (dx, dy, voltage, delay, gaze, dpitch, droll) in enumerate(
                PERTURBATIONS, start=1):
            if seed == 24111 and index >= 5:
                droll = (-.005, -.01, -.005, -.01)[index - 5]
            if seed == 24119 and index >= 5:
                dpitch = (+.005, +.01, +.005, +.01)[index - 5]
            cases.append({
                "id": f"opened-{seed}-supported-{index:02d}",
                "seed": seed,
                "cohort": "opened_score_failure_physical_perturbation",
                "anchor_opened_failure_seed": seed,
                "route_variant": anchor["route_variant"],
                "rail_v": voltage, "delay_ms": delay,
                "toy_offset_x_m": round(anchor["toy_offset_x_m"] + dx, 6),
                "toy_offset_y_m": round(anchor["toy_offset_y_m"] + dy, 6),
                "pitch_offset_rad": round(anchor["pitch_offset_rad"] + dpitch, 6),
                "roll_clearance_rad": round(anchor["roll_clearance_rad"] + droll, 6),
                "gaze_head_yaw_rad": gaze,
                "condition_delta_xy_mm": [round(dx * 1000, 3),
                                          round(dy * 1000, 3)],
                "condition_delta_pitch_roll_rad": [dpitch, droll],
            })
    if len(cases) != 16 or len({case["id"] for case in cases}) != 16:
        raise ValueError("Expected 16 unique physical conditions")
    for case in cases:
        if not (.095 - 1e-9 <= case["pitch_offset_rad"] <= .115 + 1e-9 and
                -.26 - 1e-9 <= case["roll_clearance_rad"] <= -.24 + 1e-9):
            raise ValueError(f"Unsupported articulation in {case['id']}")
    plan = {
        "schema": "robot-learning-lab.cove50-v144-opposite-direction-variance-plan.v1",
        "role": "post_score_opened_paired_training_variance_not_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**source_plan["source_sha256"],
                          "score_plan": sha(SCORE_PLAN),
                          "score_matrix": sha(SCORE_MATRIX),
                          "opened_diagnosis": sha(DIAG),
                          "direction_diagnostic": sha(DIRECTION),
                          "rejected_preflight_plan": sha(REJECTED_PLAN),
                          "rejected_preflight_record": sha(REJECTION),
                          "runner": sha(Path(__file__))},
        "case_count": 16, "visit_count": 48,
        "shifts_body_y_m": list(SHIFTS),
        "method": ("After V143 preflight rejection, freeze eight supported "
                   "physical perturbations around each of two "
                   "opened V139 misses before new contact. Vary toy X/Y, "
                   "motor voltage, delay, gaze and pitch/roll articulation. "
                   "Replay every condition with unchanged native observed-RGB "
                   "reach and prescribed −2/+2 mm post-route body-Y targets, "
                   "keeping all failures and 1.2 s margins. Seed and toy "
                   "truth are audit-only, never actor inputs. These repeated "
                   "seed variations are training data, not 16 independent "
                   "successes or a new-body score."),
        "cases": cases,
    }
    write_json(PLAN, plan, frozen=True)
    return plan


def run_pair(case: dict, plan: dict) -> dict:
    source = plan["source_sha256"]
    base = baseline.run_case({**case, "id": case["id"] + "-zero"},
                             {"source_sha256": source}, DEST)
    base_report = json.loads((ROOT / base["result"]).read_text())
    base_features = score_runner.observations.observed_features(base_report)
    choices = {}
    for label, shift in (("neg2", -.002), ("pos2", .002)):
        result = shifted.run_case(
            {**case, "id": case["id"] + "-" + label,
             "diagnostic_goal_shift_body_y_m": shift},
            {"source_sha256": {**source,
                               "evaluator": source["shift_evaluator"]}}, DEST)
        report = json.loads((ROOT / result["result"]).read_text())
        features = score_runner.observations.observed_features(report)
        faults = score_runner.entry_parity(base_features, features)
        if (report["seed"] != base_report["seed"] or
                report["body_sha256"] != base_report["body_sha256"] or
                report["selected_pointer_sha256"] !=
                base_report["selected_pointer_sha256"] or
                report["selection_mode"] != base_report["selection_mode"]):
            faults.append("paired native source or automatic route mismatch")
        choices[label] = {
            "shift_body_y_m": shift,
            "contact_s": result["contact_s"],
            "contact_delta_s": round(result["contact_s"] - base["contact_s"], 3),
            "native_pass": result["pass"],
            "pass": result["pass"] and not faults,
            "margin_1p2_s_met": result["margin_1p2_s_met"],
            "entry_parity_faults": faults,
            "faults": result["faults"] + faults,
            "result": result["result"], "trace": result["trace"],
        }
    return {
        "id": case["id"], "seed": case["seed"],
        "cohort": case["cohort"],
        "anchor_opened_failure_seed": case["anchor_opened_failure_seed"],
        "route_variant": case["route_variant"],
        "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
        "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
        "toy_offset_x_m": case["toy_offset_x_m"],
        "toy_offset_y_m": case["toy_offset_y_m"],
        "pitch_offset_rad": case["pitch_offset_rad"],
        "roll_clearance_rad": case["roll_clearance_rad"],
        "selection_mode": base["selection_mode"],
        "entry_observed_features": base_features,
        "baseline_contact_s": base["contact_s"],
        "baseline_pass": base["pass"],
        "baseline_margin_1p2_s_met": base["margin_1p2_s_met"],
        "baseline_faults": base["faults"],
        "baseline_result": base["result"], "baseline_trace": base["trace"],
        "choices": choices,
    }


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    matrix = {
        "schema": "robot-learning-lab.cove50-v144-opposite-direction-variance.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "complete": len(ordered) == plan["case_count"],
        "case_count": len(ordered), "planned_count": plan["case_count"],
        "baseline_pass_count": sum(row["baseline_pass"] for row in ordered),
        "neg2_pass_count": sum(row["choices"]["neg2"]["pass"] for row in ordered),
        "pos2_pass_count": sum(row["choices"]["pos2"]["pass"] for row in ordered),
        "neg2_margin_count": sum(row["choices"]["neg2"]["margin_1p2_s_met"]
                                 for row in ordered),
        "pos2_margin_count": sum(row["choices"]["pos2"]["margin_1p2_s_met"]
                                 for row in ordered),
        "neg2_rescue_count": sum(not row["baseline_pass"] and
                                 row["choices"]["neg2"]["pass"] for row in ordered),
        "pos2_rescue_count": sum(not row["baseline_pass"] and
                                 row["choices"]["pos2"]["pass"] for row in ordered),
        "neg2_regression_count": sum(row["baseline_pass"] and
                                     not row["choices"]["neg2"]["pass"]
                                     for row in ordered),
        "pos2_regression_count": sum(row["baseline_pass"] and
                                     not row["choices"]["pos2"]["pass"]
                                     for row in ordered),
        "entry_parity_failure_count": sum(
            bool(choice["entry_parity_faults"])
            for row in ordered for choice in row["choices"].values()),
        "cases": ordered,
    }
    write_json(DEST / "matrix.json", matrix)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        raise ValueError("Use one or two paired native evaluators")
    plan = freeze()
    print(f"V144 plan sha256={sha(PLAN)} {plan['case_count']} conditions, "
          f"{plan['visit_count']} visits", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "matrix.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if old and old["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V144 matrix belongs to another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_pair, case, plan): case
                   for case in plan["cases"] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V144 {row['id']}: unchanged {row['baseline_contact_s']:.2f}, "
                  f"−2 {row['choices']['neg2']['contact_s']:.2f}, "
                  f"+2 {row['choices']['pos2']['contact_s']:.2f}s; "
                  f"{matrix['case_count']}/{plan['case_count']}", flush=True)


if __name__ == "__main__":
    main()
