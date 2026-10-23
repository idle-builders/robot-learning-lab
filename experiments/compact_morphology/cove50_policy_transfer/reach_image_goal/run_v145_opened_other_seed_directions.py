"""Check both target directions on other opened V139 roll-high/pitch-low seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v108_native_case as shifted,
    run_v139_reserved_contact as score_runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
SCORE_PLAN = BASE / "v139/reserved-contact-score-plan.json"
SCORE_MATRIX = BASE / "v139/native-reserved-contact/matrix.json"
V144_PLAN = BASE / "v144/supported-opposite-direction-variance-plan.json"
OUT = BASE / "v145"
PLAN = OUT / "other-opened-seed-direction-plan.json"
DEST = OUT / "native-other-opened-seed-directions"
SEEDS = (24105, 24106, 24107, 24108, 24109, 24110, 24112,
         24114, 24117, 24122, 24124)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict, *, frozen: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2) + "\n"
    if frozen and path.exists() and path.read_text() != encoded:
        raise ValueError(f"Frozen V145 plan changed: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(encoded)
    os.replace(temp, path)


def freeze() -> dict:
    pointer = ROOT / "robot/ape-current-design.json"
    if json.loads(pointer.read_text())["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical rejected control sandbox only")
    source_plan = json.loads(SCORE_PLAN.read_text())
    score = json.loads(SCORE_MATRIX.read_text())
    v144 = json.loads(V144_PLAN.read_text())
    source = source_plan["source_sha256"]
    if (not score["complete"] or score["score_gate_met"] or
            score["plan_sha256"] != sha(SCORE_PLAN) or
            v144["case_count"] != 16 or
            source["body"] != sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml") or
            source["pointer"] != sha(pointer) or
            source["shift_evaluator"] != sha(shifted.EVALUATOR)):
        raise ValueError("Saved failed score or source identity differs")
    score_rows = {row["seed"]: row for row in score["cases"]}
    originals = {case["seed"]: case for case in source_plan["cases"]}
    if ({case["seed"] for case in v144["cases"]} & set(SEEDS) or
            len(SEEDS) != len(set(SEEDS)) or
            set(SEEDS) - originals.keys()):
        raise ValueError("V145 seeds must be distinct opened V139 starts")
    cases = []
    for seed in SEEDS:
        original, row = originals[seed], score_rows[seed]
        if original["route_variant"] not in ("roll-high", "pitch-low"):
            raise ValueError(f"Unexpected route on opened seed {seed}")
        existing = {
            "baseline_result": row["baseline_result"],
            "baseline_trace": row["baseline_trace"],
        }
        if row["frozen_decision"]["shift_body_y_m"] == -.002:
            existing.update({"neg2_result": row["candidate_result"],
                             "neg2_trace": row["candidate_trace"]})
        cases.append({
            "id": f"opened-{seed}-other-direction", "seed": seed,
            "cohort": "opened_other_seed_direction_retention",
            "route_variant": original["route_variant"],
            "rail_v": original["rail_v"], "delay_ms": original["delay_ms"],
            "toy_offset_x_m": original["toy_offset_x_m"],
            "toy_offset_y_m": original["toy_offset_y_m"],
            "pitch_offset_rad": original["pitch_offset_rad"],
            "roll_clearance_rad": original["roll_clearance_rad"],
            "gaze_head_yaw_rad": original["gaze_head_yaw_rad"],
            "existing_source_sha256": {key: sha(ROOT / path)
                                       for key, path in existing.items()},
            "existing_source_paths": existing,
        })
    count_new = sum(1 + ("neg2_result" not in case["existing_source_paths"])
                    for case in cases)
    if len(cases) != 11 or count_new != 16:
        raise ValueError("Expected 11 opened starts and 16 new native visits")
    plan = {
        "schema": "robot-learning-lab.cove50-v145-other-opened-direction-plan.v1",
        "role": "opened_other_seed_target_direction_training_retention_not_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**source, "score_plan": sha(SCORE_PLAN),
                          "score_matrix": sha(SCORE_MATRIX),
                          "v144_plan": sha(V144_PLAN),
                          "runner": sha(Path(__file__))},
        "case_count": 11, "new_native_visit_count": 16,
        "reused_native_visit_count": 17,
        "target_shifts_body_y_m": [0, -.002, .002],
        "method": ("Before viewing complete V144 outcomes, freeze every other "
                   "opened V139 roll-high or pitch-low seed. Reuse source-hashed "
                   "V139 unchanged and existing −2 mm native visits; run only "
                   "the missing −2 mm and all +2 mm visits. Compare same "
                   "physical start and controlled RGB entry. Include full "
                   "contact, support, motor, collision, action and margin "
                   "failures. This checks direction transfer and older-start "
                   "retention across seed groups; it is not independent score."),
        "cases": cases,
    }
    write_json(PLAN, plan, frozen=True)
    return plan


def checked_existing(case: dict, label: str, expected_shift: float,
                     source: dict) -> tuple[dict, dict]:
    paths, hashes = case["existing_source_paths"], case["existing_source_sha256"]
    result_path, trace_path = paths[f"{label}_result"], paths[f"{label}_trace"]
    if (sha(ROOT / result_path) != hashes[f"{label}_result"] or
            sha(ROOT / trace_path) != hashes[f"{label}_trace"]):
        raise ValueError(f"Existing native source changed: {case['id']} {label}")
    report = json.loads((ROOT / result_path).read_text())
    if (report["seed"] != case["seed"] or
            report["body_sha256"] != source["body"] or
            report["selected_pointer_sha256"] != source["pointer"] or
            abs(report.get("diagnostic_goal_shift_body_y_m", 0) - expected_shift) > 1e-9):
        raise ValueError(f"Existing native report changed: {case['id']} {label}")
    return report, {"result": result_path, "trace": trace_path,
                    "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"]}


def run_pair(case: dict, plan: dict) -> dict:
    source = plan["source_sha256"]
    score = next(row for row in json.loads(SCORE_MATRIX.read_text())["cases"]
                 if row["seed"] == case["seed"])
    base_report, base = checked_existing(case, "baseline", 0, source)
    observed = score_runner.observations.observed_features(base_report)
    if (observed != score["native_entry_features"] or
            base["contact_s"] != score["baseline_contact_s"]):
        raise ValueError(f"Existing V139 entry/contact differs: {case['id']}")
    choices = {}
    for label, shift in (("neg2", -.002), ("pos2", .002)):
        if label == "neg2" and "neg2_result" in case["existing_source_paths"]:
            report, prior = checked_existing(case, label, shift, source)
            result = {"contact_s": prior["contact_s"],
                      "pass": score["candidate_native_pass"],
                      "margin_1p2_s_met": score["candidate_margin_1p2_s_met"],
                      "faults": score["candidate_faults"],
                      "result": prior["result"], "trace": prior["trace"]}
            reused = True
        else:
            result = shifted.run_case(
                {**case, "id": case["id"] + "-" + label,
                 "diagnostic_goal_shift_body_y_m": shift},
                {"source_sha256": {**source,
                                   "evaluator": source["shift_evaluator"]}}, DEST)
            report = json.loads((ROOT / result["result"]).read_text())
            reused = False
        faults = score_runner.entry_parity(
            observed, score_runner.observations.observed_features(report))
        if (report["seed"] != base_report["seed"] or
                report["body_sha256"] != base_report["body_sha256"] or
                report["selected_pointer_sha256"] !=
                base_report["selected_pointer_sha256"] or
                report["selection_mode"] != base_report["selection_mode"]):
            faults.append("paired native source or automatic route mismatch")
        choices[label] = {
            "shift_body_y_m": shift,
            "reused_from_v139": reused,
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
        "cohort": case["cohort"], "route_variant": case["route_variant"],
        "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
        "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
        "selection_mode": base_report["selection_mode"],
        "entry_observed_features": observed,
        "baseline_contact_s": base["contact_s"],
        "baseline_pass": score["baseline_native_pass"],
        "baseline_margin_1p2_s_met": base["contact_s"] >= 1.2,
        "baseline_faults": score["baseline_faults"],
        "baseline_result": base["result"], "baseline_trace": base["trace"],
        "choices": choices,
    }


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    matrix = {
        "schema": "robot-learning-lab.cove50-v145-other-opened-direction.v1",
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
        "neg2_regression_count": sum(row["baseline_pass"] and
                                     not row["choices"]["neg2"]["pass"]
                                     for row in ordered),
        "pos2_regression_count": sum(row["baseline_pass"] and
                                     not row["choices"]["pos2"]["pass"]
                                     for row in ordered),
        "entry_parity_failure_count": sum(bool(choice["entry_parity_faults"])
                                          for row in ordered
                                          for choice in row["choices"].values()),
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
    print(f"V145 plan sha256={sha(PLAN)} {plan['case_count']} opened seeds, "
          f"{plan['new_native_visit_count']} new visits", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "matrix.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if old and old["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V145 matrix belongs to another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_pair, case, plan): case
                   for case in plan["cases"] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V145 {row['id']}: unchanged {row['baseline_contact_s']:.2f}, "
                  f"−2 {row['choices']['neg2']['contact_s']:.2f}, "
                  f"+2 {row['choices']['pos2']['contact_s']:.2f}s; "
                  f"{matrix['case_count']}/{plan['case_count']}", flush=True)


if __name__ == "__main__":
    main()
