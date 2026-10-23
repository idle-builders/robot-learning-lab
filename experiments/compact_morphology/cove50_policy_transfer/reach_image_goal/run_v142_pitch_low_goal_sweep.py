"""Map opened pitch-low contact against prescribed body-Y target shifts."""

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
OPENED_PLAN = BASE / "v141/opened-score-failure-plan.json"
OPENED_MATRIX = BASE / "v141/native-opened-failure-shifts/matrix.json"
OUT = BASE / "v142"
PLAN = OUT / "opened-pitch-low-y-sweep-plan.json"
DEST = OUT / "native-opened-pitch-low-y-sweep"
SEED = 24119
SHIFT_MM = (-4, -3, -1, 1, 2, 3, 4)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict, *, frozen: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2) + "\n"
    if frozen and path.exists() and path.read_text() != encoded:
        raise ValueError(f"Frozen V142 sweep changed: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(encoded)
    os.replace(temp, path)


def freeze() -> dict:
    score_plan = json.loads(SCORE_PLAN.read_text())
    score = json.loads(SCORE_MATRIX.read_text())
    opened_plan = json.loads(OPENED_PLAN.read_text())
    opened = json.loads(OPENED_MATRIX.read_text())
    if (not score["complete"] or score["score_gate_met"] or
            not opened["complete"] or opened["case_count"] != 2 or
            score["plan_sha256"] != sha(SCORE_PLAN) or
            opened["plan_sha256"] != sha(OPENED_PLAN)):
        raise ValueError("Exact failed score and complete opened diagnosis required")
    original = next(case for case in score_plan["cases"] if case["seed"] == SEED)
    score_row = next(row for row in score["cases"] if row["seed"] == SEED)
    opened_case = next(case for case in opened_plan["cases"] if case["seed"] == SEED)
    opened_row = next(row for row in opened["cases"] if row["seed"] == SEED)
    if (original["route_variant"] != "pitch-low" or
            score_row["selection_mode"] != "v16_specialist_locked" or
            original["frozen_decision"]["shift_body_y_m"] != 0 or
            score_row["candidate_contact_s"] != 1.0 or
            opened_row["diagnostic_contact_s"] != .98 or
            opened_row["diagnostic_pass"] or
            opened_case["source_baseline_sha256"] !=
            sha(ROOT / score_row["baseline_result"])):
        raise ValueError("Opened pitch-low failure identity changed")
    cases = [{**original,
              "id": f"opened-{SEED}-body-y-{shift:+d}mm",
              "diagnostic_goal_shift_body_y_m": shift / 1000,
              "expected_selection_mode": score_row["selection_mode"]}
             for shift in SHIFT_MM]
    plan = {
        "schema": "robot-learning-lab.cove50-v142-opened-pitch-low-sweep-plan.v1",
        "role": "post_score_opened_goal_direction_diagnostic_not_policy",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**score_plan["source_sha256"],
                          "evaluator": score_plan["source_sha256"]["shift_evaluator"],
                          "score_plan": sha(SCORE_PLAN),
                          "score_matrix": sha(SCORE_MATRIX),
                          "opened_plan": sha(OPENED_PLAN),
                          "opened_matrix": sha(OPENED_MATRIX),
                          "runner": sha(Path(__file__))},
        "baseline_result": score_row["baseline_result"],
        "baseline_result_sha256": sha(ROOT / score_row["baseline_result"]),
        "baseline_trace_sha256": sha(ROOT / score_row["baseline_trace"]),
        "baseline_contact_s": score_row["baseline_contact_s"],
        "prior_neg2_result": opened_row["diagnostic_result"],
        "prior_neg2_result_sha256": sha(ROOT / opened_row["diagnostic_result"]),
        "prior_neg2_contact_s": opened_row["diagnostic_contact_s"],
        "method": ("After V139's failed 1.2 s margin and V141's harmful "
                   "−2 mm visit, source-freeze seven other signed body-Y "
                   "target shifts on the same opened specialist pitch-low "
                   "start. Compare full native gates, contact duration and "
                   "observed RGB entry to the exact unchanged baseline. "
                   "This maps one local response curve, not a learned "
                   "decision, reserve score or selected-body result."),
        "case_count": len(cases), "cases": cases,
    }
    write_json(PLAN, plan, frozen=True)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    native = shifted.run_case(case, {"source_sha256": plan["source_sha256"]}, DEST)
    baseline = json.loads((ROOT / plan["baseline_result"]).read_text())
    candidate = json.loads((ROOT / native["result"]).read_text())
    base_features = score_runner.observations.observed_features(baseline)
    features = score_runner.observations.observed_features(candidate)
    faults = score_runner.entry_parity(base_features, features)
    if (candidate["seed"] != baseline["seed"] or
            candidate["body_sha256"] != baseline["body_sha256"] or
            candidate["selected_pointer_sha256"] !=
            baseline["selected_pointer_sha256"] or
            candidate["selection_mode"] != baseline["selection_mode"] or
            candidate["diagnostic_goal_shift_body_y_m"] !=
            case["diagnostic_goal_shift_body_y_m"]):
        faults.append("native paired source identity mismatch")
    return {
        "id": case["id"], "seed": case["seed"],
        "route_variant": case["route_variant"],
        "selection_mode": native["selection_mode"],
        "shift_body_y_m": case["diagnostic_goal_shift_body_y_m"],
        "baseline_contact_s": plan["baseline_contact_s"],
        "contact_s": native["contact_s"],
        "contact_delta_s": round(native["contact_s"] -
                                 plan["baseline_contact_s"], 3),
        "native_pass": native["pass"],
        "pass": native["pass"] and not faults,
        "margin_1p2_s_met": native["margin_1p2_s_met"],
        "entry_parity_faults": faults,
        "faults": native["faults"] + faults,
        "result": native["result"], "trace": native["trace"],
    }


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"]
               if case["id"] in rows]
    matrix = {
        "schema": "robot-learning-lab.cove50-v142-opened-pitch-low-sweep.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "complete": len(ordered) == plan["case_count"],
        "case_count": len(ordered), "planned_count": plan["case_count"],
        "pass_count": sum(row["pass"] for row in ordered),
        "margin_count": sum(row["margin_1p2_s_met"] for row in ordered),
        "entry_parity_failure_count": sum(bool(row["entry_parity_faults"])
                                          for row in ordered),
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
        raise ValueError("Use one or two native evaluators")
    plan = freeze()
    print(f"V142 plan sha256={sha(PLAN)} opened shifts={plan['case_count']}",
          flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "matrix.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if old and old["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V142 matrix belongs to another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case
                   for case in plan["cases"] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V142 {row['id']}: {row['contact_s']:.2f}s "
                  f"{'pass' if row['pass'] else 'fail'}; "
                  f"{matrix['case_count']}/{plan['case_count']}", flush=True)


if __name__ == "__main__":
    main()
