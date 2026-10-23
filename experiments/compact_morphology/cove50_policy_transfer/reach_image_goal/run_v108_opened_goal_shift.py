"""Source-freeze a paired body-Y contact-target sweep on opened failures."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as baseline,
    run_v108_native_case as native,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v108"
PLAN = OUT / "opened-goal-shift-plan.json"
DEST = OUT / "native-opened-goal-shift"
SCORE_PLAN = BASE / "v101/disjoint-reach-plan.json"
SCORE_MATRIX = BASE / "v101/native-disjoint/matrix.json"
SEEDS = (22021, 22028, 22031)
SHIFTS = (-.003, -.002, -.001, .001, .002)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    scored_plan = json.loads(SCORE_PLAN.read_text())
    scored = json.loads(SCORE_MATRIX.read_text())
    if (not scored["complete"] or scored["score_gate_met"] or
            scored["plan_sha256"] != sha(SCORE_PLAN) or
            tuple(row["seed"] for row in scored["cases"] if not row["pass"]) != SEEDS):
        raise ValueError("Expected complete, failed V101 score")
    if scored_plan["source_sha256"]["evaluator"] != sha(baseline.EVALUATOR):
        raise ValueError("V101 baseline evaluator changed")
    source = scored_plan["source_sha256"].copy()
    source.update(evaluator=sha(native.EVALUATOR), runner=sha(Path(native.__file__)))
    score_cases = {case["seed"]: case for case in scored_plan["cases"]}
    score_rows = {row["seed"]: row for row in scored["cases"]}
    cases = []
    for seed in SEEDS:
        base = {key: value for key, value in score_cases[seed].items()
                if key not in ("id", "baseline_contact_s") and
                not key.startswith("expected_")}
        for shift in SHIFTS:
            label = f"{'neg' if shift < 0 else 'pos'}{abs(round(shift * 1000))}"
            cases.append({**base, "id": f"shift-{seed}-{label}",
                          "diagnostic_goal_shift_body_y_m": shift,
                          "baseline_contact_s": score_rows[seed]["contact_s"],
                          "baseline_selection_mode": score_rows[seed]["selection_mode"]})
    plan = {
        "schema": "robot-learning-lab.cove50-v108-opened-goal-shift-plan.v1",
        "role": "opened_postscore_contact_teacher_search_not_disjoint_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": source,
        "baseline_evaluator_sha256": sha(baseline.EVALUATOR),
        "baseline_plan_sha256": sha(SCORE_PLAN),
        "baseline_matrix_sha256": sha(SCORE_MATRIX),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "seeds": SEEDS,
        "diagnostic_goal_shift_body_y_m": SHIFTS,
        "cases": cases,
        "method": ("On each already opened V101 contact-failure seed, preserve the "
                   "physical case and automatic observed-RGB route. Apply a bounded "
                   "extra body-frame Y goal shift only after route selection. "
                   "Measure full native RGB, contact, support, rated effort, action "
                   "and collision without a skill reset. Reuse V101's zero-shift "
                   "report as the baseline; never revise its failed score."),
        "decision_gate": ("Use any 1.2 s full-gate contact success only as a "
                          "prescribed teacher candidate. Train from opened native "
                          "visits with varied starts, then run unchanged prior "
                          "retention and wholly new disjoint starts before promotion."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V108 plan or sources changed")
    PLAN.write_bytes(encoded)
    return plan


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    result = {
        "schema": "robot-learning-lab.cove50-v108-opened-goal-shift.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
        "case_count": len(ordered),
        "pass_count": sum(row["pass"] for row in ordered),
        "margin_1p2_s_count": sum(row["margin_1p2_s_met"] for row in ordered),
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
        raise ValueError("Existing V108 matrix uses another plan")
    rows = {row["id"]: row for row in existing["cases"]} if existing else {}

    def run_case(case: dict) -> dict:
        row = native.run_case(case, plan, DEST)
        report = json.loads((ROOT / row["result"]).read_text())
        if row["selection_mode"] != case["baseline_selection_mode"]:
            raise ValueError("Target shift changed the automatic entry route")
        row.update({"baseline_selection_mode": case["baseline_selection_mode"],
                    "baseline_contact_s": case["baseline_contact_s"],
                    "diagnostic_goal_shift_body_y_m":
                    case["diagnostic_goal_shift_body_y_m"],
                    "contact_delta_s": round(row["contact_s"] -
                                             case["baseline_contact_s"], 3),
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
                              ("id", "selection_mode", "contact_s", "pass", "faults")}),
                  flush=True)
    matrix = write_matrix(plan, rows)
    print(json.dumps({key: matrix[key] for key in
                      ("case_count", "pass_count", "margin_1p2_s_count")}), flush=True)


if __name__ == "__main__":
    main()
