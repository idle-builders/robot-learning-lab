"""Stage the target correction against the weakest older native reach starts."""

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
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v111"
PLAN = OUT / "weak-prior-shift-pilot-plan.json"
DEST = OUT / "native-weak-prior-shift-pilot"
V97_PLAN = BASE / "v97/prior75-retention-plan.json"
V97_MATRIX = BASE / "v97/native-prior75-retention/matrix.json"
V110_PLAN = BASE / "v110/varied-shift-retention-plan.json"
V110_MATRIX = BASE / "v110/native-varied-shift-retention/matrix.json"
SHIFT_M = -.002
COUNT = 12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rows(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (row["contact_s"], row["seed"]))
    selected = ordered[:7]
    seen = {row["id"] for row in selected}

    def add(row: dict) -> None:
        if row["id"] not in seen:
            selected.append(row)
            seen.add(row["id"])

    for mode in sorted({row["selection_mode"] for row in rows}):
        add(next(row for row in ordered if row["selection_mode"] == mode))
    rear = [row for row in ordered if row["automatic_rear_contact_bias_applied"]]
    if rear:
        add(rear[0])
    for row in ordered:
        if len(selected) >= COUNT:
            break
        add(row)
    if len(selected) != COUNT:
        raise ValueError("Could not select twelve distinct weak prior starts")
    return selected


def freeze() -> dict:
    prior_plan = json.loads(V97_PLAN.read_text())
    prior = json.loads(V97_MATRIX.read_text())
    varied = json.loads(V110_MATRIX.read_text())
    if (not prior["complete"] or prior["case_count"] != 75 or
            prior["pass_count"] != 75 or prior["plan_sha256"] != sha(V97_PLAN) or
            prior_plan["source_sha256"]["evaluator"] != sha(baseline.EVALUATOR)):
        raise ValueError("Expected complete, source-matched V97 prior retention")
    if (not varied["complete"] or varied["case_count"] != 24 or
            varied["pass_count"] != 24 or varied["margin_1p2_s_count"] != 24 or
            varied["regression_count"] or varied["plan_sha256"] != sha(V110_PLAN)):
        raise ValueError("V111 waits for complete, passing V110 varied retention")
    sources = {**prior_plan["source_sha256"],
               "evaluator": sha(shifted.EVALUATOR),
               "runner": sha(Path(shifted.__file__))}
    prior_cases = {case["id"]: case for case in prior_plan["cases"]}
    selected = select_rows(prior["cases"])
    cases = []
    refs = []
    for row in selected:
        report_path = ROOT / row["result"]
        report = json.loads(report_path.read_text())
        if (not row["pass"] or
                report["evaluator_sha256"] != prior_plan["source_sha256"]["evaluator"] or
                report["body_sha256"] != sources["body"] or
                report["selected_pointer_sha256"] != sources["pointer"] or
                report["selection_mode"] != row["selection_mode"]):
            raise ValueError(f"Prior baseline identity mismatch: {row['id']}")
        refs.append({"id": row["id"], "path": row["result"],
                     "sha256": sha(report_path)})
        base = {key: value for key, value in prior_cases[row["id"]].items()
                if key not in ("id", "baseline_contact_s")}
        cases.append({**base, "id": f"weak-{row['seed']}-neg2",
                      "baseline_id": row["id"],
                      "baseline_contact_s": row["contact_s"],
                      "baseline_selection_mode": row["selection_mode"],
                      "diagnostic_goal_shift_body_y_m": SHIFT_M})
    plan = {
        "schema": "robot-learning-lab.cove50-v111-weak-prior-shift-plan.v1",
        "role": "opened_weak_prior_shift_pilot_not_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "baseline_plan_sha256": sha(V97_PLAN),
        "baseline_matrix_sha256": sha(V97_MATRIX),
        "baseline_reports": refs,
        "v110_plan_sha256": sha(V110_PLAN),
        "v110_matrix_sha256": sha(V110_MATRIX),
        "runner_sha256": sha(Path(__file__)),
        "case_count": COUNT, "cases": cases,
        "selection": ("The seven lowest-contact full-gate V97 prior starts, "
                      "then the lowest-contact representative of every observed "
                      "route mode and the older rear-correction branch, then "
                      "remaining lowest-contact starts until twelve are unique. "
                      "Selection uses only the unchanged V97 baseline, not V111 outcomes."),
        "method": ("Prescribe the V108-style -2 mm post-selection body-Y target "
                   "shift on each exact V97 old start. Preserve observed-RGB "
                   "route and measure native contact, RGB, support, rated "
                   "effort, action and collision. Run in two-visit stages and "
                   "stop on the first full-gate failure or contact below 1.2 s."),
        "decision_gate": ("All twelve require the full native gate and at least "
                          "1.2 s fingertip contact. If this pilot fails, reject "
                          "the blanket shift and develop an observable selective "
                          "correction. If it passes, run full prior and route "
                          "retention before a wholly new disjoint score. All "
                          "cases are opened, not a new score or body qualification."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V111 plan or sources changed")
    PLAN.write_bytes(encoded)
    return plan


def write_matrix(plan: dict, rows: dict[str, dict], *, stopped: bool) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    complete = len(ordered) == COUNT
    result = {
        "schema": "robot-learning-lab.cove50-v111-weak-prior-shift-pilot.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "complete": complete,
        "stopped_on_regression": stopped,
        "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
        "margin_1p2_s_count": sum(row["margin_1p2_s_met"] for row in ordered),
        "gate_met": complete and not stopped and all(
            row["pass"] and row["margin_1p2_s_met"] for row in ordered),
        "cases": ordered,
    }
    temporary = DEST / "matrix.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, DEST / "matrix.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    print(f"frozen {PLAN} sha256={sha(PLAN)}", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    old = DEST / "matrix.json"
    existing = json.loads(old.read_text()) if old.exists() else None
    if existing and existing["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V111 matrix belongs to another plan")
    if existing and existing["stopped_on_regression"]:
        print("V111 pilot already stopped on regression", flush=True)
        return
    rows = {row["id"]: row for row in existing["cases"]} if existing else {}

    def run_case(case: dict) -> dict:
        row = shifted.run_case(case, plan, DEST)
        if row["selection_mode"] != case["baseline_selection_mode"]:
            raise ValueError("Target shift changed the automatic entry route")
        row.update({"baseline_id": case["baseline_id"],
                    "baseline_contact_s": case["baseline_contact_s"],
                    "baseline_selection_mode": case["baseline_selection_mode"],
                    "diagnostic_goal_shift_body_y_m": SHIFT_M,
                    "contact_delta_s": round(row["contact_s"] -
                                             case["baseline_contact_s"], 3)})
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        for offset in range(0, COUNT, 2):
            batch = [case for case in plan["cases"][offset:offset+2]
                     if case["id"] not in rows]
            futures = {pool.submit(run_case, case): case for case in batch}
            for future in as_completed(futures):
                row = future.result()
                rows[row["id"]] = row
                write_matrix(plan, rows, stopped=False)
                print(json.dumps({key: row[key] for key in
                                  ("id", "baseline_contact_s", "contact_s",
                                   "pass", "margin_1p2_s_met", "faults")}), flush=True)
            if any(not rows[case["id"]]["pass"] or
                   not rows[case["id"]]["margin_1p2_s_met"]
                   for case in plan["cases"][offset:offset+2]):
                result = write_matrix(plan, rows, stopped=True)
                print(f"V111 stopped at {result['case_count']}/{COUNT}", flush=True)
                return
    result = write_matrix(plan, rows, stopped=False)
    print(json.dumps({key: result[key] for key in
                      ("case_count", "pass_count", "margin_1p2_s_count",
                       "gate_met")}), flush=True)


if __name__ == "__main__":
    main()
