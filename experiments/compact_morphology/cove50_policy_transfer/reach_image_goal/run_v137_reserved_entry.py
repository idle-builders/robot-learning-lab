"""Observe V128 reserved starts only through their first controlled RGB frame."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v98_entry_exposure as entry_runner,
    v118_observable_selector as observations,
    v135_planar_selector as selector,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
RESERVE = BASE / "v128/disjoint-contact-score-plan.json"
MODEL = BASE / "v135/planar-selector.json"
RETENTION = BASE / "v136/native-extra-shifts/matrix.json"
OUT = BASE / "v137"
PLAN = OUT / "reserved-entry-only-plan.json"
DEST = OUT / "native-entry-only"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict, *, frozen: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2) + "\n"
    if frozen and path.exists() and path.read_text() != encoded:
        raise ValueError(f"Previously frozen entry plan changed: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def freeze() -> dict:
    reserve = json.loads(RESERVE.read_text())
    model = selector.load(MODEL)
    retention = json.loads(RETENTION.read_text())
    entry_source = entry_runner.freeze(persist=False)["source_sha256"]
    if (reserve["status"] != "sealed_no_contact_visits" or
            reserve["case_count"] != 24 or
            len({case["seed"] for case in reserve["cases"]}) != 24 or
            model["role"] != "frozen_development_candidate_not_qualified" or
            model["source_sha256"]["inference"] != sha(Path(selector.__file__)) or
            model["source_sha256"]["sealed_reserve"] != sha(RESERVE) or
            not retention["complete"] or retention["case_count"] != 24 or
            retention["candidate_pass_count"] != 24 or
            retention["candidate_margin_count"] != 24 or
            retention["regression_count"] or retention["margin_loss_count"] or
            any(entry_source[key] != reserve["source_sha256"][key]
                for key in ("body", "pointer", "squat", "actor",
                            "alternate_actor", "specialist_actor",
                            "reference", "geometry", "calibration"))):
        raise ValueError("Sealed reserve, selected body or older retention gate failed")
    prior_runner_sha = entry_source.pop("runner")
    plan = {
        "schema": "robot-learning-lab.cove50-v137-reserved-entry-plan.v1",
        "role": "independent_reserve_entry_observation_only_no_contact",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**entry_source,
                          "entry_runner": prior_runner_sha,
                          "runner": sha(Path(__file__)),
                          "reserve": sha(RESERVE),
                          "candidate": sha(MODEL),
                          "older_retention": sha(RETENTION)},
        "case_count": 24,
        "method": ("After V136 passed all 24 older native contacts, use the "
                   "unchanged V98 evaluator only to the first used RGB "
                   "reach frame of each previously sealed V128 start. "
                   "Record observed pixels, camera and automatic route, "
                   "then freeze V135 choices before any V128 full-native "
                   "contact visit. This process sees no contact outcome."),
        "cases": reserve["cases"],
    }
    write_json(PLAN, plan, frozen=True)
    return plan


def run_case(case: dict, plan: dict, model: dict) -> dict:
    row = entry_runner.run_case(case, plan)
    report_path = ROOT / row["result"]
    report = json.loads(report_path.read_text())
    observed = (observations.observed_features(report)
                if row["entry_available"] else None)
    decision = (selector.predict(model, observed)
                if observed is not None else
                {"shift_body_y_m": 0.0,
                 "abstention_reasons": ["no_used_controlled_RGB_entry"]})
    return {**row, "cohort": case["cohort"],
            "route_variant": case["route_variant"],
            "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
            "observed_features": observed,
            "frozen_model_decision_preview_no_contact": decision,
            "result_sha256": sha(report_path),
            "trace_sha256": sha(ROOT / row["result"].replace(".json", ".npz"))}


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"]
               if case["id"] in rows]
    selected = Counter(row["cohort"] for row in ordered
                       if row["frozen_model_decision_preview_no_contact"]["shift_body_y_m"] < 0)
    matrix = {
        "schema": "robot-learning-lab.cove50-v137-reserved-entry-only.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "complete": len(ordered) == 24,
        "case_count": len(ordered), "planned_count": 24,
        "entry_available_count": sum(row["entry_available"] for row in ordered),
        "selected_shift_count": sum(selected.values()),
        "selected_shift_by_cohort": dict(selected),
        "contains_contact_outcomes": False,
        "cases": ordered,
    }
    write_json(DEST / "matrix.json", matrix)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four entry-only workers")
    plan = freeze()
    print(f"V137 entry plan sha256={sha(PLAN)}", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    old_path = DEST / "matrix.json"
    prior = json.loads(old_path.read_text()) if old_path.exists() else None
    if prior and prior["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing entry matrix belongs to another plan")
    rows = {row["id"]: row for row in prior["cases"]} if prior else {}
    entry_runner.DEST = DEST
    model = selector.load(MODEL)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan, model): case
                   for case in plan["cases"] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V137 {row['id']}: "
                  f"{'RGB' if row['entry_available'] else 'no RGB'} "
                  f"{row['selection_mode']} "
                  f"{'−2mm' if row['frozen_model_decision_preview_no_contact']['shift_body_y_m'] < 0 else 'unchanged'}; "
                  f"{matrix['case_count']}/24 entries", flush=True)


if __name__ == "__main__":
    main()
