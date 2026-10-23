"""Score frozen V135 choices on the independently reserved V128 starts."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    build_v118_observable_pairs as paired_source,
    run_v95_native_case as baseline,
    run_v108_native_case as shifted,
    v118_observable_selector as observations,
    v135_planar_selector as selector,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
RESERVE = BASE / "v128/disjoint-contact-score-plan.json"
MODEL = BASE / "v135/planar-selector.json"
RETENTION = BASE / "v136/native-extra-shifts/matrix.json"
ENTRY_MATRIX = BASE / "v137/native-entry-only/matrix.json"
DECISIONS = BASE / "v138/frozen-reserved-decisions.json"
OUT = BASE / "v139"
PLAN = OUT / "reserved-contact-score-plan.json"
DEST = OUT / "native-reserved-contact"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict, *, frozen: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2) + "\n"
    if frozen and path.exists() and path.read_text() != encoded:
        raise ValueError(f"Previously frozen score plan changed: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def freeze() -> dict:
    reserve = json.loads(RESERVE.read_text())
    model = selector.load(MODEL)
    retention = json.loads(RETENTION.read_text())
    decisions = json.loads(DECISIONS.read_text())
    source = reserve["source_sha256"]
    if (reserve["status"] != "sealed_no_contact_visits" or
            reserve["case_count"] != 24 or
            model["source_sha256"]["sealed_reserve"] != sha(RESERVE) or
            model["source_sha256"]["inference"] != sha(Path(selector.__file__)) or
            not retention["complete"] or retention["candidate_pass_count"] != 24 or
            retention["candidate_margin_count"] != 24 or
            retention["regression_count"] or retention["margin_loss_count"] or
            decisions["status"] != "frozen_before_any_full_native_contact_visit" or
            decisions["case_count"] != 24 or
            not decisions["correction_exposure_gate_met"] or
            decisions["source_sha256"]["reserve"] != sha(RESERVE) or
            decisions["source_sha256"]["candidate"] != sha(MODEL) or
            decisions["source_sha256"]["older_retention"] != sha(RETENTION) or
            decisions["source_sha256"]["entry_matrix"] != sha(ENTRY_MATRIX) or
            source["body"] != sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml") or
            source["pointer"] != sha(ROOT / "robot/ape-current-design.json") or
            source["evaluator"] != sha(baseline.EVALUATOR) or
            source["shift_evaluator"] != sha(shifted.EVALUATOR)):
        raise ValueError("Reserved score source, older gate or frozen choices changed")
    source_cases = {case["id"]: case for case in reserve["cases"]}
    if ({case["id"] for case in decisions["cases"]} != set(source_cases) or
            any(any(case[key] != source_cases[case["id"]][key]
                    for key in ("seed", "route_variant", "rail_v", "delay_ms",
                                "toy_offset_x_m", "toy_offset_y_m",
                                "pitch_offset_rad", "roll_clearance_rad",
                                "gaze_head_yaw_rad"))
                for case in decisions["cases"])):
        raise ValueError("Frozen decision conditions differ from reserved starts")
    plan = {
        "schema": "robot-learning-lab.cove50-v139-reserved-contact-plan.v1",
        "role": "first_independent_contact_score_of_frozen_v135_rule",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**source,
                          "reserve": sha(RESERVE),
                          "candidate": sha(MODEL),
                          "inference": sha(Path(selector.__file__)),
                          "older_retention": sha(RETENTION),
                          "entry_matrix": sha(ENTRY_MATRIX),
                          "frozen_decisions": sha(DECISIONS),
                          "runner": sha(Path(__file__))},
        "case_count": 24,
        "selected_shift_count": decisions["selected_shift_count"],
        "selected_near_failure_by_route": decisions["selected_near_failure_by_route"],
        "gate": ("All 24 full native RGB/contact/support/rated-effort/action/"
                 "collision passes, all 24 >=1.2 s contact, no older-start "
                 "regression, at least one actual frozen shift in roll-low "
                 "and roll-high near-failure cohorts, and report every "
                 "visibility or decision-entry parity failure."),
        "method": ("Choices were frozen from observed RGB in V138 before "
                   "any V128 contact. For each reserved start replay the "
                   "unchanged native baseline, check the observed entry and "
                   "runtime decision against its frozen preflight, then run "
                   "a source-identical -2 mm visit only when selected. "
                   "Finish all 24, including failures. This scores a "
                   "frozen entry choice on a rejected historical body; "
                   "online integration and new-body transfer are separate."),
        "cases": decisions["cases"],
    }
    write_json(PLAN, plan, frozen=True)
    return plan


def entry_parity(frozen: dict | None, observed: dict | None) -> list[str]:
    if frozen is None and observed is None:
        return []
    if frozen is None or observed is None:
        return ["controlled RGB entry availability differs from frozen preflight"]
    faults = []
    if (frozen["selection_mode"] != observed["selection_mode"] or
            frozen["route_goal_source"] != observed["route_goal_source"]):
        faults.append("automatic route or RGB goal source differs from frozen preflight")
    if any(abs(frozen[key] - observed[key]) > tolerance
           for key, tolerance in (("camera_x_mm", .05), ("camera_y_mm", .05),
                                  ("entry_error_x_mm", .1),
                                  ("entry_error_y_mm", .1))) or\
            abs(frozen["orange_pixels"] - observed["orange_pixels"]) > 1:
        faults.append("observed RGB entry differs from frozen preflight tolerance")
    return faults


def run_case(case: dict, plan: dict, model: dict) -> dict:
    source = plan["source_sha256"]
    native_base = baseline.run_case({**case, "id": case["id"] + "-zero"},
                                    {"source_sha256": source}, DEST)
    base_report = json.loads((ROOT / native_base["result"]).read_text())
    try:
        observed = observations.observed_features(base_report)
    except ValueError:
        observed = None
    frozen = case["entry_observed_features"]
    faults = entry_parity(frozen, observed)
    replay = (selector.predict(model, observed) if observed is not None else
              {"shift_body_y_m": 0.0,
               "abstention_reasons": ["no_used_controlled_RGB_entry"]})
    if replay["shift_body_y_m"] != case["frozen_decision"]["shift_body_y_m"]:
        faults.append("live observed-RGB correction differs from frozen choice")
    if case["frozen_decision"]["shift_body_y_m"] < 0:
        native_candidate = shifted.run_case(
            {**case, "id": case["id"] + "-selected-neg2",
             "diagnostic_goal_shift_body_y_m": -.002},
            {"source_sha256": {**source,
                               "evaluator": source["shift_evaluator"]}}, DEST)
        try:
            paired_source.make_pair(
                "V139_reserved_RGB_entry_parity", case, case,
                native_base["result"], native_candidate["result"],
                native_base["contact_s"], native_candidate["contact_s"],
                native_base["pass"], native_candidate["pass"])
        except ValueError as exc:
            faults.append(f"paired native RGB source mismatch: {exc}")
    else:
        native_candidate = native_base
    if (frozen is not None and
            native_base["selection_mode"] != frozen["selection_mode"]):
        faults.append("native automatic route differs from frozen entry")
    return {
        "id": case["id"], "seed": case["seed"],
        "cohort": case["cohort"], "route_variant": case["route_variant"],
        "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
        "selection_mode": native_base["selection_mode"],
        "frozen_decision": case["frozen_decision"],
        "native_replay_decision": replay,
        "frozen_entry_features": frozen,
        "native_entry_features": observed,
        "entry_parity_faults": faults,
        "baseline_contact_s": native_base["contact_s"],
        "candidate_contact_s": native_candidate["contact_s"],
        "contact_delta_s": round(native_candidate["contact_s"] -
                                 native_base["contact_s"], 3),
        "baseline_native_pass": native_base["pass"],
        "candidate_native_pass": native_candidate["pass"],
        "candidate_pass": native_candidate["pass"] and not faults,
        "candidate_margin_1p2_s_met": native_candidate["margin_1p2_s_met"],
        "baseline_faults": native_base["faults"],
        "candidate_faults": native_candidate["faults"] + faults,
        "baseline_result": native_base["result"],
        "baseline_trace": native_base["trace"],
        "candidate_result": native_candidate["result"],
        "candidate_trace": native_candidate["trace"],
    }


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"]
               if case["id"] in rows]
    shifted = Counter(row["route_variant"] for row in ordered
                      if row["cohort"] == "heldout_near_failure" and
                      row["frozen_decision"]["shift_body_y_m"] < 0)
    matrix = {
        "schema": "robot-learning-lab.cove50-v139-reserved-contact-matrix.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "complete": len(ordered) == 24,
        "case_count": len(ordered), "planned_count": 24,
        "baseline_native_pass_count": sum(row["baseline_native_pass"] for row in ordered),
        "candidate_native_pass_count": sum(row["candidate_native_pass"] for row in ordered),
        "candidate_pass_count": sum(row["candidate_pass"] for row in ordered),
        "candidate_margin_count": sum(row["candidate_margin_1p2_s_met"] for row in ordered),
        "entry_parity_failure_count": sum(bool(row["entry_parity_faults"]) for row in ordered),
        "selected_shift_count": sum(row["frozen_decision"]["shift_body_y_m"] < 0
                                    for row in ordered),
        "selected_near_failure_by_route": dict(shifted),
        "full_gate_rescue_count": sum(not row["baseline_native_pass"] and
                                      row["candidate_pass"] for row in ordered),
        "full_gate_regression_count": sum(row["baseline_native_pass"] and
                                          not row["candidate_pass"] for row in ordered),
        "score_gate_met": (len(ordered) == 24 and
                           all(row["candidate_pass"] and
                               row["candidate_margin_1p2_s_met"] for row in ordered) and
                           all(shifted[route] for route in ("roll-low", "roll-high"))),
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
    print(f"V139 score plan sha256={sha(PLAN)} selected "
          f"{plan['selected_shift_count']}/24", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "matrix.json"
    prior = json.loads(path.read_text()) if path.exists() else None
    if prior and prior["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing score matrix belongs to another plan")
    rows = {row["id"]: row for row in prior["cases"]} if prior else {}
    model = selector.load(MODEL)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan, model): case
                   for case in plan["cases"] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V139 {row['id']}: {row['candidate_contact_s']:.2f}s "
                  f"{'pass' if row['candidate_pass'] else 'fail'} "
                  f"{len(row['entry_parity_faults'])} entry faults; "
                  f"{matrix['case_count']}/24", flush=True)


if __name__ == "__main__":
    main()
