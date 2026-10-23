"""Freeze and run the first native held-out contact score for V118."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as baseline,
    run_v108_native_case as shifted,
    v118_observable_selector as selector,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
RESERVED = BASE / "v116/reserved-contact-validation-plan.json"
MODEL = BASE / "v118/observable-selector.json"
V113_PLAN = BASE / "v113/varied-paired-contact-data-plan.json"
OUT = BASE / "v119"
PLAN = OUT / "selector-heldout-plan.json"
DEST = OUT / "native-selector-heldout"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def freeze() -> dict:
    reserved = json.loads(RESERVED.read_text())
    if reserved["case_count"] != 24 or len(reserved["cases"]) != 24:
        raise ValueError("Incomplete reserved contact holdout")
    model = selector.load(MODEL)
    if model["role"] != "frozen_development_candidate_not_qualified":
        raise ValueError("Contact selector is not the frozen development candidate")
    if model["inference_source_sha256"] != sha(Path(selector.__file__)):
        raise ValueError("Selector inference source changed after model freeze")
    source = json.loads(V113_PLAN.read_text())["source_sha256"]
    if (source["evaluator"] != sha(baseline.EVALUATOR) or
            source["shift_evaluator"] != sha(shifted.EVALUATOR) or
            source["body"] != reserved["source_sha256"]["body"] or
            source["pointer"] != reserved["source_sha256"]["pointer"]):
        raise ValueError("Native evaluator or rejected-body source changed")
    refs = reserved["source_entry_reports"]
    if len(refs) != len(reserved["cases"]):
        raise ValueError("Reserved RGB entry reference count changed")
    cases = []
    for case, ref in zip(reserved["cases"], refs):
        report_path = ROOT / ref["path"]
        if sha(report_path) != ref["sha256"]:
            raise ValueError("Reserved observed-RGB entry source changed")
        report = json.loads(report_path.read_text())
        if (report["seed"] != case["seed"] or
                report["selection_mode"] != case["expected_selection_mode"] or
                report["body_sha256"] != source["body"] or
                report["selected_pointer_sha256"] != source["pointer"]):
            raise ValueError("Reserved entry source differs from native start")
        features = selector.observed_features(report)
        decision = selector.predict(model, features)
        cases.append({**case, "source_entry_report": ref,
                      "entry_observed_features": features,
                      "frozen_decision": decision})
    plan = {
        "schema": "robot-learning-lab.cove50-v119-selector-heldout-plan.v1",
        "role": "first_sealed_heldout_native_contact_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "reserved_plan_sha256": sha(RESERVED),
        "model_sha256": sha(MODEL),
        "inference_source_sha256": sha(Path(selector.__file__)),
        "runner_source_sha256": sha(Path(__file__)),
        "v113_native_source_plan_sha256": sha(V113_PLAN),
        "source_sha256": source,
        "selection_rule": model["selection_rule"],
        "selection_timing": ("All 24 decisions frozen from the previously "
                             "retained V105 observed-RGB entry reports, before "
                             "any V116 reserved full-native contact visit. "
                             "The model excludes these seeds and all V113 "
                             "pairs from fitting."),
        "method": ("Run a fresh unchanged native baseline for every reserved "
                   "start. For the starts frozen to −2 mm, run an additional "
                   "full-native visit at that offset. Compare native "
                   "contact, RGB, support, effort, action and collision gates. "
                   "An unchanged decision uses its unchanged native visit as "
                   "the candidate result."),
        "qualification": ("This first holdout tests a fitted entry-to-target "
                          "choice on a rejected mechanical sandbox. It is not "
                          "an integrated online selector or a qualification "
                          "for the future body."),
        "case_count": len(cases),
        "selected_shift_count": sum(case["frozen_decision"]["shift_body_y_m"] < 0
                                    for case in cases),
        "cases": cases,
    }
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen held-out score plan or source changed")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, source: dict) -> dict:
    base_case = {**case, "id": case["id"] + "-zero"}
    native_base = baseline.run_case(base_case, {"source_sha256": source}, DEST)
    base_report = json.loads((ROOT / native_base["result"]).read_text())
    replay_features = selector.observed_features(base_report)
    replay_decision = selector.predict(selector.load(MODEL), replay_features)
    planned = case["frozen_decision"]["shift_body_y_m"]
    if replay_decision["shift_body_y_m"] != planned:
        raise ValueError(f"Native entry changes frozen selector decision: {case['id']}")
    if planned < 0:
        shift_case = {**case, "id": case["id"] + "-selected-neg2",
                      "diagnostic_goal_shift_body_y_m": planned}
        shifted_source = {**source, "evaluator": source["shift_evaluator"]}
        candidate = shifted.run_case(shift_case,
                                     {"source_sha256": shifted_source}, DEST)
    else:
        candidate = native_base
    if (candidate["selection_mode"] != case["expected_selection_mode"] or
            native_base["selection_mode"] != case["expected_selection_mode"]):
        raise ValueError(f"Native route drift on reserved start {case['id']}")
    return {
        "id": case["id"], "seed": case["seed"],
        "route_variant": case["route_variant"],
        "selection_mode": candidate["selection_mode"],
        "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
        "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
        "frozen_decision": case["frozen_decision"],
        "native_replay_decision": replay_decision,
        "baseline_contact_s": native_base["contact_s"],
        "candidate_contact_s": candidate["contact_s"],
        "contact_delta_s": round(candidate["contact_s"] - native_base["contact_s"], 3),
        "baseline_pass": native_base["pass"],
        "candidate_pass": candidate["pass"],
        "baseline_margin_1p2_s_met": native_base["margin_1p2_s_met"],
        "candidate_margin_1p2_s_met": candidate["margin_1p2_s_met"],
        "baseline_faults": native_base["faults"],
        "candidate_faults": candidate["faults"],
        "baseline_result": native_base["result"],
        "baseline_trace": native_base["trace"],
        "candidate_result": candidate["result"],
        "candidate_trace": candidate["trace"],
    }


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    matrix = {
        "schema": "robot-learning-lab.cove50-v119-selector-heldout-native.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "complete": len(ordered) == plan["case_count"],
        "case_count": len(ordered), "planned_count": plan["case_count"],
        "selected_shift_count": sum(row["frozen_decision"]["shift_body_y_m"] < 0
                                    for row in ordered),
        "baseline_pass_count": sum(row["baseline_pass"] for row in ordered),
        "candidate_pass_count": sum(row["candidate_pass"] for row in ordered),
        "baseline_margin_count": sum(row["baseline_margin_1p2_s_met"] for row in ordered),
        "candidate_margin_count": sum(row["candidate_margin_1p2_s_met"] for row in ordered),
        "rescue_count": sum(not row["baseline_pass"] and row["candidate_pass"]
                            for row in ordered),
        "regression_count": sum(row["baseline_pass"] and not row["candidate_pass"]
                                for row in ordered),
        "contact_delta_total_s": round(sum(row["contact_delta_s"] for row in ordered), 3),
        "cases": ordered,
    }
    write_json(DEST / "matrix.json", matrix)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    print(f"frozen {PLAN} sha256={sha(PLAN)} "
          f"selected={plan['selected_shift_count']}/{plan['case_count']}", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "matrix.json"
    existing = json.loads(path.read_text()) if path.exists() else None
    if existing and existing["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing heldout matrix belongs to another plan")
    rows = {row["id"]: row for row in existing["cases"]} if existing else {}
    for case in plan["cases"]:
        if case["id"] in rows:
            continue
        row = run_case(case, plan["source_sha256"])
        rows[row["id"]] = row
        write_matrix(plan, rows)
        print(json.dumps({key: row[key] for key in
                          ("id", "baseline_contact_s", "candidate_contact_s",
                           "baseline_pass", "candidate_pass")}), flush=True)


if __name__ == "__main__":
    main()
