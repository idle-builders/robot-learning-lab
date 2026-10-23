"""Complete older-start retention for the frozen V135 observed-RGB rule.

V119 native baseline and identical -2 mm visits are reusable. Only newly
selected shifts receive fresh native contact visits. This is older retention,
not an independent contact score or a new-body qualification.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    build_v118_observable_pairs as source_check,
    run_v95_native_case as baseline,
    run_v108_native_case as shifted,
    v118_observable_selector as observations,
    v135_planar_selector as selector,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
MODEL = BASE / "v135/planar-selector.json"
OLD_PLAN = BASE / "v119/selector-heldout-plan.json"
OLD_MATRIX = BASE / "v119/native-selector-heldout/matrix.json"
OUT = BASE / "v136"
PLAN = OUT / "older-start-retention-plan.json"
DEST = OUT / "native-extra-shifts"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict, *, frozen: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2) + "\n"
    if frozen and path.exists() and path.read_text() != encoded:
        raise ValueError(f"Previously frozen plan changed: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def freeze() -> dict:
    model = selector.load(MODEL)
    old_plan = json.loads(OLD_PLAN.read_text())
    old = json.loads(OLD_MATRIX.read_text())
    source = old_plan["source_sha256"]
    if (model["role"] != "frozen_development_candidate_not_qualified" or
            model["source_sha256"]["inference"] != sha(Path(selector.__file__)) or
            not old["complete"] or old["case_count"] != 24 or
            old["plan_sha256"] != sha(OLD_PLAN) or
            old["baseline_pass_count"] != 24 or
            old["baseline_margin_count"] != 24 or
            source["body"] != sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml") or
            source["pointer"] != sha(ROOT / "robot/ape-current-design.json") or
            source["evaluator"] != sha(baseline.EVALUATOR) or
            source["shift_evaluator"] != sha(shifted.EVALUATOR)):
        raise ValueError("V135 model or V119 older native sources changed")
    planned = {row["id"]: row for row in old_plan["cases"]}
    cases = []
    for row in old["cases"]:
        case = planned[row["id"]]
        base_path = ROOT / row["baseline_result"]
        old_candidate_path = ROOT / row["candidate_result"]
        report = json.loads(base_path.read_text())
        features = observations.observed_features(report)
        if (report["seed"] != case["seed"] or
                report["selection_mode"] != case["expected_selection_mode"] or
                report["body_sha256"] != source["body"] or
                report["selected_pointer_sha256"] != source["pointer"]):
            raise ValueError(f"Older baseline report differs: {case['id']}")
        decision = selector.predict(model, features)
        use_shift = decision["shift_body_y_m"] < 0
        prior_shift = row["frozen_decision"]["shift_body_y_m"] < 0
        cases.append({**{key: case[key] for key in (
                          "id", "seed", "route_variant", "rail_v", "delay_ms",
                          "toy_offset_x_m", "toy_offset_y_m", "pitch_offset_rad",
                          "roll_clearance_rad", "gaze_head_yaw_rad",
                          "expected_selection_mode")},
                      "observed_features": features,
                      "frozen_decision": decision,
                      "prior_v119_shifted": prior_shift,
                      "new_native_shift_required": use_shift and not prior_shift,
                      "old_baseline_result": row["baseline_result"],
                      "old_baseline_trace": row["baseline_trace"],
                      "old_baseline_result_sha256": sha(base_path),
                      "old_candidate_result": row["candidate_result"],
                      "old_candidate_trace": row["candidate_trace"],
                      "old_candidate_result_sha256": sha(old_candidate_path)})
    plan = {
        "schema": "robot-learning-lab.cove50-v136-older-retention-plan.v1",
        "role": "opened_prior_holdout_reused_as_older_start_retention",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {"model": sha(MODEL),
                          "inference": sha(Path(selector.__file__)),
                          "runner": sha(Path(__file__)),
                          "v119_plan": sha(OLD_PLAN),
                          "v119_matrix": sha(OLD_MATRIX),
                          **source},
        "case_count": 24,
        "selected_shift_count": sum(c["frozen_decision"]["shift_body_y_m"] < 0
                                    for c in cases),
        "new_native_shift_count": sum(c["new_native_shift_required"] for c in cases),
        "method": ("Freeze all 24 new-rule choices from saved observed-RGB "
                   "baseline reports after model fitting, before extra "
                   "contact visits. Reuse source-identical saved V119 native "
                   "baseline or -2 mm contact when the new choice matches. "
                   "Run only new -2 mm choices under unchanged native gates. "
                   "V119 was previously opened for another model, so this "
                   "is older-start retention, never a fresh held-out score."),
        "cases": cases,
    }
    write_json(PLAN, plan, frozen=True)
    return plan


def run_extra(case: dict, source: dict) -> dict:
    native = shifted.run_case(
        {**case, "id": case["id"] + "-v135-neg2",
         "diagnostic_goal_shift_body_y_m": -.002},
        {"source_sha256": {**source, "evaluator": source["shift_evaluator"]}},
        DEST)
    base_path = ROOT / case["old_baseline_result"]
    base_report = json.loads(base_path.read_text())
    source_check.make_pair(
        "V136_newly_selected_older_shift", case, case,
        case["old_baseline_result"], native["result"],
        base_report["phases"]["reach"]["continuous_finger_contact_s"],
        native["contact_s"], True, native["pass"])
    return native


def compose(plan: dict, new_rows: dict[str, dict]) -> dict:
    old = {row["id"]: row for row in json.loads(OLD_MATRIX.read_text())["cases"]}
    rows = []
    for case in plan["cases"]:
        previous = old[case["id"]]
        if case["new_native_shift_required"] and case["id"] not in new_rows:
            continue
        if not case["frozen_decision"]["shift_body_y_m"]:
            candidate = {"pass": previous["baseline_pass"],
                         "margin_1p2_s_met": previous["baseline_margin_1p2_s_met"],
                         "contact_s": previous["baseline_contact_s"],
                         "faults": previous["baseline_faults"],
                         "result": previous["baseline_result"],
                         "trace": previous["baseline_trace"]}
            source_role = "reused_v119_unchanged"
        elif case["prior_v119_shifted"]:
            candidate = {"pass": previous["candidate_pass"],
                         "margin_1p2_s_met": previous["candidate_margin_1p2_s_met"],
                         "contact_s": previous["candidate_contact_s"],
                         "faults": previous["candidate_faults"],
                         "result": previous["candidate_result"],
                         "trace": previous["candidate_trace"]}
            source_role = "reused_v119_identical_shift"
        else:
            candidate = new_rows[case["id"]]
            source_role = "fresh_v136_shift"
        item = {"id": case["id"], "seed": case["seed"],
                     "route_variant": case["route_variant"],
                     "selection_mode": case["expected_selection_mode"],
                     "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
                     "frozen_decision": case["frozen_decision"],
                     "candidate_source_role": source_role,
                     "baseline_contact_s": previous["baseline_contact_s"],
                     "candidate_contact_s": candidate["contact_s"],
                     "contact_delta_s": round(candidate["contact_s"] -
                                              previous["baseline_contact_s"], 3),
                     "baseline_pass": previous["baseline_pass"],
                     "candidate_pass": candidate["pass"],
                     "baseline_margin_1p2_s_met": previous["baseline_margin_1p2_s_met"],
                     "candidate_margin_1p2_s_met": candidate["margin_1p2_s_met"],
                     "baseline_faults": previous["baseline_faults"],
                     "candidate_faults": candidate["faults"],
                     "baseline_result": previous["baseline_result"],
                     "baseline_trace": previous["baseline_trace"],
                "candidate_result": candidate["result"],
                "candidate_trace": candidate["trace"]}
        if source_role == "fresh_v136_shift":
            item["fresh_native_result"] = candidate
        rows.append(item)
    return {
        "schema": "robot-learning-lab.cove50-v136-older-retention-matrix.v1",
        "role": "opened_prior_holdout_reused_as_older_start_retention",
        "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "complete": len(rows) == 24,
        "case_count": len(rows), "planned_count": 24,
        "new_native_shift_count": sum(r["candidate_source_role"] == "fresh_v136_shift"
                                      for r in rows),
        "candidate_pass_count": sum(r["candidate_pass"] for r in rows),
        "candidate_margin_count": sum(r["candidate_margin_1p2_s_met"] for r in rows),
        "regression_count": sum(r["baseline_pass"] and not r["candidate_pass"]
                                for r in rows),
        "margin_loss_count": sum(r["baseline_margin_1p2_s_met"] and
                                 not r["candidate_margin_1p2_s_met"] for r in rows),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        raise ValueError("Use one or two native workers")
    plan = freeze()
    print(f"V136 plan sha256={sha(PLAN)} extra native shifts "
          f"{plan['new_native_shift_count']}/24", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    prior_path = DEST / "matrix.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
    if prior and prior["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V136 matrix belongs to another plan")
    new_rows = ({row["id"]: row["fresh_native_result"] for row in prior["cases"]
                 if row["candidate_source_role"] == "fresh_v136_shift"}
                if prior else {})
    write_json(prior_path, compose(plan, new_rows))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_extra, case, plan["source_sha256"]): case
                   for case in plan["cases"]
                   if case["new_native_shift_required"] and case["id"] not in new_rows}
        for future in as_completed(futures):
            case = futures[future]
            new_rows[case["id"]] = future.result()
            updated = compose(plan, new_rows)
            write_json(prior_path, updated)
            print(f"V136 {case['id']}: "
                  f"{new_rows[case['id']]['contact_s']:.2f}s "
                  f"{'pass' if new_rows[case['id']]['pass'] else 'fail'}; "
                  f"{updated['new_native_shift_count']}/"
                  f"{plan['new_native_shift_count']} extras", flush=True)


if __name__ == "__main__":
    main()
