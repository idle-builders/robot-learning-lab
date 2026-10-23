"""Freeze a disjoint reserve, then collect new-seed signed-direction training pairs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as baseline,
    run_v108_native_case as shifted,
    run_v139_reserved_contact as score_runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OLD_SCORE = BASE / "v139/native-reserved-contact/matrix.json"
OLD_PLAN = BASE / "v139/reserved-contact-score-plan.json"
AUDIT = BASE / "v146/direction-coverage-audit.json"
OUT = BASE / "v147"
TRAIN_PLAN = OUT / "new-seed-direction-training-plan.json"
SCORE_RESERVE = OUT / "untouched-score-reserve-plan.json"
DEST = OUT / "native-new-seed-direction-pairs"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict, *, frozen: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2) + "\n"
    if frozen and path.exists() and path.read_text() != encoded:
        raise ValueError(f"Frozen V147 plan changed: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(encoded)
    os.replace(temp, path)


def cases(first_seed: int, count: int, *, role: str, draw_seed: int) -> list[dict]:
    rng = random.Random(draw_seed)
    result = []
    for index in range(count):
        route = "roll-high" if index % 2 == 0 else "pitch-low"
        if route == "roll-high":
            x = rng.uniform(.00515, .00625)
            y = rng.uniform(.00515, .00635)
            pitch, roll = .105, rng.choice((-.24, -.245, -.25))
        else:
            x = rng.uniform(.00595, .00695)
            y = rng.uniform(.00600, .00695)
            pitch, roll = rng.choice((.095, .1, .105)), -.25
        result.append({
            "id": f"{role}-{first_seed + index}",
            "seed": first_seed + index,
            "cohort": role,
            "route_variant": route,
            "rail_v": rng.choice((10, 11, 12)),
            "delay_ms": rng.choice((0, 5, 10)),
            "toy_offset_x_m": round(x, 6),
            "toy_offset_y_m": round(y, 6),
            "pitch_offset_rad": pitch,
            "roll_clearance_rad": roll,
            "gaze_head_yaw_rad": rng.choice((-1.5, -1.3)),
        })
    return result


def freeze() -> dict:
    pointer = ROOT / "robot/ape-current-design.json"
    old_plan = json.loads(OLD_PLAN.read_text())
    old_score = json.loads(OLD_SCORE.read_text())
    audit = json.loads(AUDIT.read_text())
    source = old_plan["source_sha256"]
    guarded = audit["negative_rule_pitch_qualified_leave_one_seed_out"]
    if (json.loads(pointer.read_text())["status"] !=
            "rejected_no_selected_mechanical_design" or
            source["pointer"] != sha(pointer) or
            source["body"] != sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml") or
            source["evaluator"] != sha(baseline.EVALUATOR) or
            source["shift_evaluator"] != sha(shifted.EVALUATOR) or
            not old_score["complete"] or old_score["score_gate_met"] or
            old_score["plan_sha256"] != sha(OLD_PLAN) or
            audit["source_sha256"]["v145_matrix"] !=
            sha(BASE / "v145/native-other-opened-seed-directions/matrix.json") or
            guarded["selected_rescue_seed_count"] != 6 or
            guarded["selected_full_gate_regressions"] or
            guarded["selected_margin_losses"] or
            guarded["selected_unresolved_failures"]):
        raise ValueError("Prior score, grouped audit, body or evaluator changed")
    training = cases(24301, 40, role="new_seed_failure_neighborhood_training",
                     draw_seed=147)
    reserved = cases(24401, 24, role="untouched_new_seed_score_reserve",
                     draw_seed=1147)
    all_cases = training + reserved
    if (len({case["seed"] for case in all_cases}) != 64 or
            {case["seed"] for case in all_cases} &
            {row["seed"] for row in old_score["cases"]}):
        raise ValueError("Seed overlap across opened and reserved plans")
    for case in all_cases:
        if not (.095 <= case["pitch_offset_rad"] <= .115 and
                -.26 <= case["roll_clearance_rad"] <= -.24 and
                case["gaze_head_yaw_rad"] in (-1.3, -1.5) and
                10 <= case["rail_v"] <= 12 and
                0 <= case["delay_ms"] <= 10):
            raise ValueError(f"Unsupported native condition: {case['id']}")
    common = {"body_role": "rejected_historical_direct_frame_sandbox",
              "source_sha256": {**source, "old_score_plan": sha(OLD_PLAN),
                                "old_score_matrix": sha(OLD_SCORE),
                                "grouped_coverage_audit": sha(AUDIT),
                                "runner": sha(Path(__file__))}}
    reserve = {
        "schema": "robot-learning-lab.cove50-v147-untouched-score-reserve.v1",
        "role": "frozen_disjoint_new_seed_conditions_no_entry_or_contact",
        **common, "status": "sealed_no_contact_visits", "case_count": 24,
        "method": ("Freeze 24 disjoint seeds and physical conditions before any "
                   "V147 training contact. No entry, outcome or policy choice "
                   "may be read while fitting; require a separately frozen "
                   "observed-input candidate and older retention before score."),
        "cases": reserved,
    }
    write_json(SCORE_RESERVE, reserve, frozen=True)
    plan = {
        "schema": "robot-learning-lab.cove50-v147-new-seed-direction-training-plan.v1",
        "role": "new_seed_paired_training_not_independent_score",
        **common, "source_sha256": {**common["source_sha256"],
                                    "untouched_score_reserve": sha(SCORE_RESERVE)},
        "case_count": 40, "visit_count": 120,
        "target_shifts_body_y_m": [0, -.002, .002],
        "method": ("Sample 40 new seed identities, balanced roll-high and "
                   "pitch-low, around the two opened failure neighborhoods. "
                   "Vary toy X/Y, motor voltage, delay, gaze and supported "
                   "pitch/roll. Pair unchanged, −2 and +2 mm full native "
                   "reaches per start with controlled RGB-entry checks, all "
                   "safety gates and 1.2 s margins. Seed and toy truth are "
                   "audit-only. A separate 24-start score reserve was sealed "
                   "before any of these contact visits."),
        "cases": training,
    }
    write_json(TRAIN_PLAN, plan, frozen=True)
    return plan


def run_pair(case: dict, plan: dict) -> dict:
    source = plan["source_sha256"]
    base = baseline.run_case({**case, "id": case["id"] + "-zero"},
                             {"source_sha256": source}, DEST)
    base_report = json.loads((ROOT / base["result"]).read_text())
    observed = score_runner.observations.observed_features(base_report)
    choices = {}
    for label, shift in (("neg2", -.002), ("pos2", .002)):
        result = shifted.run_case(
            {**case, "id": case["id"] + "-" + label,
             "diagnostic_goal_shift_body_y_m": shift},
            {"source_sha256": {**source, "evaluator": source["shift_evaluator"]}},
            DEST)
        report = json.loads((ROOT / result["result"]).read_text())
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
            "contact_s": result["contact_s"],
            "contact_delta_s": round(result["contact_s"] - base["contact_s"], 3),
            "native_pass": result["pass"],
            "pass": result["pass"] and not faults,
            "margin_1p2_s_met": result["margin_1p2_s_met"],
            "entry_parity_faults": faults,
            "faults": result["faults"] + faults,
            "result": result["result"], "trace": result["trace"],
        }
    return {**case, "selection_mode": base["selection_mode"],
            "entry_observed_features": observed,
            "baseline_contact_s": base["contact_s"],
            "baseline_pass": base["pass"],
            "baseline_margin_1p2_s_met": base["margin_1p2_s_met"],
            "baseline_faults": base["faults"],
            "baseline_result": base["result"],
            "baseline_trace": base["trace"],
            "choices": choices}


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    matrix = {
        "schema": "robot-learning-lab.cove50-v147-new-seed-direction-pairs.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(TRAIN_PLAN),
        "complete": len(ordered) == plan["case_count"],
        "case_count": len(ordered), "planned_count": plan["case_count"],
        "baseline_pass_count": sum(row["baseline_pass"] for row in ordered),
        "neg2_pass_count": sum(row["choices"]["neg2"]["pass"] for row in ordered),
        "pos2_pass_count": sum(row["choices"]["pos2"]["pass"] for row in ordered),
        "neg2_rescue_count": sum(not row["baseline_pass"] and
                                 row["choices"]["neg2"]["pass"] for row in ordered),
        "pos2_rescue_count": sum(not row["baseline_pass"] and
                                 row["choices"]["pos2"]["pass"] for row in ordered),
        "neg2_regression_count": sum(row["baseline_pass"] and
                                     not row["choices"]["neg2"]["pass"] for row in ordered),
        "pos2_regression_count": sum(row["baseline_pass"] and
                                     not row["choices"]["pos2"]["pass"] for row in ordered),
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
        raise ValueError("Use one or two native evaluators")
    plan = freeze()
    print(f"V147 training plan sha256={sha(TRAIN_PLAN)}, reserve "
          f"sha256={sha(SCORE_RESERVE)}; 40 training seeds, 120 visits", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "matrix.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if old and old["plan_sha256"] != sha(TRAIN_PLAN):
        raise ValueError("Existing V147 matrix belongs to another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_pair, case, plan): case
                   for case in plan["cases"] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V147 {row['id']}: unchanged {row['baseline_contact_s']:.2f}, "
                  f"−2 {row['choices']['neg2']['contact_s']:.2f}, "
                  f"+2 {row['choices']['pos2']['contact_s']:.2f}s; "
                  f"{matrix['case_count']}/40", flush=True)


if __name__ == "__main__":
    main()
