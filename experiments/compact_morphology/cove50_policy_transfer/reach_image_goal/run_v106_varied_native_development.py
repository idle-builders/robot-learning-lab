"""Freeze and replay full native reach on varied, postscore development starts."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v106"
PLAN = OUT / "varied-native-development-plan.json"
DEST = OUT / "native-varied-development"
V101_PLAN = BASE / "v101/disjoint-reach-plan.json"
V101_MATRIX = BASE / "v101/native-disjoint/matrix.json"
V105_PLAN = BASE / "v105/varied-entry-exposure-plan.json"
FAILED_SCORE_SEEDS = (22021, 22028, 22031)
# Chosen from the frozen V105 condition plan before reading its entry outcomes.
VARIED_CONDITION_SEEDS = (
    19064, 19019, 19030, 19060, 19044, 19050, 19022,
    19051, 19026, 19049, 19058, 19063, 19045, 19055,
    19053, 19035, 19018, 19043, 19047, 19031, 19003,
)
FACTORS = ("route_variant", "rail_v", "delay_ms", "gaze_head_yaw_rad",
           "toy_offset_x_m", "toy_offset_y_m")
PAIRS = (("route_variant", "rail_v"), ("route_variant", "delay_ms"),
         ("route_variant", "gaze_head_yaw_rad"),
         ("route_variant", "toy_offset_x_m"),
         ("route_variant", "toy_offset_y_m"), ("rail_v", "delay_ms"),
         ("gaze_head_yaw_rad", "toy_offset_x_m"),
         ("gaze_head_yaw_rad", "toy_offset_y_m"),
         ("toy_offset_x_m", "toy_offset_y_m"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coverage(cases: list[dict]) -> dict:
    return {
        "marginals": {key: {str(value): count for value, count in
                            sorted(Counter(case[key] for case in cases).items(),
                                   key=lambda item: str(item[0]))}
                      for key in FACTORS},
        "pair_cells": {f"{a}__{b}": len({(case[a], case[b]) for case in cases})
                       for a, b in PAIRS},
    }


def frozen_plan() -> dict:
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    score_plan = json.loads(V101_PLAN.read_text())
    score = json.loads(V101_MATRIX.read_text())
    entry_plan = json.loads(V105_PLAN.read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("V106 applies only to the rejected historical sandbox")
    if (not score["complete"] or score["score_gate_met"] or
            score["plan_sha256"] != sha(V101_PLAN) or
            tuple(row["seed"] for row in score["cases"] if not row["pass"]) !=
            FAILED_SCORE_SEEDS):
        raise ValueError("Expected sealed, failed V101 score")
    if (entry_plan["case_count"] != 64 or
            sha(V105_PLAN) !=
            "a60172d1b927351f88f740bba70cf1e6439f193f49974f05cc1dcfae1577fadb"):
        raise ValueError("Expected the independently frozen V105 condition plan")
    sources = score_plan["source_sha256"]
    for key, path in (("evaluator", native.EVALUATOR),
                      ("pointer", ROOT / "robot/ape-current-design.json"),
                      ("runner", Path(native.__file__))):
        if sources[key] != sha(path):
            raise ValueError(f"V106 {key} source differs from V101")
    score_cases = {case["seed"]: case for case in score_plan["cases"]}
    entry_cases = {case["seed"]: case for case in entry_plan["cases"]}
    cases = []
    for index, source_seed in enumerate(
            FAILED_SCORE_SEEDS + VARIED_CONDITION_SEEDS, start=1):
        origin = score_cases if source_seed in FAILED_SCORE_SEEDS else entry_cases
        base = {key: value for key, value in origin[source_seed].items()
                if key not in ("id", "seed") and not key.startswith("expected_")}
        seed = 24000 + index
        cases.append({**base, "id": f"development-{seed}", "seed": seed,
                      "development_role": ("failed_score_condition_new_start"
                                           if source_seed in FAILED_SCORE_SEEDS else
                                           "balanced_varied_condition_new_start"),
                      "source_condition_seed": source_seed})
    if len(cases) != 24 or len({tuple(case[key] for key in FACTORS)
                                for case in cases}) != 24:
        raise ValueError("V106 requires 24 distinct varied condition tuples")
    spread = coverage(cases)
    minima = {"route_variant": 4, "rail_v": 6, "delay_ms": 6,
              "gaze_head_yaw_rad": 10, "toy_offset_x_m": 4,
              "toy_offset_y_m": 4}
    if any(min(spread["marginals"][key].values()) < required
           for key, required in minima.items()):
        raise ValueError("V106 marginal coverage fell below its predeclared floor")
    return {
        "schema": "robot-learning-lab.cove50-v106-varied-native-development-plan.v1",
        "role": "opened_postscore_full_native_training_diagnostic_not_disjoint_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "v101_failed_score_matrix_sha256": sha(V101_MATRIX),
        "v101_plan_sha256": sha(V101_PLAN),
        "v105_condition_plan_sha256": sha(V105_PLAN),
        "case_count": len(cases), "cases": cases,
        "coverage": spread,
        "method": ("After V101 finished and failed, repeat its three failed condition "
                   "tuples on new seeds 24001-24003. Select 21 other condition "
                   "tuples from the already frozen V105 varied plan by a fixed "
                   "coverage choice made without V105 outcomes; run them on new "
                   "seeds 24004-24024. Replay the unchanged V95 full native "
                   "controller without a physics reset, measuring RGB, fingertip "
                   "contact, two-foot support, rated effort, action and collision."),
        "decision_gate": ("All 24 reports and traces must be source-matched. "
                          "List every full-gate or 1.2 s contact-margin miss and "
                          "actual route exposure. Use only these opened "
                          "development outcomes to choose teacher comparisons "
                          "and actor/router training. V101 remains an untouched "
                          "failed score; promotion needs prior retention and "
                          "another separately frozen disjoint native score."),
    }


def freeze() -> dict:
    plan = frozen_plan()
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V106 plan or source changed")
    PLAN.write_bytes(encoded)
    return plan


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    complete = len(ordered) == plan["case_count"]
    result = {
        "schema": "robot-learning-lab.cove50-v106-varied-native-development.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "complete": complete,
        "case_count": len(ordered),
        "pass_count": sum(row["pass"] for row in ordered),
        "margin_1p2_s_count": sum(row["margin_1p2_s_met"] for row in ordered),
        "new_specialist_count": sum(
            row["automatic_alternate_gaze_specialist_applied"] for row in ordered),
        "natural_specialist_count": sum(
            row["selection_mode"] == "v16_specialist_locked" and
            not row["automatic_alternate_gaze_specialist_applied"]
            for row in ordered),
        "rear_correction_count": sum(
            row["automatic_rear_contact_bias_applied"] for row in ordered),
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
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four native evaluators")
    plan = freeze()
    print(f"frozen {PLAN} sha256={sha(PLAN)}", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    old = DEST / "matrix.json"
    existing = json.loads(old.read_text()) if old.exists() else None
    if existing and existing["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V106 matrix belongs to another frozen plan")
    rows = {row["id"]: row for row in existing["cases"]} if existing else {}

    def run_case(case: dict) -> dict:
        row = native.run_case(case, plan, DEST)
        row.update({"development_role": case["development_role"],
                    "source_condition_seed": case["source_condition_seed"],
                    "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
                    "toy_offset_x_m": case["toy_offset_x_m"],
                    "toy_offset_y_m": case["toy_offset_y_m"],
                    "gaze_head_yaw_rad": case["gaze_head_yaw_rad"]})
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case): case for case in plan["cases"]
                   if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            write_matrix(plan, rows)
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "selection_mode",
                               "automatic_alternate_gaze_specialist_applied",
                               "automatic_rear_contact_bias_applied", "faults")}),
                  flush=True)
    matrix = write_matrix(plan, rows)
    print(json.dumps({key: matrix[key] for key in
                      ("case_count", "pass_count", "margin_1p2_s_count",
                       "new_specialist_count", "natural_specialist_count",
                       "rear_correction_count")}), flush=True)


if __name__ == "__main__":
    main()
