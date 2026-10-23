"""Freeze and collect new-seed RGB entry exposure near opened score misses.

This is a training-pool preflight, never a native contact score. It may run
after the ongoing V147 full-contact collection, without opening V147's reserve.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v98_entry_exposure as entry_runner,
    v118_observable_selector as observations,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v149"
PLAN = OUT / "new-seed-route-exposure-plan.json"
DEST = OUT / "native-entry-only"
SCORE = BASE / "v139/reserved-contact-score-plan.json"
TRAIN = BASE / "v147/new-seed-direction-training-plan.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
MATRIX = DEST / "matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict, *, frozen: bool = False) -> None:
    encoded = json.dumps(value, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists() and path.read_text() != encoded:
        raise ValueError(f"Frozen V149 plan changed: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def cases() -> list[dict]:
    rng = random.Random(149)
    result = []
    for index in range(96):
        roll_high = index % 2 == 0
        anchor = index in (0, 1)
        if roll_high:
            x = .0057 if anchor else rng.uniform(.00545, .00595)
            y = .00575 if anchor else rng.uniform(.00550, .00600)
            pitch = .105
            roll = -.24 if index % 8 != 6 else -.245
        else:
            x = .0064 if anchor else rng.uniform(.00615, .00665)
            y = .00652 if anchor else rng.uniform(.00627, .00677)
            pitch = .095 if index % 8 != 7 else .1
            roll = -.25
        result.append({
            "id": f"training-entry-{24501 + index}",
            "seed": 24501 + index,
            "cohort": "opened_roll_high_neighborhood" if roll_high else
                      "opened_pitch_low_neighborhood",
            "route_variant": "roll-high" if roll_high else "pitch-low",
            "rail_v": 12 if anchor and roll_high else
                      10 if anchor else (10, 11, 12)[(index // 2) % 3],
            "delay_ms": 10 if anchor else (0, 5, 10)[(index // 6) % 3],
            "toy_offset_x_m": round(x, 6),
            "toy_offset_y_m": round(y, 6),
            "pitch_offset_rad": pitch,
            "roll_clearance_rad": roll,
            "gaze_head_yaw_rad": (-1.5 if roll_high else -1.3)
                                  if anchor else (-1.5, -1.3)[(index // 18) % 2],
        })
    return result


def freeze() -> dict:
    pointer = ROOT / "robot/ape-current-design.json"
    if json.loads(pointer.read_text())["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("V149 is limited to the rejected historical sandbox")
    score = json.loads(SCORE.read_text())
    training = json.loads(TRAIN.read_text())
    reserve = json.loads(RESERVE.read_text())
    entry_source = entry_runner.freeze(persist=False)["source_sha256"]
    if (training["source_sha256"]["untouched_score_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits" or
            any(entry_source[key] != training["source_sha256"][key]
                for key in ("body", "pointer", "squat", "actor",
                            "alternate_actor", "specialist_actor",
                            "reference", "geometry", "calibration")) or
            entry_source["original_v95_evaluator"] !=
            training["source_sha256"]["evaluator"]):
        raise ValueError("Entry evaluator, native training or sealed score source changed")
    fresh = cases()
    prior_seeds = {case["seed"] for plan in (score, training, reserve)
                   for case in plan["cases"]}
    if len({case["seed"] for case in fresh}) != 96 or any(
            case["seed"] in prior_seeds for case in fresh):
        raise ValueError("V149 entry pool must be disjoint from score and training starts")
    plan = {
        "schema": "robot-learning-lab.cove50-v149-route-exposure-plan.v1",
        "role": "new_seed_training_entry_exposure_only_no_contact_or_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**entry_source,
                          "entry_runner": entry_source["runner"],
                          "runner": sha(Path(__file__)),
                          "v139_opened_score_plan": sha(SCORE),
                          "v147_training_plan": sha(TRAIN),
                          "v147_unopened_score_reserve": sha(RESERVE)},
        "case_count": 96,
        "prior_v147_visibility": ("Frozen after the first six all-passing V147 triplets "
                                  "were inspected; the remaining 34 outcomes were "
                                  "not used to choose this entry pool."),
        "method": ("New simulator seeds 24501–24596, 48 per opened failure's "
                   "physical neighborhood, with independent toy X/Y, voltage, "
                   "delay and gaze variation. Heavily sample the exact roll "
                   "and pitch settings underrepresented in V147. Stop at the "
                   "first used controlled RGB frame. Record actual automatic "
                   "route, camera, toy pixels and entry errors. No contact or "
                   "reserve observation is allowed. Any later paired contact "
                   "training must predeclare its selection from these entry "
                   "features and score against a still-disjoint reserve."),
        "cases": fresh,
    }
    write_json(PLAN, plan, frozen=True)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    row = entry_runner.run_case(case, plan)
    result = ROOT / row["result"]
    trace = result.with_suffix(".npz")
    report = json.loads(result.read_text())
    observed = observations.observed_features(report) if row["entry_available"] else None
    if row["entry_available"] and observed["selection_mode"] != row["selection_mode"]:
        raise ValueError(f"Observed route differs from native report: {case['id']}")
    return {**row, "cohort": case["cohort"],
            "route_variant": case["route_variant"],
            "observed_features": observed,
            "result_sha256": sha(result), "trace_sha256": sha(trace)}


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    routes = Counter(row["selection_mode"] for row in ordered)
    matrix = {
        "schema": "robot-learning-lab.cove50-v149-route-exposure.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "complete": len(ordered) == 96,
        "case_count": len(ordered), "planned_count": 96,
        "entry_available_count": sum(row["entry_available"] for row in ordered),
        "observed_route_counts": dict(sorted(routes.items())),
        "contains_contact_outcomes": False,
        "cases": ordered,
    }
    write_json(MATRIX, matrix)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four entry-only workers")
    plan = freeze()
    print(f"V149 entry plan sha256={sha(PLAN)}", flush=True)
    if args.freeze_only:
        return
    prior = json.loads(MATRIX.read_text()) if MATRIX.exists() else None
    if prior and prior["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V149 entry matrix belongs to another plan")
    rows = {row["id"]: row for row in prior["cases"]} if prior else {}
    entry_runner.DEST = DEST
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case
                   for case in plan["cases"] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V149 {row['id']}: {row['selection_mode']} "
                  f"{matrix['case_count']}/96 entries", flush=True)


if __name__ == "__main__":
    main()
