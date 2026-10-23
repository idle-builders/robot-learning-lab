"""Collect paired physical perturbations of three opened reach failures.

These are development cases on the rejected Direct Frame. The seed identities
were already opened in V101; no result is a new-seed or held-out score.
"""

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
OUT = BASE / "v130"
PLAN = OUT / "opened-failure-perturbation-plan.json"
DEST = OUT / "native-opened-failure-perturbations"
V101_PLAN = BASE / "v101/disjoint-reach-plan.json"
V101_MATRIX = BASE / "v101/native-disjoint/matrix.json"
V113_PLAN = BASE / "v113/varied-paired-contact-data-plan.json"
V128_RESERVE = BASE / "v128/disjoint-contact-score-plan.json"
ANCHORS = (22021, 22028, 22031)
SHIFT_M = -.002
PERTURBATIONS = (
    (-.0004, .0, 12, 10, -1.5),
    (+.0004, .0, 11, 5, -1.3),
    (.0, -.0004, 10, 10, -1.5),
    (.0, +.0004, 12, 0, -1.3),
    (-.0002, +.0002, 11, 10, -1.3),
    (+.0002, -.0002, 10, 5, -1.5),
    (-.0004, -.0004, 12, 5, -1.3),
    (+.0004, +.0004, 11, 0, -1.5),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_write(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError(f"Frozen plan changed: {path}")
    path.write_bytes(encoded)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def freeze() -> dict:
    pointer = ROOT / "robot/ape-current-design.json"
    if json.loads(pointer.read_text())["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Only the rejected historical control sandbox is allowed")
    original = {case["seed"]: case
                for case in json.loads(V101_PLAN.read_text())["cases"]}
    v101 = json.loads(V101_MATRIX.read_text())
    source_rows = {row["seed"]: row for row in v101["cases"]}
    if (not v101["complete"] or v101["case_count"] != 32 or
            v101["plan_sha256"] != sha(V101_PLAN) or
            any(source_rows[seed]["pass"] or source_rows[seed]["contact_s"] >= 1
                for seed in ANCHORS)):
        raise ValueError("Opened V101 failure evidence changed")
    source = json.loads(V113_PLAN.read_text())["source_sha256"]
    if (source["body"] != sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml") or
            source["pointer"] != sha(pointer) or
            source["evaluator"] != sha(baseline.EVALUATOR) or
            source["shift_evaluator"] != sha(shifted.EVALUATOR)):
        raise ValueError("Body, pointer or evaluator differs from saved V113 source")
    reserve = json.loads(V128_RESERVE.read_text())
    if (reserve["status"] != "sealed_no_contact_visits" or
            len(reserve["cases"]) != 24 or
            {case["seed"] for case in reserve["cases"]} & set(ANCHORS)):
        raise ValueError("The independent V128 reserve changed or overlaps")
    cases = []
    for anchor_seed in ANCHORS:
        anchor = original[anchor_seed]
        for index, (dx, dy, voltage, delay, gaze) in enumerate(PERTURBATIONS):
            cases.append({
                "id": f"opened-{anchor_seed}-variation-{index + 1:02d}",
                "seed": anchor_seed,
                "cohort": "opened_failure_seed_physical_perturbation",
                "anchor_opened_failure_seed": anchor_seed,
                "route_variant": anchor["route_variant"],
                "rail_v": voltage, "delay_ms": delay,
                "toy_offset_x_m": round(anchor["toy_offset_x_m"] + dx, 6),
                "toy_offset_y_m": round(anchor["toy_offset_y_m"] + dy, 6),
                "pitch_offset_rad": anchor["pitch_offset_rad"],
                "roll_clearance_rad": anchor["roll_clearance_rad"],
                "gaze_head_yaw_rad": gaze,
                "condition_delta_xy_mm": [round(dx * 1000, 3),
                                          round(dy * 1000, 3)],
            })
    if len(cases) != 24 or len({case["id"] for case in cases}) != 24:
        raise ValueError("Expected 24 distinct development conditions")
    plan = {
        "schema": "robot-learning-lab.cove50-v130-opened-failure-perturbation-plan.v1",
        "role": "post_v127_opened_seed_training_only_not_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**source, "runner": sha(Path(__file__)),
                          "v101_plan": sha(V101_PLAN), "v101_matrix": sha(V101_MATRIX),
                          "v113_source_plan": sha(V113_PLAN),
                          "v128_sealed_reserve": sha(V128_RESERVE)},
        "case_count": 24, "visit_count": 48,
        "diagnostic_goal_shift_body_y_m": SHIFT_M,
        "selection_reason": ("The first eight V127 new-seed variations around "
                             "V101 failure 22021 all passed the unchanged "
                             "native gate and changed route. These new physical "
                             "perturbations retain the three opened failing "
                             "seeds to acquire failure-neighborhood training "
                             "evidence without pretending seed independence."),
        "method": ("Freeze every planned condition before V130 contact. Replay "
                   "unchanged automatically routed observed-RGB reach and the "
                   "same route with a prescribed −2 mm body-Y target on each "
                   "condition. Keep all full native outcomes, including "
                   "visibility, support, rated-effort, action, collision and "
                   "contact failures. Seed, anchor and simulator toy truth are "
                   "audit fields, never future actor inputs. V128 stays sealed."),
        "cases": cases,
    }
    checked_write(PLAN, plan)
    return plan


def run_pair(case: dict, sources: dict) -> dict:
    base = baseline.run_case({**case, "id": case["id"] + "-zero"},
                             {"source_sha256": sources}, DEST)
    shift = shifted.run_case({**case, "id": case["id"] + "-neg2",
                              "diagnostic_goal_shift_body_y_m": SHIFT_M},
                             {"source_sha256": {**sources,
                                                "evaluator": sources["shift_evaluator"]}},
                             DEST)
    if base["selection_mode"] != shift["selection_mode"]:
        raise ValueError(f"Automatic route drift within pair {case['id']}")
    return {
        "id": case["id"], "seed": case["seed"],
        "cohort": case["cohort"],
        "anchor_opened_failure_seed": case["anchor_opened_failure_seed"],
        "route_variant": case["route_variant"],
        "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
        "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
        "toy_offset_x_m": case["toy_offset_x_m"],
        "toy_offset_y_m": case["toy_offset_y_m"],
        "selection_mode": base["selection_mode"],
        "baseline_contact_s": base["contact_s"],
        "shifted_contact_s": shift["contact_s"],
        "contact_delta_s": round(shift["contact_s"] - base["contact_s"], 3),
        "baseline_pass": base["pass"], "shifted_pass": shift["pass"],
        "baseline_margin_1p2_s_met": base["margin_1p2_s_met"],
        "shifted_margin_1p2_s_met": shift["margin_1p2_s_met"],
        "baseline_faults": base["faults"], "shifted_faults": shift["faults"],
        "baseline_result": base["result"], "baseline_trace": base["trace"],
        "shifted_result": shift["result"], "shifted_trace": shift["trace"],
    }


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    matrix = {
        "schema": "robot-learning-lab.cove50-v130-opened-failure-pairs.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "complete": len(ordered) == plan["case_count"],
        "case_count": len(ordered), "planned_count": plan["case_count"],
        "baseline_pass_count": sum(row["baseline_pass"] for row in ordered),
        "shifted_pass_count": sum(row["shifted_pass"] for row in ordered),
        "rescue_count": sum(not row["baseline_pass"] and row["shifted_pass"]
                            for row in ordered),
        "regression_count": sum(row["baseline_pass"] and not row["shifted_pass"]
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
        raise ValueError("Use one or two paired native evaluators")
    plan = freeze()
    print(f"V130 plan sha256={sha(PLAN)} {plan['case_count']} pairs", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    old = DEST / "matrix.json"
    existing = json.loads(old.read_text()) if old.exists() else None
    if existing and existing["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing native matrix belongs to another plan")
    rows = {row["id"]: row for row in existing["cases"]} if existing else {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_pair, case, plan["source_sha256"]): case
                   for case in plan["cases"] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V130 {row['id']}: unchanged {row['baseline_contact_s']:.2f} s "
                  f"({'pass' if row['baseline_pass'] else 'fail'}), "
                  f"−2 mm {row['shifted_contact_s']:.2f} s "
                  f"({'pass' if row['shifted_pass'] else 'fail'}); "
                  f"{matrix['case_count']}/24 pairs", flush=True)


if __name__ == "__main__":
    main()
