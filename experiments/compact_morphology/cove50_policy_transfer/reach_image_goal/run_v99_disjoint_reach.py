"""Freeze an untouched varied reach score; run only after V97 retention passes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v99"
PLAN = OUT / "disjoint-reach-plan.json"
DEST = OUT / "native-disjoint"
PILOT_PLAN = BASE / "v95/two-region-pilot-plan.json"
ROUTES = {
    "center": (.105, -.25),
    "pitch-low": (.095, -.25),
    "pitch-high": (.115, -.25),
    "roll-low": (.105, -.26),
    "roll-high": (.105, -.24),
}
STAGE_COUNTS = {"variance16": 16, "controls4": 4, "routes63": 63,
                "prior75": 75}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shuffled(values: list, rng: random.Random) -> list:
    result = values.copy()
    rng.shuffle(result)
    return result


def cases_from_anchor(anchor: dict) -> list[dict]:
    rng = random.Random(20261012)
    toy_x = shuffled([.0058, .00605, .0063, .00655] * 8, rng)
    toy_y = shuffled([.0057, .0059, .0061, .0063] * 8, rng)
    rail = shuffled([10] * 11 + [11] * 10 + [12] * 11, rng)
    delay = shuffled([0] * 11 + [5] * 10 + [10] * 11, rng)
    gaze = shuffled([-1.1] * 8 + [-1.3] * 16 + [-1.5] * 8, rng)
    route = shuffled(["center"] * 8 + ["pitch-low", "pitch-high",
                                      "roll-low", "roll-high"] * 6, rng)
    base = {key: value for key, value in anchor.items()
            if key not in ("id", "seed", "role", "prior_role",
                           "baseline_contact_s", "v73_contact_s")
            and not key.startswith("expected_")}
    cases = []
    for index, seed in enumerate(range(22001, 22033)):
        variant = route[index]
        pitch, roll = ROUTES[variant]
        cases.append({**base, "id": f"disjoint-{seed}", "seed": seed,
                      "route_variant": variant, "pitch_offset_rad": pitch,
                      "roll_clearance_rad": roll,
                      "toy_offset_x_m": toy_x[index],
                      "toy_offset_y_m": toy_y[index],
                      "rail_v": rail[index], "delay_ms": delay[index],
                      "gaze_head_yaw_rad": gaze[index]})
    return cases


def freeze() -> dict:
    pilot = json.loads(PILOT_PLAN.read_text())
    hashes = pilot["source_sha256"]
    if (hashes["evaluator"] != sha(native.EVALUATOR) or
            hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json") or
            json.loads((ROOT / "robot/ape-current-design.json").read_text())[
                "status"] != "rejected_no_selected_mechanical_design"):
        raise ValueError("V99 source or rejected-body pointer changed")
    anchor = next(case for case in pilot["cases"]
                  if case["id"] == "rear-gaze-minus1p3-zero")
    cases = cases_from_anchor(anchor)
    plan = {
        "schema": "robot-learning-lab.cove50-v99-disjoint-reach-plan.v1",
        "role": "untouched_native_reach_score_after_v97_full_retention",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "case_count": len(cases), "cases": cases,
        "source_sha256": hashes,
        "pilot_plan_sha256": sha(PILOT_PLAN),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "predecessor": ("V97 variance16, controls4, routes63 and prior75 must each "
                        "complete with its source-matched frozen screen gate met."),
        "seed_integrity": ("Seeds 22001–22032 are untouched by V98's 18001–18064 "
                           "camera-entry scan. All 32 conditions are fixed before "
                           "any of these seeds is simulated."),
        "variation": ("32 unique starts; independently shuffled toy X/Y, 10–12 V, "
                      "0–10 ms delay, physical gaze −1.1/−1.3/−1.5 rad, and "
                      "five authored or nearby pitch/roll routes. No case's "
                      "expected branch or outcome is inferred from a preflight."),
        "gate": ("All 32 uninterrupted native visits pass RGB goal, support, "
                 "rated motor, action and collision gates with at least 1.2 s "
                 "continuous fingertip contact; at least three automatic new "
                 "specialist uses, two natural specialist controls and one older "
                 "rear correction. Insufficient branch exposure fails this score."),
        "limit": ("Even a passing score would establish only the rejected "
                  "historical Direct Frame sandbox, not a selected mechanical "
                  "body, grasp, placement or composed task."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V99 plan or source changed")
    PLAN.write_bytes(encoded)
    return plan


def require_retention() -> None:
    for stage, count in STAGE_COUNTS.items():
        root = BASE / f"v97/native-{stage}-retention"
        path = root / "matrix.json"
        if not path.exists():
            raise ValueError(f"V97 {stage} retention has not run")
        result = json.loads(path.read_text())
        plan_path = BASE / f"v97/{stage}-retention-plan.json"
        if not (result["complete"] and result["screen_gate_met"] and
                result["case_count"] == count and result["pass_count"] == count and
                result["plan_sha256"] == sha(plan_path)):
            raise ValueError(f"V97 {stage} frozen retention gate has not passed")


def write_matrix(plan: dict, rows: dict) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    complete = len(ordered) == plan["case_count"]
    automatic = sum(row["automatic_alternate_gaze_specialist_applied"]
                    for row in ordered)
    natural = sum(row["selection_mode"] == "v16_specialist_locked" and
                  not row["automatic_alternate_gaze_specialist_applied"]
                  for row in ordered)
    rear = sum(row["automatic_rear_contact_bias_applied"] for row in ordered)
    passed = sum(row["pass"] for row in ordered)
    gate = (complete and passed == plan["case_count"] and
            all(row["contact_s"] >= 1.2 for row in ordered) and
            automatic >= 3 and natural >= 2 and rear >= 1)
    matrix = {
        "schema": "robot-learning-lab.cove50-v99-disjoint-reach.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "complete": complete,
        "case_count": len(ordered), "pass_count": passed,
        "minimum_contact_s": min((row["contact_s"] for row in ordered),
                                 default=None),
        "automatic_specialist_use_count": automatic,
        "natural_specialist_count": natural,
        "rear_bias_use_count": rear,
        "score_gate_met": gate, "cases": ordered,
    }
    temp = DEST / "matrix.json.tmp"
    temp.write_text(json.dumps(matrix, indent=2) + "\n")
    os.replace(temp, DEST / "matrix.json")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four native evaluators")
    plan = freeze()
    print(f"frozen {PLAN} sha256={sha(PLAN)}", flush=True)
    if args.freeze_only:
        return
    require_retention()
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(native.run_case, case, plan, DEST): case
                   for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "selection_mode",
                               "automatic_alternate_gaze_specialist_applied",
                               "automatic_rear_contact_bias_applied", "faults")}),
                  flush=True)
    print(json.dumps({key: matrix[key] for key in
                      ("case_count", "pass_count", "automatic_specialist_use_count",
                       "natural_specialist_count", "rear_bias_use_count",
                       "score_gate_met")}), flush=True)


if __name__ == "__main__":
    main()
