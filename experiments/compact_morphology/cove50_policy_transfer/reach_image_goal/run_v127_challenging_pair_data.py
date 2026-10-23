"""Freeze new reach-failure neighborhoods and collect paired native training data.

The V128 disjoint conditions are frozen before any V127 contact visit. All
results remain on the rejected historical Direct Frame control sandbox.
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
OUT = BASE / "v127"
PLAN = OUT / "challenging-paired-development-plan.json"
DEST = OUT / "native-challenging-pairs"
RESERVE = BASE / "v128/disjoint-contact-score-plan.json"
V101_PLAN = BASE / "v101/disjoint-reach-plan.json"
V101_MATRIX = BASE / "v101/native-disjoint/matrix.json"
V113_PLAN = BASE / "v113/varied-paired-contact-data-plan.json"
ANCHORS = (22021, 22028, 22031)
SHIFT_M = -.002
NEAR_OFFSETS_TRAIN = (
    (-.00040, -.00020), (-.00040, .00020), (-.00020, .00040),
    (.00000, -.00040), (.00020, -.00020), (.00020, .00040),
    (.00040, .00000), (.00040, .00040),
)
NEAR_OFFSETS_RESERVED = (
    (-.00035, .00035), (.00035, -.00035),
    (-.00010, -.00035), (.00010, .00035),
)
MOTOR_TRAIN = ((12, 10), (11, 5), (10, 10), (12, 0),
               (11, 10), (10, 5), (12, 5), (11, 0))
MOTOR_RESERVED = ((10, 0), (11, 5), (12, 10), (11, 0))
ROUTES = {
    "center": (.105, -.25), "pitch-low": (.095, -.25),
    "pitch-high": (.115, -.25), "roll-low": (.105, -.26),
    "roll-high": (.105, -.24),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def checked_write(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError(f"Frozen source or plan changed: {path}")
    path.write_bytes(encoded)


def physical_key(case: dict) -> tuple:
    return tuple(case[k] for k in (
        "route_variant", "rail_v", "delay_ms", "toy_offset_x_m",
        "toy_offset_y_m", "pitch_offset_rad", "roll_clearance_rad",
        "gaze_head_yaw_rad"))


def near_case(anchor: dict, seed: int, index: int, reserved: bool) -> dict:
    dx, dy = (NEAR_OFFSETS_RESERVED if reserved else NEAR_OFFSETS_TRAIN)[index]
    rail_v, delay_ms = (MOTOR_RESERVED if reserved else MOTOR_TRAIN)[index]
    gaze = (-1.5, -1.3)[index % 2]
    return {
        "id": f"{'reserved' if reserved else 'challenge'}-{seed}",
        "seed": seed, "cohort": "heldout_near_failure" if reserved else
        "development_near_failure",
        "anchor_opened_failure_seed": anchor["seed"],
        "route_variant": anchor["route_variant"],
        "rail_v": rail_v, "delay_ms": delay_ms,
        "toy_offset_x_m": round(anchor["toy_offset_x_m"] + dx, 6),
        "toy_offset_y_m": round(anchor["toy_offset_y_m"] + dy, 6),
        "pitch_offset_rad": anchor["pitch_offset_rad"],
        "roll_clearance_rad": anchor["roll_clearance_rad"],
        "gaze_head_yaw_rad": gaze,
        "condition_delta_xy_mm": [round(dx * 1000, 3), round(dy * 1000, 3)],
    }


def broad_control(seed: int, index: int) -> dict:
    route = tuple(ROUTES)[index % len(ROUTES)]
    pitch, roll = ROUTES[route]
    return {
        "id": f"reserved-{seed}", "seed": seed,
        "cohort": "heldout_broad_route_control",
        "route_variant": route,
        "rail_v": (10, 11, 12)[index % 3],
        "delay_ms": (0, 5, 10)[(index // 3) % 3],
        "toy_offset_x_m": (.00592, .00616, .00640, .00652)[index % 4],
        "toy_offset_y_m": (.00592, .00616, .00640, .00652)[(index * 3 + 1) % 4],
        "pitch_offset_rad": pitch, "roll_clearance_rad": roll,
        "gaze_head_yaw_rad": (-1.3, -1.5)[index % 2],
    }


def freeze() -> tuple[dict, dict]:
    pointer = ROOT / "robot/ape-current-design.json"
    if json.loads(pointer.read_text())["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("V127/V128 are scoped only to the rejected historical body")
    v101_plan = json.loads(V101_PLAN.read_text())
    v101 = json.loads(V101_MATRIX.read_text())
    if (not v101["complete"] or v101["case_count"] != 32 or
            v101["pass_count"] != 29 or v101["plan_sha256"] != sha(V101_PLAN)):
        raise ValueError("V101 failure evidence changed")
    original = {case["seed"]: case for case in v101_plan["cases"]}
    scored = {row["seed"]: row for row in v101["cases"]}
    if any(scored[seed]["pass"] or scored[seed]["contact_s"] >= 1.0
           for seed in ANCHORS):
        raise ValueError("V101 contact-failure anchors changed")
    source = json.loads(V113_PLAN.read_text())["source_sha256"]
    if (source["body"] != sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml") or
            source["pointer"] != sha(pointer) or
            source["evaluator"] != sha(baseline.EVALUATOR) or
            source["shift_evaluator"] != sha(shifted.EVALUATOR)):
        raise ValueError("Replayed body, pointer or evaluators differ from V113")
    train = [near_case(original[anchor], 23101 + group * 8 + index,
                       index, False)
             for group, anchor in enumerate(ANCHORS)
             for index in range(8)]
    reserve_near = [near_case(original[anchor], 24101 + group * 4 + index,
                              index, True)
                    for group, anchor in enumerate(ANCHORS)
                    for index in range(4)]
    reserve_broad = [broad_control(24113 + index, index) for index in range(12)]
    reserved = reserve_near + reserve_broad
    if (len(train) != 24 or len(reserved) != 24 or
            len({case["seed"] for case in train + reserved}) != 48 or
            len({physical_key(case) for case in train + reserved}) != 48):
        raise ValueError("Development and disjoint physical conditions overlap")
    common = {
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**source, "runner": sha(Path(__file__)),
                          "v101_plan": sha(V101_PLAN), "v101_matrix": sha(V101_MATRIX),
                          "v113_source_plan": sha(V113_PLAN)},
        "anchor_opened_failure_seeds": list(ANCHORS),
        "sampling": ("Before any V127 contact visit, freeze 24 new seeded "
                     "development starts around three V101 contact failures "
                     "with eight distinct toy X/Y, voltage, delay and gaze "
                     "conditions per anchor. Separately freeze 24 new seeded "
                     "score starts: four other near-failure conditions per "
                     "anchor plus 12 broad center/pitch/roll controls. Seed and "
                     "opened failure identity are audit fields, never actor inputs."),
        "candidate_interface": ("At reach entry the candidate may use only "
                                "observed RGB goal error, camera pose, toy pixels, "
                                "selected automatic route, commanded route, "
                                "voltage, delay and gaze. Its bounded output is "
                                "the post-route body-Y goal correction in metres "
                                "(unchanged or −0.002). Seed, opened failure "
                                "identity, simulator toy truth and contact labels "
                                "are audit/training fields, never deployable inputs."),
    }
    development = {
        "schema": "robot-learning-lab.cove50-v127-challenging-paired-plan.v1",
        "role": "new_seed_failure_neighborhood_training_data_not_score",
        **common,
        "case_count": 24, "visit_count": 48,
        "diagnostic_goal_shift_body_y_m": SHIFT_M,
        "cases": train,
        "method": ("For each new start, replay unchanged observed-RGB routing "
                   "and a prescribed −2 mm body-Y target after the same route. "
                   "Keep full native RGB, contact, support, rated effort, action "
                   "and collision outcomes and failures. Fit only after all "
                   "planned pairs are reviewed; this data is development."),
    }
    score = {
        "schema": "robot-learning-lab.cove50-v128-disjoint-contact-plan.v1",
        "role": "contact_unseen_new_seed_score_reserved_before_v127_data",
        **common,
        "case_count": 24, "cases": reserved,
        "status": "sealed_no_contact_visits",
        "future_gate": ("Only after fitting an observed-input automatic candidate and "
                        "passing older-start retention, freeze its choices on "
                        "these 24 new starts without seeing contact outcomes. "
                        "Replay uninterrupted native contact under RGB, support, "
                        "rated effort, action and collision gates. Require all "
                        "24 full passes with >=1.2 s contact, no lost older "
                        "passing start, and at least one actual candidate "
                        "correction in each roll-low and roll-high near-failure "
                        "cohort. Report all 24, including failed visibility "
                        "or insufficient branch exposure, without replacement."),
    }
    # The plans refer to the exact same generated conditions without circular hashes.
    development["reserved_case_sha256"] = hashlib.sha256(
        (json.dumps(reserved, sort_keys=True) + "\n").encode()).hexdigest()
    score["development_case_sha256"] = hashlib.sha256(
        (json.dumps(train, sort_keys=True) + "\n").encode()).hexdigest()
    checked_write(PLAN, development)
    checked_write(RESERVE, score)
    return development, score


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
        "cohort": case["cohort"], "anchor_opened_failure_seed": case["anchor_opened_failure_seed"],
        "route_variant": case["route_variant"], "rail_v": case["rail_v"],
        "delay_ms": case["delay_ms"], "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
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
        "schema": "robot-learning-lab.cove50-v127-challenging-pairs.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "reserved_plan_sha256": sha(RESERVE),
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
    plan, reserve = freeze()
    print(f"V127 plan sha256={sha(PLAN)} 24 pairs; "
          f"V128 reserve sha256={sha(RESERVE)} {reserve['case_count']} starts",
          flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    old = DEST / "matrix.json"
    existing = json.loads(old.read_text()) if old.exists() else None
    if existing and (existing["plan_sha256"] != sha(PLAN) or
                     existing["reserved_plan_sha256"] != sha(RESERVE)):
        raise ValueError("Existing development matrix belongs to another plan")
    rows = {row["id"]: row for row in existing["cases"]} if existing else {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_pair, case, plan["source_sha256"]): case
                   for case in plan["cases"] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V127 {row['id']}: unchanged {row['baseline_contact_s']:.2f} s "
                  f"({'pass' if row['baseline_pass'] else 'fail'}), "
                  f"−2 mm {row['shifted_contact_s']:.2f} s "
                  f"({'pass' if row['shifted_pass'] else 'fail'}); "
                  f"{matrix['case_count']}/24 pairs", flush=True)


if __name__ == "__main__":
    main()
