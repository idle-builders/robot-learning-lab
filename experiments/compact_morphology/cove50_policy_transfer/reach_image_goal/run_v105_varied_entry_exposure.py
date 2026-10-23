"""Freeze varied opened RGB-entry development starts; do not score contact."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import sys

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v98_entry_exposure as v98,
)


ROOT = Path(__file__).resolve().parents[4]
OUT = v98.BASE / "v105"
PLAN = OUT / "varied-entry-exposure-plan.json"
DEST = OUT / "native-varied-entry-exposure"
SEEDS = tuple(range(19001, 19065))
VARIANTS = {
    "center": (.105, -.25),
    "pitch-low": (.095, -.25),
    "pitch-high": (.115, -.25),
    "roll-low": (.105, -.26),
    "roll-high": (.105, -.24),
}
EXPOSURE_GATE = {"new_specialist": 3, "natural_specialist": 2,
                 "rear_correction": 1}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shuffled_levels(values: tuple, count: int, rng: random.Random) -> list:
    levels = [values[i % len(values)] for i in range(count)]
    rng.shuffle(levels)
    return levels


def cases() -> list[dict]:
    # Independent balanced permutations prevent toy, motor, gaze and route from
    # being a single correlated sequence. These are development starts only.
    rng = random.Random(105)
    count = len(SEEDS)
    variants = shuffled_levels(tuple(VARIANTS), count, rng)
    volts = shuffled_levels((10, 11, 12), count, rng)
    delays = shuffled_levels((0, 5, 10), count, rng)
    gazes = shuffled_levels((-1.3, -1.5), count, rng)
    toy_xs = shuffled_levels((.0058, .00605, .0063, .00655), count, rng)
    toy_ys = shuffled_levels((.0057, .0059, .0061, .0063), count, rng)
    for _ in range(1000):
        tuples = {(variants[i], volts[i], delays[i], toy_xs[i], toy_ys[i],
                   gazes[i]) for i in range(count)}
        if len(tuples) == count:
            break
        rng.shuffle(toy_ys)
    else:
        raise ValueError("Could not make unique balanced entry tuples")
    result = []
    for i, seed in enumerate(SEEDS):
        variant = variants[i]
        pitch, roll = VARIANTS[variant]
        result.append({"id": f"entry-{seed}", "seed": seed,
                       "route_variant": variant, "rail_v": volts[i],
                       "delay_ms": delays[i], "toy_offset_x_m": toy_xs[i],
                       "toy_offset_y_m": toy_ys[i], "pitch_offset_rad": pitch,
                       "roll_clearance_rad": roll,
                       "gaze_head_yaw_rad": gazes[i]})
    return result


def frozen_plan() -> dict:
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("V105 is only a historical rejected-body development scan")
    source_hashes = dict(v98.freeze(persist=False)["source_sha256"])
    source_hashes["v98_runner"] = source_hashes.pop("runner")
    source_hashes["runner"] = sha(Path(__file__))
    return {
        "schema": "robot-learning-lab.cove50-v105-varied-entry-plan.v1",
        "role": "opened_varied_rgb_entry_exposure_development_not_contact_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": source_hashes,
        "case_count": len(SEEDS), "cases": cases(),
        "exposure_gate": EXPOSURE_GATE,
        "method": ("Before opening any 19001-19064 development seed, freeze 64 "
                   "unique, balanced toy X/Y, 10-12 V, 0-10 ms delay, physical "
                   "gaze -1.3/-1.5 rad and five legal pitch/roll route tuples. "
                   "Replay through only the first used-for-control RGB reach "
                   "frame with the unchanged V98 evaluator and V95 controller. "
                   "Record observed route, camera, RGB goal and pixels. No "
                   "fingertip contact is measured; none is a disjoint score."),
        "decision_gate": ("If all 64 entries exist and at least 3 new-specialist, "
                          "2 natural-specialist and 1 rear-correction routes "
                          "activate, use separately frozen full native opened "
                          "development visits to measure their actual benefit. "
                          "Otherwise diagnose selector coverage and train a "
                          "broader observed-RGB actor/router without changing "
                          "the ongoing V101 untouched score."),
    }


def freeze() -> dict:
    plan = frozen_plan()
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V105 source or plan changed")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    # V98's copied evaluator and source checks are reused exactly. Only the
    # frozen case conditions and output directory differ.
    v98.DEST = DEST
    row = v98.run_case(case, plan)
    report = json.loads((ROOT / row["result"]).read_text())
    features = report.get("selection_features") or {}
    row.update({
        "route_variant": case["route_variant"],
        "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
        "toy_offset_x_m": case["toy_offset_x_m"],
        "toy_offset_y_m": case["toy_offset_y_m"],
        "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
        "entry_error_xy_m": features.get("entry_error_xy_m"),
        "natural_specialist": (report["selection_mode"] == "v16_specialist_locked"
                               and not report["automatic_alternate_gaze_specialist_applied"]),
        "rear_correction": bool(report["automatic_rear_contact_bias_applied"]),
    })
    return row


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[c["id"]] for c in plan["cases"] if c["id"] in rows]
    entry_count = sum(row["entry_available"] for row in ordered)
    counts = {
        "new_specialist": sum(row["new_specialist_applied"] for row in ordered),
        "natural_specialist": sum(row["natural_specialist"] for row in ordered),
        "rear_correction": sum(row["rear_correction"] for row in ordered),
    }
    complete = len(ordered) == plan["case_count"]
    matrix = {
        "schema": "robot-learning-lab.cove50-v105-varied-entry.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "complete": complete,
        "case_count": len(ordered), "entry_available_count": entry_count,
        "exposure_counts": counts,
        "exposure_gate_met": complete and entry_count == plan["case_count"] and
        all(counts[key] >= required for key, required in EXPOSURE_GATE.items()),
        "cases": ordered,
    }
    temp = DEST / "matrix.json.tmp"
    temp.write_text(json.dumps(matrix, indent=2) + "\n")
    os.replace(temp, DEST / "matrix.json")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four preflight evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    old = DEST / "matrix.json"
    rows = {row["id"]: row for row in json.loads(old.read_text())["cases"]} if old.exists() else {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case for case in plan["cases"]
                   if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            print(json.dumps(row), flush=True)
            write_matrix(plan, rows)
    matrix = write_matrix(plan, rows)
    print(f"V105 varied entry exposure: {matrix['case_count']}/64; "
          f"gate {'met' if matrix['exposure_gate_met'] else 'failed'}", flush=True)


if __name__ == "__main__":
    main()
