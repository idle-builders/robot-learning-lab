"""Freeze and run camera-only entry screening on unused simulator seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v83_thin_route_pilot as v83,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v98"
PLAN = OUT / "entry-exposure-plan.json"
DEST = OUT / "native-entry-exposure"
EVALUATOR = Path(__file__).with_name("evaluate_v98_entry_exposure.py")
ORIGINAL = Path(__file__).with_name("evaluate_rear_two_region_v95.py")
SEEDS = tuple(range(18001, 18065))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(*, persist: bool = True) -> dict:
    sources = v83.v16.SOURCES
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("This exposure lesson is scoped to the rejected historical body")
    cases = [{"id": f"entry-{seed}", "seed": seed, "rail_v": 11,
              "delay_ms": 10, "toy_offset_x_m": .0062,
              "toy_offset_y_m": .006, "pitch_offset_rad": .105,
              "roll_clearance_rad": -.25, "gaze_head_yaw_rad": -1.3}
             for seed in SEEDS]
    plan = {
        "schema": "robot-learning-lab.cove50-v98-entry-exposure-plan.v1",
        "role": "camera_only_unused_seed_preflight_no_contact_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {
            "evaluator": sha(EVALUATOR), "original_v95_evaluator": sha(ORIGINAL),
            "runner": sha(Path(__file__)),
            "body": sha(sources["body"]),
            "pointer": sha(ROOT / "robot/ape-current-design.json"),
            "squat": sha(sources["squat"]),
            "actor": sha(v83.v57.ACTOR),
            "alternate_actor": sha(sources["alternate_actor"]),
            "specialist_actor": sha(sources["specialist_actor"]),
            "reference": sha(sources["reference"]),
            "geometry": sha(sources["geometry"]),
            "calibration": sha(sources["calibration"]),
        },
        "cases": cases,
        "case_count": len(cases),
        "method": ("At 11 V, 10 ms delay, fixed body-frame toy offset and physical "
                   "-1.3 rad gaze, replay each unused simulator seed only through "
                   "the first 50 Hz reach frame. Record actual used-for-control RGB "
                   "capture, camera pose and automatic branch decision. Do not inspect "
                   "fingertip contact or use any preflight as a native reach score."),
        "next_gate": ("Use camera and branch exposure only to predeclare a fresh "
                      "dynamic challenge after V97 prior75 retention passes; include "
                      "exposure-positive and exposure-negative seeds and independently "
                      "vary toy, motor and gaze in the full native scorer."),
    }
    if persist:
        OUT.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(plan, indent=2) + "\n").encode()
        if PLAN.exists() and PLAN.read_bytes() != encoded:
            raise ValueError("Frozen V98 entry-exposure plan or source changed")
        PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    sources = v83.v16.SOURCES
    stem = DEST / f"seed-{case['seed']}-{case['id']}"
    result, trace, log = (stem.with_suffix(suffix)
                          for suffix in (".json", ".npz", ".log"))
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(v83.v57.ACTOR),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(sources["reference"]),
               "--geometry", str(sources["geometry"]),
               "--entry", "continuous", "--seed", str(case["seed"]),
               "--rail-v", str(case["rail_v"]),
               "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", str(case["pitch_offset_rad"]),
               "--roll-clearance-rad", str(case["roll_clearance_rad"]),
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", ".02",
               "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7",
               "--capture-stride-steps", "5", "--max-goal-age-s", "9",
               "--out", str(result), "--trace", str(trace)]
    if not result.exists() or not trace.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
             "/root/microduck_rl/src", "/workspace/microduck_rl/src"]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        with log.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise RuntimeError(f"V98 entry preflight failed for {case['id']}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if (report["evaluator_sha256"] != hashes["evaluator"] or
            report["selected_pointer_sha256"] != hashes["pointer"] or
            report["body_sha256"] != hashes["body"] or
            report["reach_checkpoint_sha256"] != hashes["actor"] or
            not report["preflight_only"] or report["reach_duration_s"] != .02):
        raise ValueError(f"V98 entry source identity mismatch: {result}")
    entry = next((event for event in report["image_goal_capture_events"]
                  if event["used_for_control"]), None)
    if entry is None:
        return {"id": case["id"], "seed": case["seed"],
                "entry_available": False, "camera_region": False,
                "new_specialist_applied": False, "camera_xy_mm": None,
                "orange_pixels": None, "selection_mode": report["selection_mode"],
                "result": str(result.relative_to(ROOT)),
                "fault": "No used-for-control entry RGB capture"}
    x, y = entry["camera_world_xyz_m"][:2]
    limits = report["alternate_gaze_specialist_observable_thresholds"]
    camera_region = (limits["camera_world_x_range_m"][0] <= x <
                     limits["camera_world_x_range_m"][1] and
                     y < limits["camera_world_y_max_m"])
    return {"id": case["id"], "seed": case["seed"],
            "entry_available": True, "camera_region": camera_region,
            "new_specialist_applied": report["automatic_alternate_gaze_specialist_applied"],
            "camera_xy_mm": [round(1000*x, 3), round(1000*y, 3)],
            "orange_pixels": entry["orange_pixels"],
            "selection_mode": report["selection_mode"],
            "result": str(result.relative_to(ROOT)), "fault": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--pilot-seed", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four preflight evaluators")
    plan = freeze(persist=args.pilot_seed is None)
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    cases = plan["cases"]
    if args.pilot_seed is not None:
        cases = [{**cases[0], "id": f"pilot-{args.pilot_seed}",
                  "seed": args.pilot_seed}]
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case for case in cases}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            print(json.dumps(row), flush=True)
            if args.pilot_seed is None:
                ordered = [rows[case["id"]] for case in cases if case["id"] in rows]
                matrix = {
                    "schema": "robot-learning-lab.cove50-v98-entry-exposure.v1",
                    "role": plan["role"], "body_role": plan["body_role"],
                    "plan_sha256": sha(PLAN), "complete": len(ordered) == len(cases),
                    "case_count": len(ordered),
                    "camera_region_count": sum(row["camera_region"] for row in ordered),
                    "new_specialist_count": sum(row["new_specialist_applied"] for row in ordered),
                    "cases": ordered,
                }
                temp = DEST / "matrix.json.tmp"
                temp.write_text(json.dumps(matrix, indent=2) + "\n")
                os.replace(temp, DEST / "matrix.json")


if __name__ == "__main__":
    main()
