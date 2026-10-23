"""Test alternate saved branches on opened v16 fresh contact misses only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import run_v16_opened_matrix as v16

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V16 = BASE / "v16"
DEST = BASE / "v17/posthoc-forced-branch"
EVALUATOR = Path(__file__).with_name("evaluate_forced_branch_v17_diagnostic.py")
SOURCES = {**v16.SOURCES, "evaluator": EVALUATOR}
PAIRS = (
    (10401, "v16_v10_locked"), (10401, "v16_v10_height_locked"),
    (10409, "v16_v10_height_locked"),
    (10417, "v16_specialist_locked"), (10417, "v16_live"),
    (10421, "v16_specialist_locked"), (10421, "v16_live"),
    (10421, "v16_bias_y_locked"),
    (10422, "v16_specialist_locked"), (10422, "v16_live"),
    (10422, "v16_bias_y_locked"),
    (10433, "v16_specialist_locked"), (10433, "v16_live"),
)


def plan() -> dict:
    matrix_path = V16 / "fresh-reserved/matrix.json"
    fresh_plan_path = V16 / "fresh-reserved-plan.json"
    matrix = json.loads(matrix_path.read_text())
    if (matrix["plan_sha256"] != v16.sha(fresh_plan_path) or not matrix["complete"] or
            matrix["case_count"] != 36 or matrix["pass_count"] != 30):
        raise ValueError("Expected complete, sealed v16 fresh failure set")
    failed = {row["seed"] for row in matrix["cases"] if not row["pass"]}
    if failed != {10401, 10409, 10417, 10421, 10422, 10433}:
        raise ValueError("The opened failure set changed")
    cases = {case["seed"]: case for case in json.loads(fresh_plan_path.read_text())["cases"]}
    return {
        "schema": "robot-learning-lab.cove50-v17-forced-branch-plan.v1",
        "role": "posthoc_opened_diagnostic_only",
        "v16_fresh_plan_sha256": v16.sha(fresh_plan_path),
        "v16_fresh_matrix_sha256": v16.sha(matrix_path),
        "source_sha256": {name: v16.sha(path) for name, path in SOURCES.items()},
        "pairs": [{"case": cases[seed], "forced_mode": mode} for seed, mode in PAIRS],
        "purpose": "Probe whether the existing actor/route branch alone rescues each now-opened v16 miss; never modify the frozen 30/36 fresh score.",
    }


def run(pair: dict, frozen: dict) -> dict:
    case, mode = pair["case"], pair["forced_mode"]
    seed = case["seed"]
    stem = f"seed-{seed}-{mode}"
    result, trace, log = (DEST / f"{stem}.{ext}" for ext in ("json", "npz", "log"))
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(SOURCES["body"]),
               "--calibration-model", str(SOURCES["calibration"]),
               "--squat-checkpoint", str(SOURCES["squat"]),
               "--reach-checkpoint", str(SOURCES["actor"]),
               "--alternate-reach-checkpoint", str(SOURCES["alternate_actor"]),
               "--specialist-reach-checkpoint", str(SOURCES["specialist_actor"]),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(SOURCES["reference"]),
               "--geometry", str(SOURCES["geometry"]),
               "--entry", "continuous", "--seed", str(seed),
               "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--force-selection-mode", mode,
               "--out", str(result), "--trace", str(trace)]
    if not result.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
             "/workspace/microduck_rl/src"] +
            ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        with log.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise RuntimeError(f"Forced-branch evaluator failed: {stem}: {log}")
    report = json.loads(result.read_text())
    expected_actor = ("alternate_actor" if mode.startswith("v16_v10") else
                      "specialist_actor" if mode == "v16_specialist_locked" else "actor")
    if (report["evaluator_sha256"] != frozen["source_sha256"]["evaluator"] or
            report["selected_pointer_sha256"] != frozen["source_sha256"]["pointer"] or
            report["adaptive_variant"] != "v17_posthoc_forced_branch_diagnostic" or
            report["selection_mode"] != mode or report["posthoc_forced_selection_mode"] != mode or
            report["selected_reach_checkpoint_sha256"] != frozen["source_sha256"][expected_actor]):
        raise ValueError(f"Forced-branch report identity mismatch: {result}")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event["time_s"] == 7.5), None)
    if not entry or not entry["used_for_control"] or entry["orange_pixels"] < 10:
        faults.append("no valid RGB entry goal")
    if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                  entry["xy_error_truth_mm_scoring_only"] > 10):
        faults.append("RGB error above 10 mm")
    if report["image_goal_missing_at_s"] is not None:
        faults.append("image goal missing during reach")
    return {"seed": seed, "forced_mode": mode, "pass": not faults,
            "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
            "entry_error_xy_m": report["selection_features"]["entry_error_xy_m"],
            "both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
            "rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
            "faults": faults, "result": str(result.relative_to(ROOT)),
            "trace": str(trace.relative_to(ROOT)), "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Use one to three independent native evaluators")
    frozen = plan()
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "plan.json"
    planned = (json.dumps(frozen, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != planned:
            raise ValueError("Frozen forced-branch diagnostic plan changed")
    else:
        path.write_bytes(planned)
    if args.freeze_only:
        print(f"frozen {path} sha256={v16.sha(path)}")
        return
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, pair, frozen): (pair["case"]["seed"], pair["forced_mode"])
                   for pair in frozen["pairs"]}
        for future in as_completed(futures):
            key = futures[future]
            rows[key] = future.result()
            ordered = [rows[(pair["case"]["seed"], pair["forced_mode"])]
                       for pair in frozen["pairs"]
                       if (pair["case"]["seed"], pair["forced_mode"]) in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v17-forced-branch-native.v1",
                "role": "posthoc_opened_diagnostic_only",
                "plan_sha256": v16.sha(path),
                "complete": len(ordered) == len(PAIRS),
                "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(rows[key]), flush=True)


if __name__ == "__main__":
    main()
