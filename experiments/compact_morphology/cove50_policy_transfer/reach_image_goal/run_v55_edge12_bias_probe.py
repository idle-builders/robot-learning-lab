"""Frozen opened goal-bias diagnostic on the failed new edge entry state."""

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
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
    run_v41_opened_screen as screen,
    run_v50_edge_neighbors as v50,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v55"
PLAN = OUT / "new-entry-edge-bias-plan.json"
DEST = OUT / "new-entry-edge-bias"
EVALUATOR = Path(__file__).with_name("evaluate_scanned_edge_bias_v48.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v45.py")
BIASES_M = (-.003, -.004, -.005)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    baseline_path = v50.DEST / "matrix.json"
    baseline = json.loads(baseline_path.read_text())
    failed = next((r for r in baseline["cases"] if r["id"] == "edge-12"), None)
    if (not baseline["complete"] or baseline["pass_count"] != 11 or
            failed is None or failed["pass"] or failed["seed"] != 11203 or
            failed["contact_s"] != .04 or failed["edge_goal_bias_applied"] or
            failed["entry_orange_pixels"] != 24):
        raise ValueError("Expected complete V50 new-entry edge-12 contact failure")
    case = next(c for c in json.loads(v50.PLAN.read_text())["cases"]
                if c["id"] == "edge-12")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(screen.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v55-new-entry-edge-bias-plan.v1",
        "role": "opened_new_entry_contact_margin_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "v50_neighbor_plan_sha256": sha(v50.PLAN),
        "v50_neighbor_matrix_sha256": sha(baseline_path),
        "case_count": len(BIASES_M),
        "cases": [{**case, "scoped_y_bias_m": bias} for bias in BIASES_M],
        "baseline_scoped_y_bias_m": -.002,
        "baseline_contact_s": .04,
        "design": "Keep V50 failed new entry seed 11203, toy, 24-pixel RGB estimate, saved actor, timing, gaze, motor and route fixed. Vary only the selected scoped-Y route's fixed body-Y bias to -3, -4 or -5 mm; V50's -2 mm baseline held 0.04 s. This asks whether an observable goal bias can provide a motor-feasible teacher for this new entry state. Simulator toy truth is scoring only.",
        "gate": "Each opened diagnostic must pass unchanged native support, motor, >=1 s continuous contact, floor, clipping and RGB <=10 mm gates. No physical zero-pixel scan was triggered in the baseline; do not claim scan rescue. A passing bias on one new seed cannot promote a selector.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V55 plan differs from sources")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    bias = case["scoped_y_bias_m"]
    tag = f"{bias * 1000:+.0f}mm".replace("+", "plus").replace("-", "minus")
    result, trace, log = (DEST / f"seed-11203-bias-{tag}.{suffix}"
                          for suffix in ("json", "npz", "log"))
    sources = v16.SOURCES
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(screen.ACTOR),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(sources["reference"]),
               "--geometry", str(sources["geometry"]),
               "--entry", "continuous", "--seed", str(case["seed"]),
               "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", str(case["pitch_offset_rad"]),
               "--roll-clearance-rad", str(case["roll_clearance_rad"]),
               "--terminal-roll-offset-rad", "-.08",
               "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9",
               "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7",
               "--capture-stride-steps", "5", "--max-goal-age-s", "9",
               "--scoped-y-bias-m", str(bias),
               "--out", str(result), "--trace", str(trace)]
    if not result.exists() or not trace.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([
            str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
            "/root/microduck_rl/src", "/workspace/microduck_rl/src",
            env.get("PYTHONPATH", "")])
        with log.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise RuntimeError(f"V55 evaluator failed for {bias}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    expected = {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v48_opened_scanned_edge_goal_bias_diagnostic",
        "goal_bias_body_xy_m": [0.0, bias],
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V55 source or bias mismatch: {result}")
    faults = gate(report)
    entry = next((e for e in report["image_goal_capture_events"]
                  if e.get("time_s") == report["reach_start_s"]), None)
    if (report["selection_mode"] != "v21_scoped_y_locked" or
            report["reacquisition_trigger_s"] is not None or
            report["reacquisition_probe_orange_pixels"] != 24):
        faults.append("V50 entry route or observed camera branch changed")
    if not entry or entry.get("orange_pixels") != 24 or not entry.get("used_for_control"):
        faults.append("reach entry did not use the same visible toy")
    error = None if not entry else entry.get("xy_error_truth_mm_scoring_only")
    if error is None or error > 10:
        faults.append("RGB goal error above 10 mm or unavailable")
    return {
        "seed": case["seed"], "scoped_y_bias_m": bias,
        "selection_mode": report["selection_mode"],
        "pass": not faults,
        "contact_s": report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0),
        "reacquisition_probe_orange_pixels": report["reacquisition_probe_orange_pixels"],
        "entry_orange_pixels": None if not entry else entry.get("orange_pixels"),
        "entry_goal_error_mm_scoring_only": error,
        "faults": faults,
        "result": str(result.relative_to(ROOT)),
        "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 5:
        raise ValueError("Use one to five native evaluators")
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, c, plan): c["scoped_y_bias_m"]
                   for c in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["scoped_y_bias_m"]] = row
            ordered = [rows[c["scoped_y_bias_m"]] for c in plan["cases"]
                       if c["scoped_y_bias_m"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v55-new-entry-edge-bias.v1",
                "role": "opened_new_entry_contact_margin_diagnostic_not_fresh_score",
                "body_role": "rejected_historical_direct_frame_sandbox",
                "plan_sha256": sha(PLAN), "complete": len(ordered) == len(BIASES_M),
                "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("scoped_y_bias_m", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
