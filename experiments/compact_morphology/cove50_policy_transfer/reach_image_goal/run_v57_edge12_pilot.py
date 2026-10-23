"""Test the trained V57 arm on V50's new-entry contact miss."""

from __future__ import annotations

import argparse
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
    run_v57_contact_actor_screen as v57,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v57/edge12-trained-arm-pilot"
PLAN = OUT / "plan.json"
RESULT = OUT / "seed-11203.json"
TRACE = OUT / "seed-11203.npz"
LOG = OUT / "seed-11203.log"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    old_path = v50.DEST / "matrix.json"
    old = json.loads(old_path.read_text())
    prior = next(row for row in old["cases"] if row["id"] == "edge-12")
    candidate = json.loads((v57.DEST / "matrix.json").read_text())
    if (not old["complete"] or old["pass_count"] != 11 or prior["pass"] or
            prior["contact_s"] != .04 or candidate["pass_count"] != 12 or
            not candidate["screen_gate_met"]):
        raise ValueError("Need V50's failed edge-12 and complete V57 arm screen")
    case = next(row for row in json.loads(v50.PLAN.read_text())["cases"]
                if row["id"] == "edge-12")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(v57.ACTOR), evaluator=sha(v50.EVALUATOR),
                   selector=sha(v50.SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v57-edge12-pilot-plan.v1",
        "role": "opened_new_entry_actor_diagnostic_not_native_qualification",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "runner_sha256": sha(Path(__file__)),
        "source_sha256": sources,
        "v50_matrix_sha256": sha(old_path),
        "v57_screen_matrix_sha256": sha(v57.DEST / "matrix.json"),
        "case": case,
        "gate": "Replay the exact V50 edge-12 failed native start with only V41-to-V57 arm replacement. Pass requires the existing support, rated-effort, contact, floor, clipping and RGB gates. This one opened pilot tests whether the new arm generalizes outside its positive-sparse training neighborhood; it is not an automatic edge rule or a fresh score.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen pilot plan differs from source evidence")
    PLAN.write_bytes(encoded)
    return plan


def run(plan: dict) -> None:
    case, sources = plan["case"], v16.SOURCES
    command = [sys.executable, str(v50.EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(v57.ACTOR),
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
               "--terminal-roll-mode", "entry-error-x", "--reach-duration-s", "9",
               "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7",
               "--capture-stride-steps", "5", "--max-goal-age-s", "9",
               "--out", str(RESULT), "--trace", str(TRACE)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
         "/root/microduck_rl/src", "/workspace/microduck_rl/src"]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    if not RESULT.exists() or not TRACE.exists():
        with LOG.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, env=env,
                                  stdout=stream, stderr=subprocess.STDOUT,
                                  check=False).returncode
        if code:
            raise RuntimeError(f"Native pilot failed to run: {LOG}")
    report = json.loads(RESULT.read_text())
    hashes = plan["source_sha256"]
    expected = {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "selected_reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v49_observed_sparse_edge_goal_bias_candidate",
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Pilot source or selected actor differs from frozen plan")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event.get("time_s") == report["reach_start_s"]), None)
    if not entry or not entry.get("used_for_control") or entry.get("orange_pixels", 0) < 10:
        faults.append("visible RGB goal not used at entry")
    if entry and entry.get("xy_error_truth_mm_scoring_only", 1e9) > 10:
        faults.append("RGB goal error above 10 mm")
    row = {
        "schema": "robot-learning-lab.cove50-v57-edge12-pilot.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "pass": not faults,
        "seed": case["seed"], "selection_mode": report["selection_mode"],
        "contact_s": report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0),
        "entry_orange_pixels": None if entry is None else entry["orange_pixels"],
        "entry_goal_error_mm_scoring_only": None if entry is None else entry["xy_error_truth_mm_scoring_only"],
        "edge_goal_bias_applied": report["edge_goal_bias_applied"],
        "faults": faults,
        "result": str(RESULT.relative_to(ROOT)), "trace": str(TRACE.relative_to(ROOT)),
    }
    (OUT / "pilot.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
    else:
        run(plan)


if __name__ == "__main__":
    main()
