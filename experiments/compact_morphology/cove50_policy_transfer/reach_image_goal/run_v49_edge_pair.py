"""Opened pair for an observed sparse edge-view contact-bias rule."""

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
    run_v45_combined_screen as prior,
    run_v47_large_sparse_screen as v47,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v49"
PLAN = OUT / "opened-edge-pair-plan.json"
DEST = OUT / "opened-edge-pair"
EVALUATOR = Path(__file__).with_name("evaluate_edge_goal_bias_v49.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v47.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    scanned = BASE / "v45/scan-branch-probe-c/matrix.json"
    static = BASE / "v45/scan-timing-control/matrix.json"
    bias_pass = BASE / "v48/scanned-edge-bias/seed-10806-bias-minus4mm.json"
    baseline, paired, changed = (json.loads(path.read_text())
                                 for path in (scanned, static, bias_pass))
    if (not baseline["complete"] or baseline["pass_count"] != 0 or
            baseline["cases"][0]["contact_s"] != .78 or
            not paired["complete"] or paired["pass_count"] != 0 or
            paired["cases"][0]["contact_s"] != .5 or
            changed["phases"]["reach"]["continuous_finger_contact_s"] < 3 or
            changed["goal_bias_body_xy_m"] != [0.0, -.004]):
        raise ValueError("Expected paired edge failures and -4 mm bias rescue")
    original = next(c for c in json.loads(prior.PLAN.read_text())["cases"]
                    if c["seed"] == 10806)
    cases = [
        {**original, "id": "scanned", "toy_offset_x_m": .012,
         "development_role": "opened_physical_scan"},
        {**original, "id": "static-alternate-gaze", "toy_offset_x_m": .012,
         "gaze_head_yaw_rad": -1.5,
         "development_role": "opened_static_gaze_paired_control"},
    ]
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(screen.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v49-opened-edge-pair-plan.v1",
        "role": "opened_observable_rule_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "v47_development_plan_sha256": sha(v47.PLAN),
        "scanned_edge_probe_matrix_sha256": sha(scanned),
        "static_gaze_matrix_sha256": sha(static),
        "v48_minus4_report_sha256": sha(bias_pass),
        "case_count": 2, "cases": cases,
        "design": "Keep the V41 saved arm, V47 image selector and motor route. An entry-view rule chooses -4 rather than -2 mm body-Y goal bias when the observed toy has 24-50 pixels, camera world X -55 to -40 mm, camera world Y -95 to -85 mm, and the scoped-Y arm route was selected. The same toy and seed test physical scan and static alternate gaze. Neither seed identity nor simulator toy truth enters control.",
        "gate": "Both paired native visits must pass complete support, motor, >=1 s continuous contact, floor, clipping and RGB gates; the first must physically scan from zero pixels. A 2/2 result only justifies a neighboring varied development screen, not promotion.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V49 pair plan differs from sources")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    ident = case["id"]
    result, trace, log = (DEST / f"seed-10806-{ident}.{suffix}"
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
            raise RuntimeError(f"V49 evaluator failed for {ident}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    expected = {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v49_observed_sparse_edge_goal_bias_candidate",
        "goal_bias_body_xy_m": [0.0, -.004],
        "edge_goal_bias_applied": True,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V49 source or rule mismatch: {result}")
    faults = gate(report)
    entry = next((e for e in report["image_goal_capture_events"]
                  if e.get("time_s") == report["reach_start_s"]), None)
    if ident == "scanned" and (report["reacquisition_probe_orange_pixels"] != 0 or
                               report["reacquisition_trigger_s"] is None):
        faults.append("physical zero-pixel scan did not trigger")
    if not entry or entry.get("orange_pixels", 0) < 10 or not entry.get("used_for_control"):
        faults.append("scanned reach entry did not use visible toy")
    error = None if not entry else entry.get("xy_error_truth_mm_scoring_only")
    if error is None or error > 10:
        faults.append("RGB goal error above 10 mm or unavailable")
    return {
        "id": ident, "seed": case["seed"],
        "development_role": case["development_role"],
        "selection_mode": report["selection_mode"],
        "edge_goal_bias_applied": report["edge_goal_bias_applied"],
        "pass": not faults,
        "contact_s": report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0),
        "reacquisition_probe_orange_pixels": report["reacquisition_probe_orange_pixels"],
        "reacquisition_trigger_s": report["reacquisition_trigger_s"],
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
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_case, c, plan): c["id"]
                   for c in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[c["id"]] for c in plan["cases"] if c["id"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v49-opened-edge-pair.v1",
                "role": "opened_observable_rule_diagnostic_not_fresh_score",
                "body_role": "rejected_historical_direct_frame_sandbox",
                "plan_sha256": sha(PLAN), "complete": len(ordered) == 2,
                "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
