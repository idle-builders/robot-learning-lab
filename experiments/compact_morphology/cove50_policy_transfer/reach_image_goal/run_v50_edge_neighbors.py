"""Varied native edge-neighbor screen for the observed contact-bias rule."""

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
    run_v49_edge_pair as pair,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v50"
PLAN = OUT / "edge-neighbor-screen-plan.json"
DEST = OUT / "edge-neighbor-screen"
EVALUATOR = pair.EVALUATOR
SELECTOR = pair.SELECTOR


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    paired_path = pair.DEST / "matrix.json"
    paired = json.loads(paired_path.read_text())
    if (not paired["complete"] or paired["pass_count"] != 2 or
            sorted(row["contact_s"] for row in paired["cases"]) != [3.06, 3.16]):
        raise ValueError("Expected complete V49 physical-scan/static-gaze rescue pair")
    original = next(c for c in json.loads(prior.PLAN.read_text())["cases"]
                    if c["seed"] == 10806)
    variants = [
        (10806, .01175, .005, 10, 0, -1.3),
        (10806, .012, .005, 11, 5, -1.3),
        (10806, .01225, .005, 12, 10, -1.3),
        (10806, .012, .0045, 10, 10, -1.3),
        (10806, .012, .0055, 11, 0, -1.3),
        (10806, .0125, .0055, 12, 5, -1.3),
        (10806, .012, .005, 10, 5, -1.5),
        (10806, .01225, .005, 11, 10, -1.5),
        (10806, .0125, .0055, 12, 0, -1.5),
        (11201, .012, .005, 10, 0, -1.3),
        (11202, .01225, .005, 11, 5, -1.3),
        (11203, .0125, .0055, 12, 10, -1.3),
    ]
    cases = [
        {**original, "id": f"edge-{index:02d}", "seed": seed,
         "toy_offset_x_m": toy_x, "toy_offset_y_m": toy_y,
         "rail_v": rail, "delay_ms": delay, "gaze_head_yaw_rad": yaw,
         "development_role": ("opened_seed_motor_toy_gaze_neighbor" if seed == 10806
                              else "new_entry_seed_motor_toy_neighbor"),
         "baseline_v41_contact_s": None}
        for index, (seed, toy_x, toy_y, rail, delay, yaw) in enumerate(variants, 1)
    ]
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(screen.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v50-edge-neighbor-screen-plan.v1",
        "role": "opened_and_new_entry_neighbor_development_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "v49_pair_plan_sha256": sha(pair.PLAN),
        "v49_pair_matrix_sha256": sha(paired_path),
        "case_count": len(cases), "cases": cases,
        "design": "Keep the V49 observable edge-view bias, V47 image selector, V41 actor and exact authored route. Vary toy X/Y around the visibility boundary, 10/11/12 V, 0/5/10 ms delay, both legal gaze commands, and three entirely new entry seeds. Nine visits vary the previously opened 10806 entry seed; three test new entry states. Neither seed nor simulator toy truth enters control.",
        "gate": "All 12 native visits must pass support, rated-effort, >=1 s continuous fingertip contact, floor, clipping and RGB checks. Any zero-pixel probe must physically trigger a scan and reacquire >=10 pixels; require at least three physical scans and three edge-rule activations across the screen before calling the branch exposed. This is development, not promotion or a disjoint score.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V50 edge-neighbor plan differs from sources")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    ident = case["id"]
    result, trace, log = (DEST / f"seed-{case['seed']}-{ident}.{suffix}"
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
            raise RuntimeError(f"V50 edge-neighbor evaluator failed for {ident}: {log}")
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
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V50 source mismatch: {result}")
    faults = gate(report)
    entry = next((e for e in report["image_goal_capture_events"]
                  if e.get("time_s") == report["reach_start_s"]), None)
    camera = None if not entry else entry.get("camera_world_xyz_m")
    pixels = None if not entry else entry.get("orange_pixels")
    edge_expected = bool(
        report["selection_mode"] == "v21_scoped_y_locked" and
        pixels is not None and 24 <= pixels <= 50 and camera is not None and
        -.055 <= camera[0] <= -.040 and -.095 <= camera[1] <= -.085)
    if report["edge_goal_bias_applied"] != edge_expected:
        faults.append("observable edge goal-bias rule mismatch")
    expected_bias = (-.004 if edge_expected else -.002 if report["selection_mode"] in
                     ("v16_bias_y_locked", "v21_scoped_y_locked") else 0.0)
    if report["goal_bias_body_xy_m"] != [0.0, expected_bias]:
        faults.append("goal bias differs from observed entry rule")
    if report["reacquisition_probe_orange_pixels"] == 0 and not (
            report["reacquisition_trigger_s"] is not None and
            entry and entry.get("orange_pixels", 0) >= 10):
        faults.append("physical zero-pixel scan did not reacquire toy")
    if not entry or entry.get("orange_pixels", 0) < 10 or not entry.get("used_for_control"):
        faults.append("scanned reach entry did not use visible toy")
    error = None if not entry else entry.get("xy_error_truth_mm_scoring_only")
    if error is None or error > 10:
        faults.append("RGB goal error above 10 mm or unavailable")
    return {
        "id": ident, "seed": case["seed"],
        "development_role": case["development_role"],
        "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
        "toy_offset_x_m": case["toy_offset_x_m"],
        "toy_offset_y_m": case["toy_offset_y_m"],
        "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
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
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four native evaluators")
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, c, plan): c["id"]
                   for c in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[c["id"]] for c in plan["cases"] if c["id"] in rows]
            scans = sum(r["reacquisition_probe_orange_pixels"] == 0 and
                        r["reacquisition_trigger_s"] is not None for r in ordered)
            edge_uses = sum(r["edge_goal_bias_applied"] for r in ordered)
            matrix = {
                "schema": "robot-learning-lab.cove50-v50-edge-neighbor-screen.v1",
                "role": "opened_and_new_entry_neighbor_development_not_fresh_score",
                "body_role": "rejected_historical_direct_frame_sandbox",
                "plan_sha256": sha(PLAN), "complete": len(ordered) == 12,
                "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                "physical_scan_count": scans, "edge_bias_use_count": edge_uses,
                "screen_gate_met": len(ordered) == 12 and all(r["pass"] for r in ordered)
                                   and scans >= 3 and edge_uses >= 3,
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
