"""Varied native positive-camera sparse-image abstention screen."""

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
OUT = BASE / "v51"
PLAN = OUT / "positive-sparse-neighbor-plan.json"
DEST = OUT / "positive-sparse-neighbors"
EVALUATOR = v47.EVALUATOR
SELECTOR = v47.SELECTOR


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    previous_path = v47.DEST / "matrix.json"
    previous = json.loads(previous_path.read_text())
    hit = next((row for row in previous["cases"] if row["seed"] == 11003), None)
    if (not previous["complete"] or previous["case_count"] != 39 or
            hit is None or not hit["pass"] or
            not hit["positive_camera_large_sparse_abstention"]):
        raise ValueError("Expected complete V47 screen with 11003 positive abstention rescue")
    original = next(c for c in json.loads(prior.PLAN.read_text())["cases"]
                    if c["seed"] == 11003)
    variants = [
        (11003, .0005, .016, 10, 0, -1.3),
        (11003, .001, .016, 11, 5, -1.3),
        (11003, .0015, .016, 12, 10, -1.3),
        (11003, .001, .0155, 10, 10, -1.3),
        (11003, .001, .0165, 11, 0, -1.3),
        (11003, .0015, .0165, 12, 5, -1.3),
        (11003, .001, .016, 10, 5, -1.5),
        (11003, .0015, .016, 11, 10, -1.5),
        (11003, .002, .0165, 12, 0, -1.5),
        (11301, .001, .016, 10, 0, -1.3),
        (11302, .0015, .016, 11, 5, -1.3),
        (11303, .002, .0165, 12, 10, -1.3),
    ]
    cases = [
        {**original, "id": f"positive-{index:02d}", "seed": seed,
         "toy_offset_x_m": toy_x, "toy_offset_y_m": toy_y,
         "rail_v": rail, "delay_ms": delay, "gaze_head_yaw_rad": yaw,
         "development_role": ("opened_seed_motor_toy_gaze_neighbor" if seed == 11003
                              else "new_entry_seed_motor_toy_neighbor"),
         "baseline_v41_contact_s": None}
        for index, (seed, toy_x, toy_y, rail, delay, yaw) in enumerate(variants, 1)
    ]
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(screen.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v51-positive-sparse-neighbor-plan.v1",
        "role": "opened_and_new_entry_neighbor_development_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "v47_development_plan_sha256": sha(v47.PLAN),
        "v47_development_matrix_sha256": sha(previous_path),
        "case_count": len(cases), "cases": cases,
        "design": "Keep the V47 positive-camera large sparse-correction abstention, V41 actor and exact authored route. Vary toy X/Y around the 11003 33-pixel image boundary, 10/11/12 V, 0/5/10 ms delay, both legal gaze commands, and three entirely new entry seeds. Nine visits vary the previously opened 11003 entry seed; three test new entry states. Neither seed nor simulator toy truth enters control.",
        "gate": "All 12 native visits must pass support, rated-effort, >=1 s continuous fingertip contact, floor, clipping and RGB checks. Require at least three observed positive-camera large sparse-correction abstentions across the screen; a single 11003 rescue is insufficient. Any zero-pixel probe must physically reacquire >=10 pixels. This is development, not promotion or a disjoint score.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V51 positive-sparse plan differs from sources")
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
            raise RuntimeError(f"V51 positive-sparse evaluator failed for {ident}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    expected = {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v47_large_sparse_positive_camera_abstention_candidate",
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V51 source mismatch: {result}")
    faults = gate(report)
    entry = next((e for e in report["image_goal_capture_events"]
                  if e.get("time_s") == report["reach_start_s"]), None)
    camera = None if not entry else entry.get("camera_world_xyz_m")
    pixels = None if not entry else entry.get("orange_pixels")
    template_expected = bool(entry and screen.expected_template(entry))
    negative_expected = bool(
        template_expected and pixels is not None and
        24 <= pixels < 64 and camera is not None and
        camera[0] < -.05 and camera[1] < 0)
    correction = [0, 0] if not entry else entry.get("template_correction_xy_mm") or [0, 0]
    positive_expected = bool(
        template_expected and not negative_expected and
        pixels is not None and 24 <= pixels < 64 and camera is not None and
        camera[0] >= 0 and camera[1] < -.08 and
        correction[0] ** 2 + correction[1] ** 2 >= 36)
    if (report["negative_camera_sparse_abstention_entry"] != negative_expected or
            report["positive_camera_large_sparse_abstention_entry"] != positive_expected):
        faults.append("observable sparse-template rule mismatch")
    if entry and entry.get("template_used_for_control") != (
            template_expected and not negative_expected and not positive_expected):
        faults.append("observed template-use rule mismatch")
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
        "positive_camera_large_sparse_abstention": report["positive_camera_large_sparse_abstention_entry"],
        "negative_camera_sparse_abstention": report["negative_camera_sparse_abstention_entry"],
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
            positive_uses = sum(r["positive_camera_large_sparse_abstention"] for r in ordered)
            matrix = {
                "schema": "robot-learning-lab.cove50-v51-positive-sparse-neighbor-screen.v1",
                "role": "opened_and_new_entry_neighbor_development_not_fresh_score",
                "body_role": "rejected_historical_direct_frame_sandbox",
                "plan_sha256": sha(PLAN), "complete": len(ordered) == 12,
                "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                "physical_scan_count": scans, "positive_sparse_use_count": positive_uses,
                "screen_gate_met": len(ordered) == 12 and all(r["pass"] for r in ordered)
                                   and positive_uses >= 3,
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
