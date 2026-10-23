"""Opened paired native diagnostic for the V47 negative-camera RGB miss."""

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
    run_v47_large_sparse_screen as v47,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v52"
PLAN = OUT / "negative-sparse-blend-plan.json"
DEST = OUT / "opened-negative-sparse-blend"
EVALUATOR = Path(__file__).with_name("evaluate_negative_sparse_probe_v52.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v52_probe.py")
BLENDS = (0.5, 1.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    failed_path = v47.DEST / "seed-11103.json"
    failed = json.loads(failed_path.read_text())
    event = failed["image_goal_capture_events"][0]
    if (failed["adaptive_variant"] != "v47_large_sparse_positive_camera_abstention_candidate" or
            failed["seed"] != 11103 or not failed["negative_camera_sparse_abstention_entry"] or
            event["template_correction_xy_mm"] != [8.0, -6.0] or
            event["xy_error_truth_mm_scoring_only"] != 10.223 or
            failed["phases"]["reach"]["continuous_finger_contact_s"] != 2.56):
        raise ValueError("Expected opened V47 11103 RGB localization failure")
    case = next(c for c in json.loads(v47.PLAN.read_text())["cases"] if c["seed"] == 11103)
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(screen.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v52-negative-sparse-blend-plan.v1",
        "role": "opened_paired_localization_and_contact_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "v47_development_plan_sha256": sha(v47.PLAN),
        "v47_failed_report_sha256": sha(failed_path),
        "case_count": len(BLENDS),
        "cases": [{**case, "negative_sparse_blend": blend} for blend in BLENDS],
        "baseline_v47_calibrated_rgb_contact_s": 2.56,
        "baseline_v47_calibrated_rgb_goal_error_mm": 10.223,
        "design": "Use exactly opened 11103 with the same actor, native dynamics, camera, motors and authored route. At sparse negative-camera capture, apply 50% or 100% of the observed template correction instead of V47's complete abstention. This paired probe measures whether localization can cross the 10 mm gate while keeping native contact. The simulator toy truth is scoring only.",
        "gate": "Each visit uses the unchanged native support, rated-effort, >=1 s continuous fingertip contact, floor, clipping and RGB <=10 mm gates. Both are diagnostics on one opened seed; neither promotes a policy.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V52 plan differs from sources")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    blend = case["negative_sparse_blend"]
    tag = f"blend-{blend:.1f}"
    result, trace, log = (DEST / f"seed-11103-{tag}.{suffix}"
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
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--negative-sparse-blend", str(blend),
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
            raise RuntimeError(f"V52 evaluator failed for {tag}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    expected = {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v52_opened_negative_sparse_blend_diagnostic",
        "negative_sparse_blend": blend,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V52 source or blend mismatch: {result}")
    faults = gate(report)
    event = next((e for e in report["image_goal_capture_events"]
                  if e.get("time_s") == report["reach_start_s"]), None)
    if (not event or not event.get("used_for_control") or
            not event.get("negative_camera_sparse_abstention") or
            event.get("negative_sparse_blend_diagnostic") != blend or
            event.get("orange_pixels") != 36 or
            event.get("template_correction_xy_mm") != [8.0, -6.0]):
        faults.append("opened negative-camera sparse rule or image changed")
    error = None if event is None else event.get("xy_error_truth_mm_scoring_only")
    if error is None or error > 10:
        faults.append("RGB goal error above 10 mm or unavailable")
    return {
        "seed": case["seed"], "negative_sparse_blend": blend,
        "pass": not faults,
        "contact_s": report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0),
        "entry_orange_pixels": None if event is None else event.get("orange_pixels"),
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
    if not 1 <= args.workers <= 2:
        raise ValueError("Use one or two native evaluators")
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["negative_sparse_blend"]
                   for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["negative_sparse_blend"]] = row
            ordered = [rows[blend] for blend in BLENDS if blend in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v52-negative-sparse-blend.v1",
                "role": "opened_paired_localization_and_contact_diagnostic_not_fresh_score",
                "body_role": "rejected_historical_direct_frame_sandbox",
                "plan_sha256": sha(PLAN), "complete": len(ordered) == len(BLENDS),
                "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("negative_sparse_blend", "pass", "contact_s", "entry_goal_error_mm_scoring_only", "faults")}), flush=True)


if __name__ == "__main__":
    main()
