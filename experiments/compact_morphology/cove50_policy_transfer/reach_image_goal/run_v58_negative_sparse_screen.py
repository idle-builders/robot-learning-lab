"""Opened native retention screen for the observed V58 negative-sparse rule."""

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
    run_v41_fresh_challenge as fresh,
    run_v41_opened_gate as opened,
    run_v41_opened_screen as screen,
    run_v45_combined_screen as prior,
    run_v47_large_sparse_screen as v47,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v58"
PLAN = OUT / "development-screen-plan.json"
DEST = OUT / "opened-and-varied-screen"
EVALUATOR = Path(__file__).with_name("evaluate_negative_sparse_partial_v58.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v58.py")


def variance_cases() -> list[dict]:
    """Twelve new nearby starts; none are in the sealed 109xx score."""
    result = []
    for i in range(12):
        result.append({
            "seed": 11101 + i, "development_role": "new_positive_camera_neighbor_development",
            "rail_v": (10, 11, 12)[i % 3],
            "delay_ms": (0, 5, 10)[(i // 3 + i % 3) % 3],
            "toy_offset_x_m": (0.0, .001, .002)[i // 4],
            "toy_offset_y_m": (.014, .016, .018)[(i // 4 + i % 3) % 3],
            "gaze_head_yaw_rad": -1.3 if i % 2 == 0 else -1.5,
            "pitch_offset_rad": .105,
            "roll_clearance_rad": -.25,
            "baseline_v41_contact_s": None,
        })
    return result


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases() -> list[dict]:
    frozen = json.loads(v47.PLAN.read_text())["cases"]
    if len(frozen) != 39:
        raise ValueError("Expected 39 frozen V47 development starts")
    return frozen


def freeze() -> dict:
    previous_path = v47.DEST / "matrix.json"
    previous = json.loads(previous_path.read_text())
    v54_path = BASE / "v54/opened-negative-sparse-margin/matrix.json"
    v54 = json.loads(v54_path.read_text())
    if (not previous["complete"] or previous["case_count"] != 39 or
            previous["pass_count"] != 37 or
            [r["seed"] for r in previous["cases"] if not r["pass"]] != [11103, 11109] or
            not v54["complete"] or v54["pass_count"] != 6):
        raise ValueError("V58 requires V47 RGB failures and complete V54 margin probes")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(screen.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v58-negative-sparse-screen-plan.v1",
        "role": "postscore_opened_controls_and_new_neighbor_variance",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v47_screen_plan_sha256": sha(v47.PLAN),
        "v47_screen_matrix_sha256": sha(previous_path),
        "v54_negative_sparse_margin_matrix_sha256": sha(v54_path),
        "runner_sha256": sha(Path(__file__)),
        "case_count": 39,
        "cases": cases(),
        "gate": "All 39 frozen V47 opened/varied native starts must pass the unchanged support, rated-effort, >=1 s continuous contact, floor, clipping and RGB <=10 mm gates. The partial correction must activate from observed pixels, camera pose and correction magnitude only on applicable negative-camera sparse views. At least the two prior RGB failures must activate, and the prior 37 V47 passes must retain their gate. This is development, not disjoint scoring or selected-body qualification.",
        "design": "Keep V47's V41 arm, motor route, positive-camera abstention, physical zero-pixel scan and exact 39-case variance. In the negative-camera sparse region only, use 15% of a rendered-template correction when its observed magnitude is >=9 mm; retain V47 otherwise. V54 paired native probes on the two V47 RGB misses supplied the fraction and contact-margin evidence. No simulator toy truth or seed enters the rule.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V58 development screen plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    result, trace, log = (DEST / f"seed-{seed}.{suffix}" for suffix in ("json", "npz", "log"))
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
               "--entry", "continuous", "--seed", str(seed),
               "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", str(case["pitch_offset_rad"]),
               "--roll-clearance-rad", str(case["roll_clearance_rad"]),
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--out", str(result), "--trace", str(trace)]
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
            raise RuntimeError(f"V58 native evaluator failed for seed {seed}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if any(report.get(key) != value for key, value in {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v58_observed_negative_sparse_partial_correction_candidate",
    }.items()):
        raise ValueError(f"V58 source identity mismatch: {result}")
    faults = gate(report)
    entry_time = report["reach_start_s"]
    entry = next((e for e in report["image_goal_capture_events"]
                  if e.get("time_s") == entry_time), None)
    if not entry or not entry.get("used_for_control") or not entry.get("template_applied"):
        faults.append("image template unavailable at reach entry")
    if entry:
        expected_negative = bool(screen.expected_template(entry) and
                                24 <= entry["orange_pixels"] < 64 and
                                entry["camera_world_xyz_m"][0] < -.05 and
                                entry["camera_world_xyz_m"][1] < 0)
        correction = entry.get("template_correction_xy_mm") or [0, 0]
        expected_positive = bool(screen.expected_template(entry) and not expected_negative and
                                 24 <= entry["orange_pixels"] < 64 and
                                 entry["camera_world_xyz_m"][0] >= 0 and
                                 entry["camera_world_xyz_m"][1] < -.08 and
                                 (correction[0] ** 2 + correction[1] ** 2) >= 36)
        correction_magnitude = (correction[0] ** 2 + correction[1] ** 2) ** .5
        expected_partial = expected_negative and correction_magnitude >= 9.0
        if entry.get("negative_camera_sparse_abstention") != expected_negative:
            faults.append("observable sparse-quadrant rule mismatch")
        if entry.get("negative_camera_sparse_partial_correction") != expected_partial:
            faults.append("observable negative sparse partial-correction rule mismatch")
        if entry.get("positive_camera_large_sparse_abstention") != expected_positive:
            faults.append("observable large sparse-correction rule mismatch")
        if entry.get("template_used_for_control") != (
                screen.expected_template(entry) and (not expected_negative or expected_partial)
                and not expected_positive):
            faults.append("observable template rule mismatch")
    entry_error = None if entry is None else entry.get("xy_error_truth_mm_scoring_only")
    if entry_error is not None and entry_error > 10:
        faults.append("RGB goal error above 10 mm")
    if report.get("image_goal_missing_at_s") is not None:
        faults.append("image goal missing during reach")
    if report["reacquisition_probe_orange_pixels"] == 0 and not (
            report["reacquisition_trigger_s"] is not None and
            report["reacquisition_head_yaw_rad"] in (-1.3, -1.5) and
            entry and entry.get("orange_pixels", 0) >= 10):
        faults.append("physical camera reacquisition did not observe toy")
    return {"seed": seed, "development_role": case["development_role"],
            "baseline_v41_contact_s": case["baseline_v41_contact_s"],
            "selection_mode": report.get("selection_mode"), "pass": not faults,
            "contact_s": report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0),
            "reacquisition_trigger_s": report["reacquisition_trigger_s"],
            "reacquisition_probe_orange_pixels": report["reacquisition_probe_orange_pixels"],
            "reacquisition_head_yaw_rad": report["reacquisition_head_yaw_rad"],
            "negative_camera_sparse_abstention": report["negative_camera_sparse_abstention_entry"],
            "negative_camera_sparse_partial_correction": report["negative_camera_sparse_partial_correction_entry"],
            "positive_camera_large_sparse_abstention": report["positive_camera_large_sparse_abstention_entry"],
            "reach_start_s": entry_time,
            "entry_orange_pixels": None if entry is None else entry.get("orange_pixels"),
            "entry_goal_error_mm_scoring_only": entry_error,
            "faults": faults,
            "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["seed"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[c["seed"]] for c in plan["cases"] if c["seed"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v58-negative-sparse-screen.v1",
                      "role": "opened_and_neighbor_varied_native_development_not_fresh_score",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "partial_correction_use_count": sum(r["negative_camera_sparse_partial_correction"] for r in ordered),
                      "screen_gate_met": (len(ordered) == len(plan["cases"]) and
                                          all(r["pass"] for r in ordered) and
                                          sum(r["negative_camera_sparse_partial_correction"] for r in ordered) >= 2),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "selection_mode", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
