"""Opened and varied native screen for the combined V45 recovery candidate."""

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
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v45"
PLAN = OUT / "combined-screen-plan.json"
DEST = OUT / "opened-and-varied-screen"
EVALUATOR = Path(__file__).with_name("evaluate_combined_recovery_v45.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v45.py")
SEEDS = (10806, 10810, 10827, 10830,
         10545, 10529, 10520, 10612, 10546,
         10807, 10808, 10826, 10829, 10831, 10602)


def variance_cases() -> list[dict]:
    """Twelve new development starts; none are in the sealed 109xx score."""
    result = []
    for i in range(12):
        result.append({
            "seed": 11001 + i, "development_role": "new_variance_training_development",
            "rail_v": (10, 11, 12)[i % 3],
            "delay_ms": (0, 5, 10)[(i // 3 + i % 3) % 3],
            "toy_offset_x_m": (.001, .002, .003)[i // 4],
            "toy_offset_y_m": (.012, .014, .016)[(i // 4 + i % 3) % 3],
            "gaze_head_yaw_rad": -1.3 if i % 2 == 0 else -1.5,
            "pitch_offset_rad": .105,
            "roll_clearance_rad": -.25,
            "baseline_v41_contact_s": None,
        })
    return result


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases() -> list[dict]:
    older = {row["seed"]: row for row in json.loads(opened.PLAN.read_text())["cases"]}
    newer = {row["seed"]: row for row in json.loads(fresh.PLAN.read_text())["cases"]}
    old_score = {row["seed"]: row for row in json.loads((opened.DEST / "matrix.json").read_text())["cases"]}
    new_score = {row["seed"]: row for row in json.loads((fresh.DEST / "matrix.json").read_text())["cases"]}
    opened_cases = [{**(newer[seed] if seed in newer else older[seed]),
                     "pitch_offset_rad": .105, "roll_clearance_rad": -.25,
                     "baseline_v41_contact_s": (new_score[seed] if seed in new_score else old_score[seed])["contact_s"],
                     "development_role": ("fresh_failure_reopened" if seed in (10806, 10810, 10827, 10830) else
                                          "fresh_passing_control" if seed in newer else
                                          "older_passing_control")}
                    for seed in SEEDS]
    return opened_cases + variance_cases()


def freeze() -> dict:
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(screen.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v45-combined-screen-plan.v1",
        "role": "postscore_opened_failures_controls_and_new_training_variance",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v41_fresh_matrix_sha256": sha(fresh.DEST / "matrix.json"),
        "v41_opened_matrix_sha256": sha(opened.DEST / "matrix.json"),
        "v42b_screen_matrix_sha256": sha(BASE / "v42b/opened-reacquire-screen/matrix.json"),
        "v43_effort_matrix_sha256": sha(BASE / "v43/opened-effort-probe/matrix.json"),
        "v43b_thin_margin_matrix_sha256": sha(BASE / "v43b/opened-thin-margin/matrix.json"),
        "v44b_sparse_matrix_sha256": sha(BASE / "v44b/opened-sparse-probe/matrix.json"),
        "v44c_negative_camera_matrix_sha256": sha(BASE / "v44c/opened-negative-camera/matrix.json"),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(SEEDS) + 12,
        "cases": cases(),
        "gate": "All 15 reopened failures/controls and 12 new varied development starts must pass the unchanged native support, rated-effort, >=1 s continuous contact, floor, clipping and RGB gates before full retention. If the entry probe lacks toy pixels, a real head turn must find the target at the delayed reach entry. Image control may only abstain from sparse templates when observed toy pixels are 24-63 and capture camera world X<-0.05 m, Y<0 m. This screen is development, not a new disjoint score or new-body qualification.",
        "design": "Combine V43's 2.5 s motor arm preposition, V42b's physical zero-pixel head reacquisition, and a restricted negative-X/negative-Y sparse-template abstention motivated by V44b/V44c. Retain V41's saved actor and all other route, motor and RGB rules. The 12 new 110xx starts vary seed, toy XY by +/-2 mm, 10/11/12 V, 0/5/10 ms delay and legal gaze. Pitch and clearance remain at checkpoint-declared .105/-.25 rad; route variation requires a separately trained actor and is not claimed here.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V45 combined screen plan differs from source")
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
            raise RuntimeError(f"V45 native evaluator failed for seed {seed}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if any(report.get(key) != value for key, value in {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v45_slow_preposition_reacquire_negative_camera_sparse_abstention_candidate",
    }.items()):
        raise ValueError(f"V45 source identity mismatch: {result}")
    faults = gate(report)
    entry_time = report["reach_start_s"]
    entry = next((e for e in report["image_goal_capture_events"]
                  if e.get("time_s") == entry_time), None)
    if not entry or not entry.get("used_for_control") or not entry.get("template_applied"):
        faults.append("image template unavailable at reach entry")
    if entry:
        expected_abstain = bool(screen.expected_template(entry) and
                                24 <= entry["orange_pixels"] < 64 and
                                entry["camera_world_xyz_m"][0] < -.05 and
                                entry["camera_world_xyz_m"][1] < 0)
        if entry.get("negative_camera_sparse_abstention") != expected_abstain:
            faults.append("observable sparse-quadrant rule mismatch")
        if entry.get("template_used_for_control") != (
                screen.expected_template(entry) and not expected_abstain):
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
    previous = json.loads((fresh.DEST / "matrix.json").read_text())
    if not previous["complete"] or previous["case_count"] != 32 or previous["pass_count"] == 32:
        raise ValueError("Postscore V45 screen requires complete failed V41 disjoint score")
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["seed"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[c["seed"]] for c in plan["cases"] if c["seed"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v45-combined-screen.v1",
                      "role": "opened_and_varied_native_development_not_fresh_score",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "selection_mode", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
