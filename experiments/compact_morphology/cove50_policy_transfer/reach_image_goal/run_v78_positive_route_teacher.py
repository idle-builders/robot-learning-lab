"""Probe prescribed goal offsets on the first V73 different-anchor contact miss."""

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
    run_v57_contact_actor_screen as v57,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v78"
PLAN = OUT / "positive-route-prescribed-offset-plan.json"
DEST = OUT / "native-prescribed-offsets"
V73_PLAN = BASE / "v73/route-holdout-run-plan.json"
FAILURE = BASE / "v73/opened-route-holdout/seed-11003-positive-01-pitch-low.json"
EVALUATOR = Path(__file__).with_name("evaluate_ybias_probe_v61.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v58.py")
CASE_ID = "positive-01-pitch-low"
BIASES_M = (-.006, -.004, -.002, 0.0, .002, .004, .006)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    failed = json.loads(FAILURE.read_text())
    if (failed["phases"]["reach"]["continuous_finger_contact_s"] >= 1 or
            not failed["off_authored_route_probe"] or
            failed["goal_bias_body_xy_m"] != [0.0, 0.0] or
            failed["selected_reach_checkpoint_sha256"] != sha(v57.ACTOR)):
        raise ValueError("Expected exact V73 uncorrected V57 arm contact miss")
    source = json.loads(V73_PLAN.read_text())
    case = next(row for row in source["cases"] if row["id"] == CASE_ID)
    if (case["seed"] != failed["seed"] or case["rail_v"] != failed["rail_v"] or
            case["delay_ms"] != failed["delay_ms"]):
        raise ValueError("V73 case and failed native report disagree")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(v57.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v78-prescribed-route-offset-plan.v1",
        "role": "post_holdout_experimenter_prescribed_motion_teacher_search_not_controller",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources, "runner_sha256": sha(Path(__file__)),
        "v73_plan_sha256": sha(V73_PLAN),
        "failed_v73_report_sha256": sha(FAILURE),
        "case_count": len(BIASES_M),
        "cases": [{**case, "diagnostic_y_bias_m": bias} for bias in BIASES_M],
        "gate": "Run seven symmetric ±6 mm body-Y goal offsets on the exact V73 pitch-low miss. Count full native support, rated effort, >=1 s uninterrupted fingertip contact, floor, self-contact, clipping and <=10 mm RGB error. A useful teacher needs >=1.2 s contact. These prescribed offsets are post-holdout diagnostics, not automatic policy or a new validation set.",
        "design": "Keep the failed seed, toy, voltage, delay, gaze, route, V57 actor and observed RGB entry fixed. The V70 controller applied no route correction at this view. Probe signed body-Y offsets in 2 mm steps without using toy truth for control. Keep all failures and full native traces.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V78 teacher plan differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    bias = case["diagnostic_y_bias_m"]
    tag = f"{bias * 1000:+.0f}mm".replace("+", "plus").replace("-", "minus")
    result, trace, log = (DEST / f"seed-{case['seed']}-{CASE_ID}-{tag}.{suffix}"
                          for suffix in ("json", "npz", "log"))
    sources = v16.SOURCES
    command = [sys.executable, str(EVALUATOR),
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
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--diagnostic-y-bias-m", str(bias),
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
            raise RuntimeError(f"V78 prescribed route replay failed for {tag}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if any(report.get(key) != value for key, value in {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v61_prescribed_y_bias_off_route_diagnostic_not_qualified",
        "diagnostic_y_bias_m": bias,
        "goal_bias_body_xy_m": [0.0, bias],
        "off_authored_route_probe": True,
    }.items()):
        raise ValueError(f"V78 source or prescribed bias mismatch: {result}")
    if (report["pitch_offset_rad"] != case["pitch_offset_rad"] or
            report["roll_clearance_rad"] != case["roll_clearance_rad"]):
        raise ValueError(f"V78 route differs from failure: {result}")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event.get("time_s") == report["reach_start_s"]), None)
    error = None if entry is None else entry.get("xy_error_truth_mm_scoring_only")
    if error is None or error > 10:
        faults.append("RGB goal error above 10 mm or unavailable")
    if report.get("image_goal_missing_at_s") is not None:
        faults.append("image goal missing during reach")
    return {"id": tag, "seed": case["seed"], "diagnostic_y_bias_m": bias,
            "pass": not faults,
            "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
            "entry_goal_error_mm_scoring_only": error,
            "faults": faults,
            "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
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
        futures = {pool.submit(run_case, case, plan): case["diagnostic_y_bias_m"]
                   for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["diagnostic_y_bias_m"]] = row
            ordered = [rows[case["diagnostic_y_bias_m"]] for case in plan["cases"]
                       if case["diagnostic_y_bias_m"] in rows]
            teachers = [r for r in ordered if r["pass"] and r["contact_s"] >= 1.2]
            matrix = {
                "schema": "robot-learning-lab.cove50-v78-prescribed-route-offsets.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                "teacher_found": bool(teachers),
                "best_full_gate_teacher": max(teachers, key=lambda r: r["contact_s"])["id"] if teachers else None,
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "diagnostic_y_bias_m", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
