"""Pair a wider observable template rule against five opened V32 starts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import run_v16_opened_matrix as v16

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v35"
PLAN = OUT / "wider-rgb-screen-plan.json"
DEST = OUT / "wider-rgb-screen"
EVALUATOR = Path(__file__).with_name("evaluate_selective_goal_v35.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v35.py")
TEMPLATE = Path(__file__).with_name("template_virtual_rgb_goal_v30.py")
EXPECTED_VARIANT = "v35_wider_selective_template_height_router_candidate"
RECENT_PLAN = BASE / "v32/opened-gate-plan.json"
RECENT = BASE / "v32/opened-native/matrix.json"
PRIOR_PLAN = BASE / "v32/prior-failure-screen-plan.json"
PRIOR = BASE / "v32/prior-failure-screen/matrix.json"
TARGETS = (10538, 10612, 10608, 10604, 10514)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    recent = json.loads(RECENT.read_text())
    prior = json.loads(PRIOR.read_text())
    if not recent["complete"] or recent["case_count"] != 24 or not prior["complete"]:
        raise ValueError("Need complete V32 paired baselines")
    cases = {row["seed"]: row for p in (RECENT_PLAN, PRIOR_PLAN)
             for row in json.loads(p.read_text())["cases"]}
    prior_rows = {row["seed"]: row for matrix in (recent, prior) for row in matrix["cases"]}
    plan = {
        "schema": "robot-learning-lab.cove50-v35-wider-rgb-screen-plan.v1",
        "role": "opened_observable_template_threshold_diagnostic_not_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**{name: sha(path) for name, path in v16.SOURCES.items()},
                          "evaluator": sha(EVALUATOR), "selector": sha(SELECTOR),
                          "template": sha(TEMPLATE)},
        "runner_sha256": sha(Path(__file__)),
        "recent_plan_sha256": sha(RECENT_PLAN), "recent_matrix_sha256": sha(RECENT),
        "prior_plan_sha256": sha(PRIOR_PLAN), "prior_matrix_sha256": sha(PRIOR),
        "case_count": len(TARGETS),
        "cases": [{"seed": seed, "role": "failed_target" if seed == 10538 else "pass_control",
                   "baseline_contact_s": prior_rows[seed]["contact_s"],
                   "baseline_pass": prior_rows[seed]["pass"],
                   "baseline_result": prior_rows[seed]["result"],
                   "baseline_trace": prior_rows[seed]["trace"],
                   **{key: cases[seed][key] for key in
                      ("rail_v", "delay_ms", "toy_offset_x_m", "toy_offset_y_m", "gaze_head_yaw_rad")}}
                  for seed in TARGETS],
        "rule": "Use rendered template when observed orange pixels <64 and image-derived correction norm >=3 mm; otherwise retain calibrated RGB. No simulator toy truth enters inference.",
        "qualification": "Five opened paired diagnostics test whether widening V32's visible-template branch rescues 10538 while retaining four prior passes. Full 24-start and 51-start retention and disjoint fresh scoring remain separate gates.",
    }
    if not (not prior_rows[10538]["pass"] and all(prior_rows[s]["pass"] for s in TARGETS[1:])):
        raise ValueError("Expected one prior failure and four passing controls")
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists():
        if PLAN.read_bytes() != encoded:
            raise ValueError("Frozen V35 plan differs from current sources")
    else:
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
               "--reach-checkpoint", str(sources["actor"]),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(sources["reference"]),
               "--geometry", str(sources["geometry"]),
               "--entry", "continuous", "--seed", str(seed),
               "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
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
            raise RuntimeError(f"V35 native evaluator failed for seed {seed}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if (report["evaluator_sha256"] != hashes["evaluator"] or
            report["selector_source_sha256"] != hashes["selector"] or
            report["template_source_sha256"] != hashes["template"] or
            report["body_sha256"] != hashes["body"] or
            report["selected_pointer_sha256"] != hashes["pointer"] or
            report["adaptive_variant"] != EXPECTED_VARIANT):
        raise ValueError(f"V35 source identity mismatch: {result}")
    faults = gate(report)
    entry = next((e for e in report["image_goal_capture_events"] if e["time_s"] == 7.5), None)
    if not entry or not entry["used_for_control"] or not entry.get("template_applied"):
        faults.append("image template unavailable at entry")
    if entry:
        correction = np.asarray(entry["template_correction_xy_mm"])
        expected = entry["orange_pixels"] < 64 and np.linalg.norm(correction) >= 3
        if entry.get("template_used_for_control") != bool(expected):
            faults.append("observable template rule mismatch")
        if entry["xy_error_truth_mm_scoring_only"] > 10:
            faults.append("RGB goal error above 10 mm")
    if report["image_goal_missing_at_s"] is not None:
        faults.append("image goal missing during reach")
    with np.load(trace) as new, np.load(ROOT / case["baseline_trace"]) as old:
        qpos_delta = float(np.max(np.abs(new["qpos"][:375] - old["qpos"][:375])))
        boundary_delta = float(np.max(np.abs(new["qpos"][374] - old["qpos"][374])))
        action_delta = float(np.max(np.abs(new["action"][:375] - old["action"][:375])))
    if qpos_delta > .002 or boundary_delta > 5e-5 or action_delta > .001:
        faults.append("pre-reach native action/state drift")
    return {"seed": seed, "role": case["role"], "baseline_pass": case["baseline_pass"],
            "baseline_contact_s": case["baseline_contact_s"],
            "pass": not faults,
            "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
            "entry_orange_pixels": None if entry is None else entry["orange_pixels"],
            "template_used_for_control": None if entry is None else entry["template_used_for_control"],
            "entry_goal_error_mm_scoring_only": None if entry is None else entry["xy_error_truth_mm_scoring_only"],
            "paired_pre_reach_qpos_max_abs": qpos_delta,
            "paired_boundary_qpos_max_abs": boundary_delta,
            "paired_pre_reach_action_max_abs": action_delta,
            "faults": faults,
            "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["seed"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v35-wider-rgb-screen.v1",
                      "role": "opened_observable_template_threshold_diagnostic_not_fresh",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "pass", "contact_s", "entry_goal_error_mm_scoring_only", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
