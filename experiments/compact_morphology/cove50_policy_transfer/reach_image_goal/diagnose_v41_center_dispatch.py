"""Check whether V41 retains the V70 authored centers alongside V83 varied routes."""

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
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
    run_v41_opened_screen as v41,
    run_v70_center_route_screen as v70,
)
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
DEST = BASE / "v83/v41-authored-center-diagnostic"
PLAN = DEST / "plan.json"
SUPERSEDED = DEST / "invalid-abort-scorer-plan.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    v41.configure()
    older_path = BASE / "v41/opened-gate-plan.json"
    older_matrix_path = BASE / "v41/opened-native/matrix.json"
    older = json.loads(older_path.read_text())
    older_matrix = json.loads(older_matrix_path.read_text())
    v70_plan_path = v70.PLAN
    v70_matrix_path = v70.DEST / "matrix.json"
    v70_plan = json.loads(v70_plan_path.read_text())
    v70_matrix = json.loads(v70_matrix_path.read_text())
    if (older_matrix["pass_count"] != 75 or not older_matrix["complete"] or
            v70_matrix["pass_count"] != 35 or not v70_matrix["complete"]):
        raise ValueError("Need complete passing V41 and V70 source matrices")
    old_case = next(row for row in older["cases"] if row["seed"] == 10534)
    old_row = next(row for row in older_matrix["cases"] if row["seed"] == 10534)
    baseline = {row["id"]: row for row in v70_matrix["cases"]}
    centers = [{**case, "baseline_v83_contact_s": baseline[case["id"]]["contact_s"]}
               for case in v70_plan["cases"] if case["route_variant"] == "center"]
    if len(centers) != 7 or not all(baseline[x["id"]]["pass"] for x in centers):
        raise ValueError("Expected seven passing V70 center routes")
    cases = [{**old_case, "id": "prior-10534", "route_variant": "center",
              "baseline_v41_contact_s": old_row["contact_s"]}, *centers]
    sources = older["source_sha256"]
    if any(sources[key] != sha(path) for key, path in {
            "actor": v41.ACTOR, "evaluator": v41.EVALUATOR,
            "selector": v41.SELECTOR, "body": v16.SOURCES["body"],
            "pointer": v16.SOURCES["pointer"]}.items()):
        raise ValueError("V41 replay source differs from frozen older gate")
    plan = {
        "schema": "robot-learning-lab.cove50-v83-v41-center-dispatch-diagnostic-plan.v1",
        "role": "post_pilot_commanded_route_dispatch_diagnostic_not_retention",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "v41_plan_sha256": sha(older_path),
        "v41_matrix_sha256": sha(older_matrix_path),
        "v70_plan_sha256": sha(v70_plan_path),
        "v70_matrix_sha256": sha(v70_matrix_path),
        "superseded_invalid_harness_plan_sha256": (sha(SUPERSEDED) if SUPERSEDED.exists()
                                                     else None),
        "case_count": len(cases), "cases": cases,
        "gate": "The V41 evaluator must freshly pass the old 10534 start and all seven formerly passing V70 authored-center routes under their unchanged seed, toy, motor and gaze inputs. Its image goal, actor, support, effort and contact gates remain active. This is a post-pilot diagnostic for a commanded-route router; V83 already passed the 56 off-author routes. It does not repair the frozen V83 older pilot or count as a new disjoint score.",
    }
    DEST.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V41 center diagnostic differs from source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    ident, seed = case["id"], case["seed"]
    result, trace, log = (DEST / f"seed-{seed}-{ident}.{suffix}"
                          for suffix in ("json", "npz", "log"))
    sources = v16.SOURCES
    command = [sys.executable, str(v41.EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(v41.ACTOR),
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
            raise RuntimeError(f"V41 center replay failed for {ident}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if any(report.get(key) != value for key, value in {
            "evaluator_sha256": hashes["evaluator"],
            "selector_source_sha256": hashes["selector"],
            "template_source_sha256": hashes["template"],
            "body_sha256": hashes["body"],
            "selected_pointer_sha256": hashes["pointer"],
            "reach_checkpoint_sha256": hashes["actor"],
            "adaptive_variant": v41.VARIANT,
            "pitch_offset_rad": .105,
            "roll_clearance_rad": -.25,
    }.items()):
        raise ValueError(f"V41 authored-center source/route identity mismatch: {result}")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event["time_s"] == 7.5), None)
    if not entry or not entry.get("used_for_control") or not entry.get("template_applied"):
        faults.append("image template unavailable at reach entry")
    if entry and entry.get("template_used_for_control") != v41.expected_template(entry):
        faults.append("V41 observable template rule mismatch")
    if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                  entry["xy_error_truth_mm_scoring_only"] > 10):
        faults.append("RGB goal error above 10 mm")
    if report["image_goal_missing_at_s"] is not None:
        faults.append("image goal missing during reach")
    return {
        "id": ident, "seed": seed,
        "baseline_contact_s": case.get("baseline_v83_contact_s",
                                       case.get("baseline_v41_contact_s")),
        "selection_mode": report["selection_mode"],
        "entry_orange_pixels": None if entry is None else entry["orange_pixels"],
        "entry_goal_error_mm_scoring_only": (None if entry is None else
                                             entry.get("xy_error_truth_mm_scoring_only")),
        "contact_s": report["phases"].get("reach", {}).get(
            "continuous_finger_contact_s", 0),
        "pass": not faults, "faults": faults,
        "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(plan["cases"])
        pending = {}
        for _ in range(args.workers):
            case = next(iterator, None)
            if case is not None:
                pending[pool.submit(run_case, case, plan)] = case
        while pending:
            future = next(as_completed(tuple(pending)))
            case = pending.pop(future)
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in plan["cases"] if item["id"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v83-v41-center-dispatch-diagnostic.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN),
                "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "gate_met": (len(ordered) == len(plan["cases"]) and
                             all(item["pass"] for item in ordered)),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "faults")}), flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(run_case, next_case, plan)] = next_case


if __name__ == "__main__":
    main()
