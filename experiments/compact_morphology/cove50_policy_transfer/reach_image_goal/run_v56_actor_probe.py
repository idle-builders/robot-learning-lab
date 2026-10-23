"""Frozen saved-arm actor diagnostic on V51 new motion failures."""

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
    run_v51_positive_sparse_neighbors as v51,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v56"
PLAN = OUT / "opened-actor-probe-plan.json"
DEST = OUT / "opened-actor-probe"
EVALUATOR = Path(__file__).with_name("evaluate_actor_probe_v56.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v47.py")
FAILURE_IDS = ("positive-07", "positive-08", "positive-10")
ACTORS = ("v10", "v15")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    previous_path = v51.DEST / "matrix.json"
    previous = json.loads(previous_path.read_text())
    failed = {r["id"]: r for r in previous["cases"] if not r["pass"]}
    if (not previous["complete"] or previous["case_count"] != 12 or
            previous["pass_count"] != 9 or set(failed) != set(FAILURE_IDS) or
            [failed[i]["contact_s"] for i in FAILURE_IDS] != [.94, .86, .28]):
        raise ValueError("Expected complete V51 three-failure actor screen")
    original = {c["id"]: c for c in json.loads(v51.PLAN.read_text())["cases"]}
    cases = [{**original[ident], "actor_override": actor,
              "baseline_v51_contact_s": failed[ident]["contact_s"]}
             for ident in FAILURE_IDS for actor in ACTORS]
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(screen.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE),
                   alternate_actor=sha(v16.SOURCES["alternate_actor"]),
                   specialist_actor=sha(v16.SOURCES["specialist_actor"]))
    plan = {
        "schema": "robot-learning-lab.cove50-v56-saved-actor-probe-plan.v1",
        "role": "opened_saved_actor_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "v51_neighbor_plan_sha256": sha(v51.PLAN),
        "v51_neighbor_matrix_sha256": sha(previous_path),
        "failed_report_sha256": {ident: sha(ROOT / failed[ident]["result"])
                                 for ident in FAILURE_IDS},
        "case_count": len(cases), "cases": cases,
        "design": "Replay three opened V51 motion failures with the same native dynamics, RGB goal, toy, voltage, delay, gaze and authored route. Override only the selected four-arm residual with saved V10 or V15 actors that satisfy the same 11-feature contract. The V41 actor baseline is the V51 matrix. These paired visits look for motor-feasible teacher actions; seed or simulator truth is not a runtime selector.",
        "gate": "Each native visit is checked for support, rated effort, >=1 s continuous contact, floor, clipping and RGB <=10 mm. An alternate actor pass is a teacher diagnostic only; retention and an observable automatic router are required before promotion.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V56 plan differs from sources")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    ident, actor = case["id"], case["actor_override"]
    result, trace, log = (DEST / f"seed-{case['seed']}-{ident}-{actor}.{suffix}"
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
               "--actor-override", actor,
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
               "--max-goal-age-s", "9", "--out", str(result), "--trace", str(trace)]
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
            raise RuntimeError(f"V56 evaluator failed for {ident}/{actor}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    selected_hash = hashes["alternate_actor" if actor == "v10" else "specialist_actor"]
    expected = {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "selected_reach_checkpoint_sha256": selected_hash,
        "adaptive_variant": "v56_opened_saved_actor_override_diagnostic",
        "actor_override": actor,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V56 source or actor mismatch: {result}")
    faults = gate(report)
    entry = next((e for e in report["image_goal_capture_events"]
                  if e.get("time_s") == report["reach_start_s"]), None)
    if not entry or entry.get("orange_pixels", 0) < 10 or not entry.get("used_for_control"):
        faults.append("RGB entry goal missing")
    error = None if entry is None else entry.get("xy_error_truth_mm_scoring_only")
    if error is None or error > 10:
        faults.append("RGB goal error above 10 mm or unavailable")
    return {
        "id": ident, "seed": case["seed"], "actor_override": actor,
        "baseline_v51_contact_s": case["baseline_v51_contact_s"],
        "selection_mode": report["selection_mode"],
        "pass": not faults,
        "contact_s": report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0),
        "entry_orange_pixels": None if entry is None else entry.get("orange_pixels"),
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
        futures = {pool.submit(run_case, case, plan): (case["id"], case["actor_override"])
                   for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[(row["id"], row["actor_override"])] = row
            ordered = [rows[(case["id"], case["actor_override"])] for case in plan["cases"]
                       if (case["id"], case["actor_override"]) in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v56-saved-actor-probe.v1",
                "role": "opened_saved_actor_diagnostic_not_fresh_score",
                "body_role": "rejected_historical_direct_frame_sandbox",
                "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "actor_override", "pass", "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
