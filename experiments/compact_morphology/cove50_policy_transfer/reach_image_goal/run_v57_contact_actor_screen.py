"""Native opened positive-sparse neighborhood screen for the V57 distilled arm."""

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
    run_v51_positive_sparse_neighbors as v51,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v57"
PLAN = OUT / "contact-actor-screen-plan.json"
DEST = OUT / "positive-sparse-neighbors"
ACTOR = OUT / "actor.pt"
TRAIN_PLAN = OUT / "contact-actor-training-plan.json"
TRAIN_REPORT = OUT / "training-report.json"
EVALUATOR = v51.EVALUATOR
SELECTOR = v51.SELECTOR


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    baseline_path = v51.DEST / "matrix.json"
    baseline = json.loads(baseline_path.read_text())
    training = json.loads(TRAIN_REPORT.read_text())
    if (not baseline["complete"] or baseline["case_count"] != 12 or
            baseline["pass_count"] != 9 or not training["actor_sha256"] == sha(ACTOR) or
            training["training_plan_sha256"] != sha(TRAIN_PLAN)):
        raise ValueError("Need V51 baseline and source-frozen V57 actor")
    cases = json.loads(v51.PLAN.read_text())["cases"]
    prior = {row["id"]: row for row in baseline["cases"]}
    if len(cases) != 12 or {c["id"] for c in cases} != set(prior):
        raise ValueError("V51 neighbor cases changed")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(ACTOR), evaluator=sha(EVALUATOR), selector=sha(SELECTOR),
                   template=sha(v51.screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v57-contact-arm-screen-plan.v1",
        "role": "opened_native_actor_candidate_screen_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "training_plan_sha256": sha(TRAIN_PLAN),
        "training_report_sha256": sha(TRAIN_REPORT),
        "v51_neighbor_plan_sha256": sha(v51.PLAN),
        "v51_neighbor_matrix_sha256": sha(baseline_path),
        "case_count": len(cases), "cases": cases,
        "design": "Replay all twelve V51 positive-sparse native neighbors with only the learned arm checkpoint replaced by V57. Toy, motor, gaze, route, camera selector, timing and body are unchanged. The three V51 failures were opened teacher training states; nine passing variations test immediate retention. Seed and simulator toy truth are not actor inputs.",
        "gate": "All twelve visits must pass native support, rated effort, >=1 s continuous fingertip contact, floor, clipping and RGB <=10 mm. Require at least three actual observed positive sparse-rule activations and no lost visible entry. This is opened development, not a fresh score or selected-body qualification."
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V57 screen plan differs from source evidence")
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
               "--reach-checkpoint", str(ACTOR),
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
            raise RuntimeError(f"V57 evaluator failed for {ident}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    expected = {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "selected_reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v47_large_sparse_positive_camera_abstention_candidate",
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V57 source or selected actor mismatch: {result}")
    faults = gate(report)
    entry = next((e for e in report["image_goal_capture_events"]
                  if e.get("time_s") == report["reach_start_s"]), None)
    if report["reacquisition_probe_orange_pixels"] == 0 and not (
            report["reacquisition_trigger_s"] is not None and
            entry and entry.get("orange_pixels", 0) >= 10):
        faults.append("physical zero-pixel scan did not reacquire toy")
    if not entry or entry.get("orange_pixels", 0) < 10 or not entry.get("used_for_control"):
        faults.append("reach entry did not use a visible RGB toy")
    error = None if not entry else entry.get("xy_error_truth_mm_scoring_only")
    if error is None or error > 10:
        faults.append("RGB goal error above 10 mm or unavailable")
    baseline = next(row for row in json.loads((v51.DEST / "matrix.json").read_text())["cases"]
                    if row["id"] == ident)
    return {
        "id": ident, "seed": case["seed"],
        "development_role": case["development_role"],
        "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
        "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
        "baseline_v51_contact_s": baseline["contact_s"],
        "positive_camera_large_sparse_abstention": report["positive_camera_large_sparse_abstention_entry"],
        "pass": not faults,
        "contact_s": report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0),
        "entry_orange_pixels": None if not entry else entry.get("orange_pixels"),
        "entry_goal_error_mm_scoring_only": error,
        "faults": faults,
        "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
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
        futures = {pool.submit(run_case, case, plan): case["id"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v57-contact-arm-screen.v1",
                "role": "opened_native_actor_candidate_screen_not_fresh_score",
                "body_role": "rejected_historical_direct_frame_sandbox",
                "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                "positive_sparse_use_count": sum(r["positive_camera_large_sparse_abstention"]
                                                 for r in ordered),
                "screen_gate_met": (len(ordered) == len(plan["cases"]) and
                                   all(r["pass"] for r in ordered) and
                                   sum(r["positive_camera_large_sparse_abstention"]
                                       for r in ordered) >= 3),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
