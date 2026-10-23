"""Opened native roll sensitivity probe for V41's intermittent 10810 contact."""

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
    run_v41_opened_screen as screen,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v44/opened-contact-roll"
PLAN = OUT / "plan.json"
EVALUATOR = Path(__file__).with_name("evaluate_selective_goal_v41.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v41.py")
VISITS = [{"seed": 10810, "terminal_roll_rad": value}
          for value in (.02, .04, .06)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    matrix = json.loads((fresh.DEST / "matrix.json").read_text())
    original = next(row for row in matrix["cases"] if row["seed"] == 10810)
    if not matrix["complete"] or matrix["case_count"] != 32 or original["pass"]:
        raise ValueError("Expected completed V41 fresh score and failed 10810")
    starts = {row["seed"]: row for row in json.loads(fresh.PLAN.read_text())["cases"]}
    plan = {
        "schema": "robot-learning-lab.cove50-v44-contact-roll-probe.v1",
        "role": "postscore_opened_diagnostic_not_promotion",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v41_fresh_matrix_sha256": sha(fresh.DEST / "matrix.json"),
        "v41_fresh_plan_sha256": sha(fresh.PLAN),
        "source_sha256": {**{key: sha(path) for key, path in v16.SOURCES.items()},
                          "actor": sha(screen.ACTOR), "evaluator": sha(EVALUATOR),
                          "selector": sha(SELECTOR), "runner": sha(Path(__file__))},
        "visits": [{**visit, "start": starts[visit["seed"]]} for visit in VISITS],
        "question": "Does a small positive authored terminal shoulder-roll target stabilize the intermittent 10810 fingertip contact? Baseline V41 is zero applied roll and 0.60 s contact. This diagnostic does not constitute training or a fresh score; any useful offset must retain other opened starts before candidacy.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen contact roll probe changed")
    PLAN.write_bytes(encoded)
    return plan


def run(visit: dict, plan: dict) -> dict:
    offset = visit["terminal_roll_rad"]
    label = f"seed-10810-roll-{round(offset * 1000):03d}mrad"
    report, trace, log = (OUT / f"{label}.{ext}" for ext in ("json", "npz", "log"))
    case = visit["start"]
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
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
               "--terminal-roll-offset-rad", str(offset), "--terminal-roll-mode", "fixed",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--out", str(report), "--trace", str(trace)]
    if not report.exists() or not trace.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
             "/root/microduck_rl/src", "/workspace/microduck_rl/src"]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        with log.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise RuntimeError(f"Native contact probe failed: {label} ({log})")
    data = json.loads(report.read_text())
    hashes = plan["source_sha256"]
    if any(data.get(key) != value for key, value in {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "reach_checkpoint_sha256": hashes["actor"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
    }.items()):
        raise ValueError(f"Source identity mismatch: {label}")
    entry = next(e for e in data["image_goal_capture_events"] if e["time_s"] == 7.5)
    faults = gate(data)
    if not entry["used_for_control"] or entry["xy_error_truth_mm_scoring_only"] > 10:
        faults.append("RGB goal acceptance or scored error")
    return {"seed": 10810, "terminal_roll_rad": offset, "pass": not faults,
            "contact_s": data["phases"]["reach"]["continuous_finger_contact_s"],
            "entry_orange_pixels": entry["orange_pixels"],
            "entry_goal_error_mm_scoring_only": entry["xy_error_truth_mm_scoring_only"],
            "faults": faults, "result": str(report.relative_to(ROOT)),
            "trace": str(trace.relative_to(ROOT)), "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= len(VISITS):
        raise ValueError("Invalid native worker count")
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, visit, plan): visit["terminal_roll_rad"]
                   for visit in plan["visits"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["terminal_roll_rad"]] = row
            ordered = [rows[visit["terminal_roll_rad"]] for visit in plan["visits"]
                       if visit["terminal_roll_rad"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v44-contact-roll-probe-matrix.v1",
                      "role": "postscore_opened_diagnostic_not_promotion",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(VISITS),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "cases": ordered}
            (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
