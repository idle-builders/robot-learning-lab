"""Opened postscore actor/image ablation for the stopped V40 retention gate."""

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
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import run_v16_opened_matrix as v16

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v40/postscore-regression-diagnostic"
PLAN = OUT / "plan.json"
V40_GATE = BASE / "v40/opened-native/gate-stop.json"
V40_PLAN = BASE / "v40/opened-gate-plan.json"
EVALUATORS = {version: Path(__file__).with_name(f"evaluate_selective_goal_{version}.py")
              for version in ("v38", "v40")}
ACTORS = {"v13": BASE / "v13/actor.pt", "v36": BASE / "v36/actor.pt"}
VISITS = [
    {"id": "10545-v40-v13", "seed": 10545, "evaluator": "v40", "actor": "v13",
     "question": "Does the earlier actor retain the V40 calibrated goal at the first regression?"},
    {"id": "10546-v40-v13", "seed": 10546, "evaluator": "v40", "actor": "v13",
     "question": "Does the earlier actor tolerate the V40 sparse image correction?"},
    {"id": "10546-v38-v36", "seed": 10546, "evaluator": "v38", "actor": "v36",
     "question": "Does the new actor retain the calibrated RGB goal without the sparse correction?"},
    {"id": "10546-v38-v13", "seed": 10546, "evaluator": "v38", "actor": "v13",
     "question": "Does the earlier actor and calibrated RGB goal reproduce baseline contact?"},
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    stop = json.loads(V40_GATE.read_text())
    if stop["regressions"] != [10545, 10546] or stop["fresh_challenge_state"] != "sealed_not_run":
        raise ValueError("Expected the stopped V40 gate before postscore diagnosis")
    cases = {c["seed"]: c for c in json.loads(V40_PLAN.read_text())["cases"]}
    plan = {"schema": "robot-learning-lab.cove50-v40-postscore-diagnostic-plan.v1",
            "role": "opened_postscore_diagnostic_not_promotion",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "gate_stop_sha256": sha(V40_GATE),
            "source_sha256": {**{name: sha(path) for name, path in v16.SOURCES.items()},
                              **{f"evaluator_{name}": sha(path) for name, path in EVALUATORS.items()},
                              **{f"actor_{name}": sha(path) for name, path in ACTORS.items()},
                              "runner": sha(Path(__file__))},
            "visits": [{**visit, "start": cases[visit["seed"]]} for visit in VISITS],
            "question": "Factor actor checkpoint (V13/V36) against calibrated versus sparse-template RGB goal on the two newly opened prior-pass regressions. No fresh start is opened."}
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.is_file() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen postscore plan changed")
    PLAN.write_bytes(encoded)
    return plan


def run(visit: dict, plan: dict) -> dict:
    ident = visit["id"]
    case = visit["start"]
    report = OUT / f"{ident}.json"
    trace = OUT / f"{ident}.npz"
    log = OUT / f"{ident}.log"
    command = [sys.executable, str(EVALUATORS[visit["evaluator"]]),
               "--body-xml", str(v16.SOURCES["body"]),
               "--calibration-model", str(v16.SOURCES["calibration"]),
               "--squat-checkpoint", str(v16.SOURCES["squat"]),
               "--reach-checkpoint", str(ACTORS[visit["actor"]]),
               "--alternate-reach-checkpoint", str(v16.SOURCES["alternate_actor"]),
               "--specialist-reach-checkpoint", str(v16.SOURCES["specialist_actor"]),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(v16.SOURCES["reference"]),
               "--geometry", str(v16.SOURCES["geometry"]),
               "--entry", "continuous", "--seed", str(case["seed"]),
               "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--out", str(report), "--trace", str(trace)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
         "/root/microduck_rl/src", "/workspace/microduck_rl/src"]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    with log.open("w") as stream:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, check=False).returncode
    if code:
        raise RuntimeError(f"Native evaluator failed: {ident} ({log})")
    data = json.loads(report.read_text())
    if (data["evaluator_sha256"] != plan["source_sha256"][f"evaluator_{visit['evaluator']}"] or
            data["reach_checkpoint_sha256"] != plan["source_sha256"][f"actor_{visit['actor']}"] or
            data["body_sha256"] != plan["source_sha256"]["body"] or
            data["selected_pointer_sha256"] != plan["source_sha256"]["pointer"]):
        raise ValueError(f"Source identity mismatch: {ident}")
    entry = next(e for e in data["image_goal_capture_events"] if e["time_s"] == 7.5)
    faults = gate(data)
    if not entry["used_for_control"] or entry["xy_error_truth_mm_scoring_only"] > 10:
        faults.append("RGB goal acceptance or scoring error")
    return {"id": ident, "seed": case["seed"], "evaluator": visit["evaluator"],
            "actor": visit["actor"], "pass": not faults,
            "contact_s": data["phases"]["reach"]["continuous_finger_contact_s"],
            "selection_mode": data["selection_mode"],
            "orange_pixels": entry["orange_pixels"],
            "template_used_for_control": entry.get("template_used_for_control", False),
            "template_correction_xy_mm": entry.get("template_correction_xy_mm"),
            "entry_goal_error_mm_scoring_only": entry["xy_error_truth_mm_scoring_only"],
            "faults": faults, "result": str(report.relative_to(ROOT)),
            "trace": str(trace.relative_to(ROOT)), "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= len(VISITS):
        raise ValueError("Use one to four evaluators")
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, visit, plan): visit["id"] for visit in plan["visits"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[visit["id"]] for visit in plan["visits"] if visit["id"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v40-postscore-diagnostic.v1",
                      "role": "opened_postscore_diagnostic_not_promotion",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(VISITS),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "cases": ordered}
            (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
