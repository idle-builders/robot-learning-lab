"""Opened native 10810 contact probe with fractional sparse RGB correction."""

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
OUT = BASE / "v44b/opened-sparse-probe"
PLAN = OUT / "plan.json"
EVALUATOR = Path(__file__).with_name("evaluate_sparse_template_blend_v44b.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v44b.py")
BLENDS = (0.0, 0.5)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    score = json.loads((fresh.DEST / "matrix.json").read_text())
    failed = next(row for row in score["cases"] if row["seed"] == 10810)
    if not score["complete"] or score["case_count"] != 32 or failed["pass"]:
        raise ValueError("Need failed completed V41 disjoint score")
    start = next(row for row in json.loads(fresh.PLAN.read_text())["cases"]
                 if row["seed"] == 10810)
    plan = {
        "schema": "robot-learning-lab.cove50-v44b-sparse-template-probe.v1",
        "role": "postscore_opened_diagnostic_not_promotion",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v41_fresh_matrix_sha256": sha(fresh.DEST / "matrix.json"),
        "v41_fresh_plan_sha256": sha(fresh.PLAN),
        "source_sha256": {**{key: sha(path) for key, path in v16.SOURCES.items()},
                          "actor": sha(screen.ACTOR), "evaluator": sha(EVALUATOR),
                          "selector": sha(SELECTOR), "v41_selector": sha(SELECTOR.with_name("selective_template_rgb_goal_v41.py")),
                          "runner": sha(Path(__file__))},
        "visits": [{"seed": 10810, "sparse_template_blend": blend, "start": start}
                   for blend in BLENDS],
        "question": "For the 57-pixel 10810 RGB view, does applying zero or half of the V41 image-template correction stabilize fingertip contact? V41's full correction yielded 0.60 s. The route, saved actor, body, motor law, gaze, timing and native start are unchanged. This is postscore opened diagnosis; any successful blend requires passing controls and a future disjoint score.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen sparse probe changed")
    PLAN.write_bytes(encoded)
    return plan


def run(visit: dict, plan: dict) -> dict:
    blend = visit["sparse_template_blend"]
    label = f"seed-10810-blend-{int(blend * 100):03d}pct"
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
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--sparse-template-blend", str(blend),
               "--out", str(report), "--trace", str(trace)]
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
            raise RuntimeError(f"Native sparse probe failed: {label} ({log})")
    data = json.loads(report.read_text())
    hashes = plan["source_sha256"]
    if any(data.get(key) != value for key, value in {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "reach_checkpoint_sha256": hashes["actor"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "adaptive_variant": "v44b_sparse_template_blend_diagnostic",
        "sparse_template_blend": blend,
    }.items()):
        raise ValueError(f"Source identity mismatch: {label}")
    entry = next(e for e in data["image_goal_capture_events"] if e["time_s"] == 7.5)
    faults = gate(data)
    if not entry["used_for_control"] or entry["xy_error_truth_mm_scoring_only"] > 10:
        faults.append("RGB goal acceptance or scored error")
    return {"seed": 10810, "sparse_template_blend": blend, "pass": not faults,
            "contact_s": data["phases"]["reach"]["continuous_finger_contact_s"],
            "selection_mode": data["selection_mode"],
            "entry_orange_pixels": entry["orange_pixels"],
            "entry_goal_error_mm_scoring_only": entry["xy_error_truth_mm_scoring_only"],
            "faults": faults, "result": str(report.relative_to(ROOT)),
            "trace": str(trace.relative_to(ROOT)), "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= len(BLENDS):
        raise ValueError("Invalid native worker count")
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, visit, plan): visit["sparse_template_blend"]
                   for visit in plan["visits"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["sparse_template_blend"]] = row
            ordered = [rows[b] for b in BLENDS if b in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v44b-sparse-probe-matrix.v1",
                      "role": "postscore_opened_diagnostic_not_promotion",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(BLENDS),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "cases": ordered}
            (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
