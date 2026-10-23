"""Opened V41 fresh short-contact actor ablation with matched controls."""

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
OUT = BASE / "v42/postscore-short-contact-actor"
PLAN = OUT / "plan.json"
V41_SCORE = BASE / "v41/fresh-reserved/matrix.json"
V41_PLAN = BASE / "v41/fresh-challenge-plan.json"
EVALUATOR = Path(__file__).with_name("evaluate_selective_goal_v41.py")
ACTORS = {"v13": BASE / "v13/actor.pt", "v36": BASE / "v36/actor.pt"}
VISITS = [
    {"id": f"{seed}-{actor}", "seed": seed, "actor": actor,
     "question": ("Does this saved actor stabilize 10810's observed RGB goal?"
                  if seed == 10810 else "Does that change retain a nearby V41 fresh pass?")}
    for seed in (10810, 10813, 10816) for actor in ("v13", "v36")
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    score = json.loads(V41_SCORE.read_text())
    if (not score["complete"] or score["case_count"] != 32 or
            next(row for row in score["cases"] if row["seed"] == 10810)["pass"]):
        raise ValueError("Need completed failed V41 disjoint score")
    cases = {c["seed"]: c for c in json.loads(V41_PLAN.read_text())["cases"]}
    plan = {"schema": "robot-learning-lab.cove50-v42-short-contact-actor-plan.v1",
            "role": "opened_postscore_diagnostic_not_promotion",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "v41_fresh_score_sha256": sha(V41_SCORE),
            "source_sha256": {**{name: sha(path) for name, path in v16.SOURCES.items()},
                              "evaluator": sha(EVALUATOR),
                              "selector": sha(Path(__file__).with_name("selective_template_rgb_goal_v41.py")),
                              **{f"actor_{name}": sha(path) for name, path in ACTORS.items()},
                              "runner": sha(Path(__file__))},
            "visits": [{**visit, "start": cases[visit["seed"]]} for visit in VISITS],
            "question": "On the 10810 short-contact miss, compare prior saved V13 and V36 arms under the exact V41 RGB selector and route against V41's 0.60 s baseline. Replay both actors at two nearby passing high-Y starts to expose regressions. These are postscore opened diagnostics, not new fresh passes."}
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.is_file() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen short-contact actor plan changed")
    PLAN.write_bytes(encoded)
    return plan


def run(visit: dict, plan: dict) -> dict:
    ident = visit["id"]
    case = visit["start"]
    report = OUT / f"{ident}.json"
    trace = OUT / f"{ident}.npz"
    log = OUT / f"{ident}.log"
    command = [sys.executable, str(EVALUATOR),
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
    if (data["evaluator_sha256"] != plan["source_sha256"]["evaluator"] or
            data["selector_source_sha256"] != plan["source_sha256"]["selector"] or
            data["reach_checkpoint_sha256"] != plan["source_sha256"][f"actor_{visit['actor']}"] or
            data["body_sha256"] != plan["source_sha256"]["body"] or
            data["selected_pointer_sha256"] != plan["source_sha256"]["pointer"]):
        raise ValueError(f"Source identity mismatch: {ident}")
    entry = next(e for e in data["image_goal_capture_events"] if e["time_s"] == 7.5)
    faults = gate(data)
    if not entry["used_for_control"] or entry["xy_error_truth_mm_scoring_only"] > 10:
        faults.append("RGB goal acceptance or scoring error")
    return {"id": ident, "seed": case["seed"], "evaluator": "v41",
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
        raise ValueError("Use one to six evaluators")
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, visit, plan): visit["id"] for visit in plan["visits"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[visit["id"]] for visit in plan["visits"] if visit["id"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v42-short-contact-actor-diagnostic.v1",
                      "role": "opened_postscore_diagnostic_not_promotion",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(VISITS),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "cases": ordered}
            (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
