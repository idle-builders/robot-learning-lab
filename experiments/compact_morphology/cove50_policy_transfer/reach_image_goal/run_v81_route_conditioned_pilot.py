"""Native pilot for the route-conditioned arm against opened contact failures."""

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
    run_v70_center_route_screen as v70,
    run_v73_route_holdout as v73,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v81"
PLAN = OUT / "route-conditioned-native-pilot-plan.json"
DEST = OUT / "native-pilot"
ACTOR = OUT / "actor.pt"
TRAINING_PLAN = OUT / "route-conditioned-arm-training-plan.json"
EVALUATOR = Path(__file__).with_name("evaluate_route_conditioned_arm_v81.py")
SELECTOR = v70.SELECTOR
TEACHER_IDS = ("positive-01-pitch-low", "positive-01-roll-high",
               "positive-07-pitch-high", "edge-11-roll-low")
CONTROL_IDS = (("v70", "positive-10-center"),
               ("v70", "positive-10-roll-high"),
               ("v70", "edge-12-roll-high"),
               ("v70", "large-20-center"),
               ("v73", "large-23-roll-high"),
               ("v73", "edge-10-pitch-low"),
               ("v73", "positive-11-pitch-low"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    training = json.loads((OUT / "training-report.json").read_text())
    if training["actor_sha256"] != sha(ACTOR) or training["training_plan_sha256"] != sha(TRAINING_PLAN):
        raise ValueError("V81 actor and training plan do not match")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    matrices = {"v70": json.loads(v70.DEST.joinpath("matrix.json").read_text()),
                "v73": json.loads(v73.DEST.joinpath("matrix.json").read_text())}
    plans = {"v70": json.loads(v70.PLAN.read_text()),
             "v73": json.loads(v73.PLAN.read_text())}
    if not all(m["complete"] for m in matrices.values()):
        raise ValueError("Need complete opened V70 and V73 source matrices")
    cases = []
    for cohort, ident, role in ([ ("v73", ident, "teacher_recovery") for ident in TEACHER_IDS ] +
                                [ (cohort, ident, "retention_control") for cohort, ident in CONTROL_IDS ]):
        case = next(case for case in plans[cohort]["cases"] if case["id"] == ident)
        row = next(row for row in matrices[cohort]["cases"] if row["id"] == ident)
        report = ROOT / row["result"]
        cases.append({**case, "role": role, "cohort": cohort,
                      "baseline_contact_s": row["contact_s"],
                      "baseline_pass": row["pass"],
                      "baseline_report_sha256": sha(report)})
    if len(cases) != 11 or cases[0]["baseline_pass"]:
        raise ValueError("Expected one opened V73 miss, three thin teacher routes and seven passing controls")
    plan = {
        "schema": "robot-learning-lab.cove50-v81-native-pilot-plan.v1",
        "role": "opened_route_conditioned_actor_pilot_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources, "runner_sha256": sha(Path(__file__)),
        "training_plan_sha256": sha(TRAINING_PLAN),
        "training_report_sha256": sha(OUT / "training-report.json"),
        "source_matrix_sha256": {cohort: sha(path) for cohort, path in
                                 (("v70", v70.DEST / "matrix.json"),
                                  ("v73", v73.DEST / "matrix.json"))},
        "case_count": len(cases), "teacher_case_count": len(TEACHER_IDS),
        "cases": cases,
        "gate": "The first four source-opened teacher routes must each pass the complete native safety, support, effort, RGB and >=1 s contact gate and hold >=1.2 s. Continue to seven controls only after that stage. All eleven must then pass; each control must hold >=1.2 s and lose no more than 0.3 s from its matched V70/V73 contact. A pilot is not a full retention or fresh score.",
        "design": "Use a newly trained 13-feature V81 arm with commanded pitch/roll route deltas as observed inputs. Keep V70's automatic image-goal rules, native body, timing, toy, motor and gaze of each source visit. No prescribed offset is applied at inference; seed and simulator toy truth do not select a route or bias.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V81 pilot plan differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    stem = DEST / f"seed-{case['seed']}-{case['id']}"
    result, trace, log = (stem.with_suffix(f".{suffix}")
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
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
             "/root/microduck_rl/src", "/workspace/microduck_rl/src"]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        with log.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise RuntimeError(f"V81 native pilot failed for {case['id']}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    expected = {"evaluator_sha256": hashes["evaluator"],
                "selector_source_sha256": hashes["selector"],
                "template_source_sha256": hashes["template"],
                "body_sha256": hashes["body"],
                "selected_pointer_sha256": hashes["pointer"],
                "reach_checkpoint_sha256": hashes["actor"],
                "selected_reach_checkpoint_sha256": hashes["actor"],
                "adaptive_variant": "v81_route_conditioned_arm_and_v70_observed_goal_candidate"}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V81 report source or actor mismatch: {result}")
    if (report["pitch_offset_rad"] != case["pitch_offset_rad"] or
            report["roll_clearance_rad"] != case["roll_clearance_rad"]):
        raise ValueError(f"V81 commanded route mismatch: {result}")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event.get("time_s") == report["reach_start_s"]), None)
    error = None if entry is None else entry.get("xy_error_truth_mm_scoring_only")
    if error is None or error > 10:
        faults.append("RGB goal error above 10 mm or unavailable")
    if report.get("image_goal_missing_at_s") is not None:
        faults.append("image goal missing during reach")
    contact = report["phases"]["reach"]["continuous_finger_contact_s"]
    margin = (contact >= 1.2 if case["role"] == "teacher_recovery" else
              contact >= max(1.2, case["baseline_contact_s"] - .3))
    return {"id": case["id"], "seed": case["seed"], "role": case["role"],
            "cohort": case["cohort"], "baseline_contact_s": case["baseline_contact_s"],
            "pass": not faults, "contact_s": contact, "margin_met": margin,
            "selection_mode": report["selection_mode"],
            "entry_goal_error_mm_scoring_only": error, "faults": faults,
            "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--stage", choices=("teachers", "all"), default="teachers")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    if args.stage == "all":
        prior_path = DEST / "teacher-stage-matrix.json"
        prior = json.loads(prior_path.read_text())
        if not prior["teacher_stage_gate_met"] or prior["plan_sha256"] != sha(PLAN):
            raise ValueError("Teacher stage did not meet the frozen V81 native gate")
    cases = plan["cases"][:4] if args.stage == "teachers" else plan["cases"]
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["id"] for case in cases}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in cases if case["id"] in rows]
            teacher_gate = len(ordered) >= 4 and all(r["pass"] and r["margin_met"]
                                                   for r in ordered[:4])
            matrix = {
                "schema": "robot-learning-lab.cove50-v81-route-conditioned-native-pilot.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "stage": args.stage,
                "complete": len(ordered) == len(cases),
                "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                "teacher_stage_gate_met": teacher_gate,
                "pilot_gate_met": len(ordered) == len(plan["cases"]) and
                                  all(r["pass"] and r["margin_met"] for r in ordered),
                "cases": ordered,
            }
            name = "teacher-stage-matrix.json" if args.stage == "teachers" else "matrix.json"
            (DEST / name).write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "role", "pass", "contact_s", "margin_met", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
