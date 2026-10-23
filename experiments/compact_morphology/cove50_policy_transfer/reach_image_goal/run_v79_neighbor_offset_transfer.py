"""Check whether the V78 prescribed offset direction transfers to other thin routes."""

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
    run_v78_positive_route_teacher as v78,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v79"
PLAN = OUT / "neighbor-offset-transfer-plan.json"
DEST = OUT / "native-neighbor-offsets"
V73_PLAN = BASE / "v73/route-holdout-run-plan.json"
V73_MATRIX = BASE / "v73/opened-route-holdout/matrix.json"
V78_MATRIX = v78.DEST / "matrix.json"
EVALUATOR = v78.EVALUATOR
SELECTOR = v78.SELECTOR
NEIGHBORS = ("positive-01-roll-high", "positive-07-pitch-high", "edge-11-roll-low")
BIASES_M = (-.004, 0.0, .004)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    previous = json.loads(V78_MATRIX.read_text())
    if (not previous["complete"] or previous["pass_count"] != 2 or
            [row["contact_s"] for row in previous["cases"]] !=
            [1.34, 1.36, .76, .56, .32, .24, .2]):
        raise ValueError("Expected the complete V78 contact-direction diagnostic")
    source_plan = json.loads(V73_PLAN.read_text())
    source = json.loads(V73_MATRIX.read_text())
    if (not source["complete"] or source["pass_count"] != 27 or
            not all(next(r for r in source["cases"] if r["id"] == ident)["pass"]
                    for ident in NEIGHBORS)):
        raise ValueError("Expected three source-matched passing but thin V73 neighbors")
    originals = {case["id"]: case for case in source_plan["cases"]}
    cases = [{**originals[ident], "diagnostic_y_bias_m": bias}
             for ident in NEIGHBORS for bias in BIASES_M]
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(v57.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v79-neighbor-offset-plan.v1",
        "role": "post_holdout_neighbor_bias_transfer_diagnostic_not_controller",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources, "runner_sha256": sha(Path(__file__)),
        "v73_plan_sha256": sha(V73_PLAN), "v73_matrix_sha256": sha(V73_MATRIX),
        "v78_plan_sha256": sha(v78.PLAN), "v78_matrix_sha256": sha(V78_MATRIX),
        "case_count": len(cases), "cases": cases,
        "gate": "Replay three previously thin V73 routes at -4, 0 and +4 mm prescribed body-Y goal offsets. Require the full native gate and >=1.2 s uninterrupted contact for an improved margin. Zero offset is a fresh-process control. Compare across selection mode, anchor and pitch/roll perturbation. These post-result probes cannot change the V73 holdout verdict.",
        "design": "The V78 -4 mm teacher rescued one V73 miss but held only 1.36 s. This sweep measures whether the same signed goal correction helps or harms the paired same-anchor roll route, another positive-camera pitch route, and a different edge route. Runtime control uses observed RGB; the offset is assigned by the experimenter, not inferred by a policy.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V79 plan differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    bias = case["diagnostic_y_bias_m"]
    tag = f"{bias * 1000:+.0f}mm".replace("+", "plus").replace("-", "minus")
    stem = DEST / f"seed-{case['seed']}-{case['id']}-{tag}"
    result, trace, log = (stem.with_suffix(f".{suffix}")
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
            raise RuntimeError(f"V79 prescribed route replay failed for {case['id']}/{tag}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    expected = {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v61_prescribed_y_bias_off_route_diagnostic_not_qualified",
        "diagnostic_y_bias_m": bias,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V79 source mismatch: {result}")
    if (report["pitch_offset_rad"] != case["pitch_offset_rad"] or
            report["roll_clearance_rad"] != case["roll_clearance_rad"]):
        raise ValueError(f"V79 route mismatch: {result}")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event.get("time_s") == report["reach_start_s"]), None)
    error = None if entry is None else entry.get("xy_error_truth_mm_scoring_only")
    if error is None or error > 10:
        faults.append("RGB goal error above 10 mm or unavailable")
    if report.get("image_goal_missing_at_s") is not None:
        faults.append("image goal missing during reach")
    return {
        "id": case["id"], "seed": case["seed"], "diagnostic_y_bias_m": bias,
        "selection_mode": report["selection_mode"], "pass": not faults,
        "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
        "entry_goal_error_mm_scoring_only": error, "faults": faults,
        "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }


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
        futures = {pool.submit(run_case, case, plan): (case["id"], case["diagnostic_y_bias_m"])
                   for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[(row["id"], row["diagnostic_y_bias_m"])] = row
            ordered = [rows[(case["id"], case["diagnostic_y_bias_m"])]
                       for case in plan["cases"]
                       if (case["id"], case["diagnostic_y_bias_m"]) in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v79-neighbor-offset-transfer.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                "margin_1p2_s_count": sum(r["pass"] and r["contact_s"] >= 1.2 for r in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "diagnostic_y_bias_m", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
