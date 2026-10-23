"""Find prescribed contact teachers for three V60 off-route edge-entry misses."""

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
    run_v60_route_probe as v60,
    run_v61_ybias_probe as v61,
)

OUT = v60.BASE / "v61-edge"
PLAN = OUT / "prescribed-edge-y-bias-plan.json"
DEST = OUT / "opened-edge-y-bias"
EVALUATOR = v61.EVALUATOR
SELECTOR = v61.SELECTOR
CASE_IDS = ("edge-12-pitch-low", "edge-12-roll-low", "edge-12-roll-high")
ADDITIONAL_BIASES_M = (-.004, -.003, -.002, 0.0, .002)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    route_path = v60.DEST / "matrix.json"
    route = json.loads(route_path.read_text())
    if not route["complete"] or route["case_count"] != 35:
        raise ValueError("Expected complete V60 nearby-route diagnostic")
    by_id = {row["id"]: row for row in route["cases"]}
    if any(by_id[ident]["pass"] for ident in CASE_IDS):
        raise ValueError("Expected three failed V60 edge-12 route visits")
    original = {case["id"]: case for case in json.loads(v60.PLAN.read_text())["cases"]}
    cases = [{**original[ident], "baseline_v60_contact_s": by_id[ident]["contact_s"],
              "diagnostic_y_bias_m": bias,
              "id": f"{ident}-{bias*1000:+.0f}mm"}
             for ident in CASE_IDS for bias in ADDITIONAL_BIASES_M]
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(v57.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v61-prescribed-edge-y-bias-plan.v1",
        "role": "opened_experimenter_prescribed_edge_goal_offset_search_not_controller",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources, "runner_sha256": sha(Path(__file__)),
        "v60_plan_sha256": sha(v60.PLAN), "v60_matrix_sha256": sha(route_path),
        "v60_failure_report_sha256": {ident: sha(ROOT / by_id[ident]["result"])
                                      for ident in CASE_IDS},
        "case_count": len(cases), "cases": cases,
        "gate": "Find at least one >=1.2 s full-gate native passing offset for each of the three exact edge-12 V60 contact failures. Each of fifteen visits must evaluate support, rated effort, fingertip contact, hand floor, self-contact, clipping and <=10 mm RGB error. Prescribed offsets are diagnostic teachers, not an automatic route or image policy.",
        "design": "Hold each failed edge-12 seed, toy, motor, gaze, pitch, roll, trained V57 actor and V59 observed image selector fixed. Sweep five additional body-Y offsets on top of the ordinary -2 mm scoped-Y bias: -4, -3, -2, 0 and +2 mm. This tests total biases -6 to 0 mm with no simulator truth in control.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V61 edge plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    bias = case["diagnostic_y_bias_m"]
    ident = case["id"]
    result, trace, log = (DEST / f"seed-{case['seed']}-{ident}.{suffix}"
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
            raise RuntimeError(f"V61 edge evaluator failed for {ident}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if any(report.get(key) != value for key, value in {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v61_prescribed_y_bias_off_route_diagnostic_not_qualified",
        "diagnostic_y_bias_m": bias,
        "goal_bias_body_xy_m": [0.0, -.002 + bias],
        "off_authored_route_probe": True,
    }.items()):
        raise ValueError(f"V61 edge source or bias mismatch: {result}")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event.get("time_s") == report["reach_start_s"]), None)
    error = None if entry is None else entry.get("xy_error_truth_mm_scoring_only")
    if error is None or error > 10:
        faults.append("RGB goal error above 10 mm or unavailable")
    if report.get("image_goal_missing_at_s") is not None:
        faults.append("image goal missing during reach")
    return {"id": ident, "anchor_id": case["anchor_id"],
            "route_variant": case["route_variant"], "seed": case["seed"],
            "diagnostic_y_bias_m": bias, "total_goal_bias_y_m": -.002 + bias,
            "baseline_v60_contact_s": case["baseline_v60_contact_s"],
            "pass": not faults,
            "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
            "entry_goal_error_mm_scoring_only": error, "faults": faults,
            "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
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
        futures = {pool.submit(run_case, case, plan): case["id"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
            found = {ident: any(r["pass"] and r["contact_s"] >= 1.2
                                for r in ordered if r["id"].startswith(ident + "-"))
                     for ident in CASE_IDS}
            matrix = {"schema": "robot-learning-lab.cove50-v61-prescribed-edge-y-bias.v1",
                      "role": plan["role"], "body_role": plan["body_role"],
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "teacher_found_by_route": found,
                      "teacher_gate_met": len(ordered) == len(plan["cases"]) and all(found.values()),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "entry_goal_error_mm_scoring_only", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
