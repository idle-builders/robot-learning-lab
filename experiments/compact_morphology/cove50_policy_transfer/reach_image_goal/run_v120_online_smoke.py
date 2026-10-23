"""Source-frozen opened replay of the observed selector inside native reach."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v83_thin_route_pilot as v83,
    v118_observable_selector as selector,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V113_PLAN = BASE / "v113/varied-paired-contact-data-plan.json"
V113_MATRIX = BASE / "v113/native-varied-paired-contact-data/matrix.json"
MODEL = BASE / "v118/observable-selector.json"
EVALUATOR = Path(__file__).with_name("evaluate_online_selector_v120.py")
OUT = BASE / "v120"
PLAN = OUT / "online-opened-smoke-plan.json"
DEST = OUT / "native-opened-smoke"
SEEDS = (19050, 19057, 19030, 19036)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)


def freeze() -> dict:
    if PLAN.exists():
        plan = json.loads(PLAN.read_text())
        if (plan["source_v113_plan_sha256"] != sha(V113_PLAN) or
                plan["evaluator_sha256"] != sha(EVALUATOR) or
                plan["model_sha256"] != sha(MODEL) or
                plan["inference_source_sha256"] != sha(Path(selector.__file__)) or
                plan["runner_source_sha256"] != sha(Path(__file__)) or
                any(sha(ROOT / case["source_baseline_report"]) !=
                    case["source_baseline_sha256"] for case in plan["cases"])):
            raise ValueError("Frozen online smoke source changed")
        return plan
    source_plan = json.loads(V113_PLAN.read_text())
    paired = json.loads(V113_MATRIX.read_text())
    model = selector.load(MODEL)
    if model["inference_source_sha256"] != sha(Path(selector.__file__)):
        raise ValueError("Frozen selector inference source changed")
    source_cases = {case["seed"]: case for case in source_plan["cases"]}
    paired_cases = {case["seed"]: case for case in paired["cases"]}
    cases = []
    for seed in SEEDS:
        case, row = source_cases[seed], paired_cases[seed]
        baseline_path = ROOT / row["baseline_result"]
        report = json.loads(baseline_path.read_text())
        features = selector.observed_features(report)
        decision = selector.predict(model, features)
        cases.append({**case, "source_baseline_report": row["baseline_result"],
                      "source_baseline_sha256": sha(baseline_path),
                      "entry_observed_features": features,
                      "frozen_expected_decision": decision,
                      "paired_reference_contact_s": (row["shifted_contact_s"]
                                                     if decision["shift_body_y_m"] < 0
                                                     else row["baseline_contact_s"]),
                      "paired_reference_pass": (row["shifted_pass"]
                                                 if decision["shift_body_y_m"] < 0
                                                 else row["baseline_pass"])})
    plan = {
        "schema": "robot-learning-lab.cove50-v120-online-opened-smoke-plan.v1",
        "role": "opened_online_integration_smoke_not_qualification",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_v113_plan_sha256": sha(V113_PLAN),
        "source_v113_matrix_sha256_at_freeze": sha(V113_MATRIX),
        "base_source_sha256": source_plan["source_sha256"],
        "evaluator_sha256": sha(EVALUATOR),
        "model_sha256": sha(MODEL),
        "inference_source_sha256": sha(Path(selector.__file__)),
        "runner_source_sha256": sha(Path(__file__)),
        "method": ("Run one uninterrupted native simulation per opened V113 "
                   "start. Infer the target shift at reach entry from observed RGB, "
                   "camera, route and motor conditions inside that simulation. "
                   "Compare its full gate and contact with the independent "
                   "V113 unchanged or prescribed −2 mm visit. No outcome is "
                   "used as an inference input."),
        "case_count": len(cases), "cases": cases,
    }
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    OUT.mkdir(parents=True, exist_ok=True)
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    stem = f"seed-{seed}-online-{seed}"
    result, trace, log = (DEST / f"{stem}.{suffix}"
                          for suffix in ("json", "npz", "log"))
    sources = v83.v16.SOURCES
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(v83.v57.ACTOR),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
               "--reach-controller", "learned-residual",
               "--selector-model", str(MODEL),
               "--reach-reference", str(sources["reference"]),
               "--geometry", str(sources["geometry"]),
               "--entry", "continuous", "--seed", str(seed),
               "--rail-v", str(case["rail_v"]),
               "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", str(case["pitch_offset_rad"]),
               "--roll-clearance-rad", str(case["roll_clearance_rad"]),
               "--terminal-roll-offset-rad", "-.08",
               "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9",
               "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7",
               "--capture-stride-steps", "5", "--max-goal-age-s", "9",
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
            raise RuntimeError(f"V120 native replay failed for {seed}: {log}")
    report = json.loads(result.read_text())
    source = plan["base_source_sha256"]
    expected = {
        "evaluator_sha256": plan["evaluator_sha256"],
        "online_selector_model_sha256": plan["model_sha256"],
        "online_selector_source_sha256": plan["inference_source_sha256"],
        "body_sha256": source["body"],
        "selected_pointer_sha256": source["pointer"],
        "reach_checkpoint_sha256": source["actor"],
        "adaptive_variant": "v120_online_observable_contact_selector",
        "selection_mode": case["expected_selection_mode"],
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V120 source or route identity mismatch: {result}")
    decision = report["online_selector_decision"]
    if decision != case["frozen_expected_decision"]:
        raise ValueError(f"Online decision differs from frozen observed entry: {seed}")
    faults = v83.gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event["time_s"] == report["reach_start_s"]), None)
    if not entry or not entry.get("used_for_control") or not entry.get("template_applied"):
        faults.append("image template unavailable at reach entry")
    if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                  entry["xy_error_truth_mm_scoring_only"] > 10):
        faults.append("RGB goal error above 10 mm")
    if report["image_goal_missing_at_s"] is not None:
        faults.append("image goal missing during reach")
    if report["reacquisition_probe_orange_pixels"] == 0 and not (
            report["reacquisition_trigger_s"] is not None and
            report["reacquisition_head_yaw_rad"] in (-1.3, -1.5) and
            entry and entry.get("orange_pixels", 0) >= 10):
        faults.append("physical camera reacquisition did not observe toy")
    contact = report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0)
    return {
        "id": case["id"], "seed": seed,
        "route_variant": case["route_variant"],
        "selection_mode": report["selection_mode"],
        "online_decision": decision,
        "entry_feature_exact_match": report["online_selector_features"] ==
                                     case["entry_observed_features"],
        "contact_s": contact,
        "paired_reference_contact_s": case["paired_reference_contact_s"],
        "contact_difference_s": round(contact - case["paired_reference_contact_s"], 3),
        "pass": not faults, "faults": faults,
        "paired_reference_pass": case["paired_reference_pass"],
        "gate_parity": (not faults) == case["paired_reference_pass"],
        "margin_1p2_s_met": contact >= 1.2,
        "result": str(result.relative_to(ROOT)),
        "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    print(f"frozen {PLAN} sha256={sha(PLAN)} cases={plan['case_count']}", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "matrix.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if old and old["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing online matrix belongs to another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    for case in plan["cases"]:
        if case["id"] in rows:
            continue
        row = run_case(case, plan)
        rows[row["id"]] = row
        ordered = [rows[item["id"]] for item in plan["cases"] if item["id"] in rows]
        write_json(path, {
            "schema": "robot-learning-lab.cove50-v120-online-opened-smoke.v1",
            "role": plan["role"], "body_role": plan["body_role"],
            "plan_sha256": sha(PLAN),
            "case_count": len(ordered), "planned_count": plan["case_count"],
            "complete": len(ordered) == plan["case_count"],
            "pass_count": sum(item["pass"] for item in ordered),
            "gate_parity_count": sum(item["gate_parity"] for item in ordered),
            "cases": ordered,
        })
        print(json.dumps({key: row[key] for key in
                          ("id", "online_decision", "contact_s", "pass",
                           "gate_parity")}), flush=True)


if __name__ == "__main__":
    main()
