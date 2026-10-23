"""Pilot an observable offset confined to opened thin-contact route views."""

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
    run_v73_route_holdout as v73,
    run_v82_offset_coverage as v82,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v83"
PLAN = OUT / "observed-thin-route-pilot-plan.json"
DEST = OUT / "native-pilot"
EVALUATOR = Path(__file__).with_name("evaluate_observed_thin_route_v83.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v58.py")
AFFECTED = ("positive-01-pitch-low", "positive-01-roll-high",
            "positive-07-pitch-high", "edge-11-roll-low")
CONTROLS = ("large-32-pitch-low", "edge-02-roll-high",
            "large-28-roll-low", "positive-11-roll-high")
ORDER = AFFECTED + CONTROLS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    prior = json.loads((v73.DEST / "matrix.json").read_text())
    coverage = json.loads((v82.DEST / "matrix.json").read_text())
    if (not prior["complete"] or prior["pass_count"] != 27 or
            not coverage["complete"] or coverage["case_count"] != 20 or
            coverage["broad_minus4_gate_met"]):
        raise ValueError("Need completed V73 miss and failed V82 broad-offset gate")
    originals = {case["id"]: case for case in json.loads(v73.PLAN.read_text())["cases"]}
    baseline = {row["id"]: row for row in prior["cases"]}
    cases = [{**originals[ident], "v73_contact_s": baseline[ident]["contact_s"],
              "expected_thin_route_bias": ident in AFFECTED} for ident in ORDER]
    if not all(not baseline[ident]["pass"] for ident in AFFECTED[:1]) or not all(
            baseline[ident]["pass"] for ident in ORDER[1:]):
        raise ValueError("Opened pilot labels differ from frozen V73 screen")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(v57.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE),
                   runner=sha(Path(__file__)))
    plan = {
        "schema": "robot-learning-lab.cove50-v83-observed-thin-route-pilot-plan.v1",
        "role": "post_holdout_observable_rule_development_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v73_plan_sha256": sha(v73.PLAN),
        "v73_matrix_sha256": sha(v73.DEST / "matrix.json"),
        "v82_plan_sha256": sha(v82.PLAN),
        "v82_matrix_sha256": sha(v82.DEST / "matrix.json"),
        "affected_ids": AFFECTED, "control_ids": CONTROLS,
        "case_count": len(cases), "cases": cases,
        "gate": "All eight native visits must pass contact, support, rated effort, hand-floor, self-contact, clipping and scored RGB gates. The four opened thin routes must each hold at least 1.2 s uninterrupted fingertip contact and activate the new observed rule. All four controls must not activate it. A failed pilot stops before broader retention or new scoring. These are post-holdout opened cases on a rejected body.",
        "design": "The V82 blanket −4 mm offset harms two strong routes by more than its predeclared 0.5 s tolerance. V83 applies the same offset only for a commanded ±0.01 rad route with no earlier V70 route correction, where observed camera position and selected image-goal Y error identify the two opened thin-contact view regions. Positive-entry v16_locked uses camera X >= 0, Y <= −0.08 m and image error Y >= 0.018 m; edge-entry v21_scoped_y_locked uses camera X <= −0.06, Y >= 0.05 m and image error Y >= 0.0075 m. No seed, toy truth or contact score selects the action. The four targeted V73 routes and four off-region controls are frozen before V83 native replay.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V83 pilot differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    ident, seed = case["id"], case["seed"]
    result, trace, log = (DEST / f"seed-{seed}-{ident}.{suffix}"
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
               "--entry", "continuous", "--seed", str(seed),
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
            raise RuntimeError(f"V83 native replay failed for {ident}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if any(report.get(key) != value for key, value in {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v83_observed_thin_route_image_goal_candidate",
    }.items()):
        raise ValueError(f"V83 source identity mismatch: {result}")
    if (report["pitch_offset_rad"] != case["pitch_offset_rad"] or
            report["roll_clearance_rad"] != case["roll_clearance_rad"] or
            not report["off_authored_route_probe"]):
        raise ValueError(f"V83 route identity mismatch: {result}")
    faults = gate(report)
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
    if entry:
        camera_x, camera_y = entry["camera_world_xyz_m"][:2]
        selected_y = report["selection_features"]["selected_image_error_xy_m"][1]
        pitch_delta = case["pitch_offset_rad"] - .105
        roll_delta = case["roll_clearance_rad"] + .25
        eligible = (abs(pitch_delta) >= .005 or abs(roll_delta) >= .005) and not (
            report["automatic_front_route_bias_applied"] or
            report["automatic_front_center_bias_applied"] or
            report["automatic_edge_route_bias_applied"])
        expected_region = None
        if eligible:
            if (report["selection_mode"] == "v16_locked" and selected_y >= .018 and
                    camera_x >= 0 and camera_y <= -.08):
                expected_region = "positive_entry"
            elif (report["selection_mode"] == "v21_scoped_y_locked" and
                  selected_y >= .0075 and camera_x <= -.06 and camera_y >= .05):
                expected_region = "edge_entry"
        if (report["automatic_thin_route_bias_applied"] != (expected_region is not None) or
                report["automatic_thin_route_region"] != expected_region):
            faults.append("observable thin-route rule mismatch")
        if (expected_region is not None) != case["expected_thin_route_bias"]:
            faults.append("pilot branch exposure differs from frozen expectation")
        expected_bias = (-.004 if report["edge_goal_bias_applied"] else
                         -.002 if report["selection_mode"] in
                         ("v16_bias_y_locked", "v21_scoped_y_locked") else 0.0)
        expected_bias += report["automatic_front_route_bias_y_m"]
        expected_bias += -.004 if report["automatic_front_center_bias_applied"] else 0.0
        expected_bias += -.004 if report["automatic_edge_route_bias_applied"] else 0.0
        expected_bias += -.004 if expected_region is not None else 0.0
        if report["goal_bias_body_xy_m"] != [0.0, expected_bias]:
            faults.append("goal bias differs from observed entry rule")
    row = {
        "id": ident, "seed": seed, "route_variant": case["route_variant"],
        "v73_contact_s": case["v73_contact_s"],
        "selection_mode": report["selection_mode"],
        "expected_thin_route_bias": case["expected_thin_route_bias"],
        "automatic_thin_route_bias_applied": report["automatic_thin_route_bias_applied"],
        "automatic_thin_route_region": report["automatic_thin_route_region"],
        "entry_orange_pixels": None if entry is None else entry["orange_pixels"],
        "entry_goal_error_mm_scoring_only": (None if entry is None else
                                             entry.get("xy_error_truth_mm_scoring_only")),
        "contact_s": report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0),
        "pass": not faults, "faults": faults,
        "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }
    row["margin_1p2_s_met"] = row["contact_s"] >= 1.2
    return row


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
        futures = {pool.submit(run_case, case, plan): case["id"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
            gate_met = (len(ordered) == len(plan["cases"]) and
                        all(item["pass"] and
                            (item["margin_1p2_s_met"] if item["expected_thin_route_bias"] else
                             not item["automatic_thin_route_bias_applied"])
                            for item in ordered))
            matrix = {
                "schema": "robot-learning-lab.cove50-v83-observed-thin-route-pilot.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered), "pass_count": sum(item["pass"] for item in ordered),
                "thin_rule_use_count": sum(item["automatic_thin_route_bias_applied"]
                                           for item in ordered),
                "pilot_gate_met": gate_met, "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "automatic_thin_route_region", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
