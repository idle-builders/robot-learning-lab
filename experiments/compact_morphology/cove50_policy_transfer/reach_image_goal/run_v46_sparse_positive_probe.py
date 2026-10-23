"""Paired opened goal-blend diagnostic on the V45 11003 contact miss."""

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
    run_v45_combined_screen as screen,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v46"
PLAN = OUT / "sparse-positive-probe-plan.json"
DEST = OUT / "opened-sparse-positive-probe"
EVALUATOR = Path(__file__).with_name("evaluate_sparse_positive_probe_v46.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v46_probe.py")
BLENDS = (.5, 0.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    screened_path = screen.DEST / "matrix.json"
    screened = json.loads(screened_path.read_text())
    if (not screened["complete"] or screened["case_count"] != 27 or
            screened["plan_sha256"] != sha(screen.PLAN)):
        raise ValueError("Complete V45 screen required before postscore diagnostic")
    baseline = next(r for r in screened["cases"] if r["seed"] == 11003)
    if baseline["pass"] or baseline["contact_s"] != .92:
        raise ValueError("Expected opened V45 11003 short-contact failure")
    source_plan = json.loads(screen.PLAN.read_text())
    case = next(c for c in source_plan["cases"] if c["seed"] == 11003)
    sources = {**source_plan["source_sha256"],
               "evaluator": sha(EVALUATOR), "selector": sha(SELECTOR)}
    visits = [{"id": f"11003-blend-{str(blend).replace('.', '')}",
               "blend": blend, "case": case} for blend in BLENDS]
    plan = {
        "schema": "robot-learning-lab.cove50-v46-sparse-positive-probe-plan.v1",
        "role": "opened_postscore_goal_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v45_screen_plan_sha256": sha(screen.PLAN),
        "v45_screen_matrix_sha256": sha(screened_path),
        "baseline_v45_contact_s": baseline["contact_s"],
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(visits), "visits": visits,
        "hypothesis": "V45 11003 sees 33 toy pixels and a +6/+2 mm rendered-template shift, with 0.194 mm scored entry error yet only 0.92 s continuous contact. Vary only the positive-camera sparse template-goal blend; keep its native start, actor, timing and motor route fixed. This probes contact sensitivity and is not a general selector rule.",
        "gate": "Score each visit on the unchanged native support, rated effort, >=1 s continuous fingertip contact, floor, clipping and RGB <10 mm gates. Any proposed policy needs matched controls, full retention, route-variance training and new disjoint score.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V46 diagnostic plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def run(visit: dict, plan: dict) -> dict:
    case = visit["case"]
    ident = visit["id"]
    result = DEST / f"{ident}.json"
    trace = DEST / f"{ident}.npz"
    log = DEST / f"{ident}.log"
    sources = screen.v16.SOURCES
    cmd = [sys.executable, str(EVALUATOR),
           "--body-xml", str(sources["body"]),
           "--calibration-model", str(sources["calibration"]),
           "--squat-checkpoint", str(sources["squat"]),
           "--reach-checkpoint", str(screen.screen.ACTOR),
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
           "--max-goal-age-s", "9", "--sparse-positive-blend", str(visit["blend"]),
           "--out", str(result), "--trace", str(trace)]
    if not result.exists() or not trace.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
             "/root/microduck_rl/src", "/workspace/microduck_rl/src"]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        with log.open("w") as stream:
            code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise RuntimeError(f"V46 native evaluator failed: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    for key, expected in (("evaluator_sha256", hashes["evaluator"]),
                          ("selector_source_sha256", hashes["selector"]),
                          ("template_source_sha256", hashes["template"]),
                          ("body_sha256", hashes["body"]),
                          ("selected_pointer_sha256", hashes["pointer"]),
                          ("reach_checkpoint_sha256", hashes["actor"]),
                          ("adaptive_variant", "v46_opened_sparse_positive_camera_blend_diagnostic"),
                          ("sparse_positive_blend", visit["blend"])):
        if report.get(key) != expected:
            raise ValueError(f"V46 source/parameter mismatch {key}: {result}")
    faults = gate(report)
    entry = next((e for e in report["image_goal_capture_events"]
                  if e.get("used_for_control")), None)
    if not entry or entry.get("sparse_positive_blend_diagnostic") != visit["blend"]:
        faults.append("diagnostic image blend not exercised at entry")
    if entry and entry.get("xy_error_truth_mm_scoring_only", 1e9) > 10:
        faults.append("RGB goal error above 10 mm")
    if report.get("image_goal_missing_at_s") is not None:
        faults.append("image goal missing during reach")
    return {"id": ident, "seed": case["seed"], "sparse_positive_blend": visit["blend"],
            "baseline_v45_contact_s": plan["baseline_v45_contact_s"],
            "pass": not faults,
            "contact_s": report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0),
            "selection_mode": report.get("selection_mode"),
            "entry_orange_pixels": None if entry is None else entry.get("orange_pixels"),
            "entry_goal_error_mm_scoring_only": None if entry is None else entry.get("xy_error_truth_mm_scoring_only"),
            "faults": faults,
            "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, visit, plan) for visit in plan["visits"]]
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[v["id"]] for v in plan["visits"] if v["id"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v46-sparse-positive-probe.v1",
                      "role": "opened_postscore_goal_diagnostic_not_fresh_score",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["visits"]),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "entry_goal_error_mm_scoring_only", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
