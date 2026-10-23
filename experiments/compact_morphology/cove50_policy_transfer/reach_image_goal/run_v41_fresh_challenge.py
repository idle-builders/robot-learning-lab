"""Sealed disjoint 32-start native score for the V41 historical candidate."""

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
    run_v41_opened_gate as opened,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v41"
PLAN = OUT / "fresh-challenge-plan.json"
DEST = OUT / "fresh-reserved"
PREDECLARED = BASE / "v39/fresh-challenge-plan.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases() -> list[dict]:
    rows = json.loads(PREDECLARED.read_text())["cases"]
    if len(rows) != 32 or [r["seed"] for r in rows] != list(range(10801, 10833)):
        raise ValueError("Expected 32 disjoint starts")
    return rows


def freeze() -> dict:
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(screen.ACTOR), evaluator=sha(screen.EVALUATOR),
                   selector=sha(screen.SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {"schema": "robot-learning-lab.cove50-v41-fresh-challenge-plan.v1",
            "role": "predeclared_disjoint_native_score_only_after_opened_gate",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "source_sha256": sources,
            "opened_gate_plan_sha256": sha(opened.PLAN),
            "still_sealed_v39_case_plan_sha256": sha(PREDECLARED),
            "runner_sha256": sha(Path(__file__)),
            "case_count": 32, "cases": cases(),
            "promotion_gate": "All 32 newly seeded native starts pass the complete contact, support, effort, floor, clipping and RGB gates. At least four V10-height and four scoped-Y selections and at least one calibrated-route corner refinement must actually occur. One miss rejects promotion. The historical body remains rejected and no selected-body qualification follows.",
            "design": "The 10801-10832 case parameters were frozen for V39 and never executed because V39 and V40 both failed opened gates. V41 reuses those still-sealed cases without changing a seed, voltage, delay, toy offset or gaze. Four broad toy/gap strata and +/-1 mm perturbations were specified before this candidate's retention score; inference sees pixels and pose, not origins or seeds."}
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V41 fresh plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    result, trace, log = (DEST / f"seed-{seed}.{suffix}" for suffix in ("json", "npz", "log"))
    sources = v16.SOURCES
    command = [sys.executable, str(screen.EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(screen.ACTOR),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(sources["reference"]),
               "--geometry", str(sources["geometry"]),
               "--entry", "continuous", "--seed", str(seed),
               "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
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
            raise RuntimeError(f"V41 native evaluator failed for seed {seed}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if (report["evaluator_sha256"] != hashes["evaluator"] or
            report["selector_source_sha256"] != hashes["selector"] or
            report["template_source_sha256"] != hashes["template"] or
            report["body_sha256"] != hashes["body"] or
            report["selected_pointer_sha256"] != hashes["pointer"] or
            report["reach_checkpoint_sha256"] != hashes["actor"] or
            report["adaptive_variant"] != "v41_low_visibility_abstention_and_dagger_arm_candidate"):
        raise ValueError(f"V41 source identity mismatch: {result}")
    faults = gate(report)
    entry = next((e for e in report["image_goal_capture_events"] if e["time_s"] == 7.5), None)
    if not entry or not entry["used_for_control"] or not entry["template_applied"]:
        faults.append("image template unavailable at entry")
    if entry and entry["template_used_for_control"] != screen.expected_template(entry):
        faults.append("observable template rule mismatch")
    if entry and entry["xy_error_truth_mm_scoring_only"] > 10:
        faults.append("RGB goal error above 10 mm")
    if report["image_goal_missing_at_s"] is not None:
        faults.append("image goal missing during reach")
    route_goal_source = report["selection_features"]["route_goal_source"]
    if entry and entry["template_used_for_control"] and entry["orange_pixels"] >= 64:
        if route_goal_source != "pre_template_calibrated_rgb":
            faults.append("corner route not based on calibrated RGB")
    elif route_goal_source != "selected_image_goal":
        faults.append("ordinary route source changed")
    return {"seed": seed, "stratum": case["stratum"],
            "selection_mode": report["selection_mode"],
            "route_goal_source": route_goal_source, "pass": not faults,
            "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
            "entry_orange_pixels": None if entry is None else entry["orange_pixels"],
            "template_used_for_control": None if entry is None else entry["template_used_for_control"],
            "entry_goal_error_mm_scoring_only": None if entry is None else entry["xy_error_truth_mm_scoring_only"],
            "faults": faults, "result": str(result.relative_to(ROOT)),
            "trace": str(trace.relative_to(ROOT)), "log": str(log.relative_to(ROOT))}


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
    opened_result = json.loads((opened.DEST / "matrix.json").read_text())
    if (not opened_result["complete"] or opened_result["case_count"] != 75 or
            opened_result["pass_count"] != 75 or opened_result["regressions"] or
            opened_result["stopped_after_failed_stage"] or
            opened_result["plan_sha256"] != sha(opened.PLAN)):
            raise ValueError("Do not open fresh starts until all 75 V41 native retention cases pass")
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["seed"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[c["seed"]] for c in plan["cases"] if c["seed"] in rows]
            counts = {mode: sum(r["selection_mode"] == mode for r in ordered)
                      for mode in sorted({r["selection_mode"] for r in ordered})}
            corner_count = sum(r["route_goal_source"] == "pre_template_calibrated_rgb"
                               for r in ordered)
            matrix = {"schema": "robot-learning-lab.cove50-v41-fresh-challenge.v1",
                      "role": "predeclared_disjoint_native_score",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "opened_matrix_sha256": sha(opened.DEST / "matrix.json"),
                      "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "selection_counts": counts,
                      "calibrated_corner_route_count": corner_count,
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "selection_mode", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
