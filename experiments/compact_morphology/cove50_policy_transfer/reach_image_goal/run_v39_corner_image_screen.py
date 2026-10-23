"""Paired native screen of an observable corner image correction."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
    run_v35_wider_rgb_screen as core,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v39"
PLAN = OUT / "corner-image-screen-plan.json"
DEST = OUT / "corner-image-screen"
ACTOR = BASE / "v36/actor.pt"
EVALUATOR = Path(__file__).with_name("evaluate_selective_goal_v39.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v39.py")
TARGETS = (10508, 10501, 10513, 10538, 10609, 10615)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_template(event: dict) -> bool:
    if not event["template_applied"]:
        return False
    correction = float(np.linalg.norm(event["template_correction_xy_mm"]))
    camera_x, camera_y = event["camera_world_xyz_m"][:2]
    return ((event["orange_pixels"] < 64 and correction >= 3.0) or
            (0.03 <= camera_x < 0.08 and camera_y < -0.08 and
             correction >= 2.0 and event["template_pixels_disagree"] <= 10))


def freeze() -> dict:
    cohorts = (("v21", "opened-native"), ("v32", "opened-native"))
    cases = {}
    baselines = {}
    identities = {}
    for version, directory in cohorts:
        plan_path = BASE / version / "opened-gate-plan.json"
        matrix_path = BASE / version / directory / "matrix.json"
        identities[str(plan_path.relative_to(ROOT))] = sha(plan_path)
        identities[str(matrix_path.relative_to(ROOT))] = sha(matrix_path)
        cases.update({c["seed"]: c for c in json.loads(plan_path.read_text())["cases"]})
        baselines.update({c["seed"]: c for c in json.loads(matrix_path.read_text())["cases"]})
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(core.TEMPLATE))
    rows = []
    for seed in TARGETS:
        case, old = cases[seed], baselines[seed]
        rows.append({"seed": seed, "role": "corner_failure" if seed == 10508 else "route_or_image_retention",
                     "baseline_pass": old["pass"], "baseline_contact_s": old["contact_s"],
                     "baseline_trace": old["trace"],
                     **{key: case[key] for key in ("rail_v", "delay_ms", "toy_offset_x_m",
                                                    "toy_offset_y_m", "gaze_head_yaw_rad")}})
    plan = {"schema": "robot-learning-lab.cove50-v39-corner-image-screen-plan.v1",
            "role": "opened_observable_corner_image_diagnostic_not_fresh",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "source_sha256": sources, "source_plan_and_matrix_sha256": identities,
            "v38_plan_sha256": sha(BASE / "v38/observable-router-gate-plan.json"),
            "core_runner_sha256": sha(Path(core.__file__)),
            "runner_sha256": sha(Path(__file__)),
            "cases": rows, "case_count": len(rows),
            "rule": "Retain V35 sparse-image rule; additionally use rendered image correction when 0.03 <= observed camera X < 0.08 m, camera Y < -0.08 m, correction >=2 mm, and pixel disagreement <=10. No toy truth or seed enters inference.",
            "qualification": "Six opened paired starts screen the 10508 corner and adjacent retention only. Full 75-start retention and fresh disjoint challenge are required before promotion."}
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V39 plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    row = core.run_case(case, plan)
    report = json.loads((ROOT / row["result"]).read_text())
    entry = next(e for e in report["image_goal_capture_events"] if e["time_s"] == 7.5)
    faults = [f for f in row["faults"] if f != "observable template rule mismatch"]
    if entry["template_used_for_control"] != expected_template(entry):
        faults.append("V39 observable template rule mismatch")
    row["faults"] = faults
    row["pass"] = not faults
    row["selection_mode"] = report["selection_mode"]
    row["entry_camera_world_xyz_m"] = entry["camera_world_xyz_m"]
    row["template_correction_xy_mm"] = entry["template_correction_xy_mm"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    v16.SOURCES["actor"] = ACTOR
    core.EVALUATOR = EVALUATOR
    core.SELECTOR = SELECTOR
    core.DEST = DEST
    core.EXPECTED_VARIANT = "v39_v36_actor_observable_route_and_corner_image_candidate"
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["seed"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[c["seed"]] for c in plan["cases"] if c["seed"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v39-corner-image-screen.v1",
                      "role": "opened_observable_corner_image_diagnostic_not_fresh",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "pass", "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
