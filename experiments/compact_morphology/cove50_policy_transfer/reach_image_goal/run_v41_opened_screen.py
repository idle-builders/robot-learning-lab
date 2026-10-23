"""Frozen native screen for the V41 actor and low-visibility image rule."""

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
    run_v39_corner_image_screen as image,
    run_v40_route_stability_screen as route,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v41"
PLAN = OUT / "opened-screen-plan.json"
DEST = OUT / "opened-screen"
ACTOR = OUT / "actor.pt"
EVALUATOR = Path(__file__).with_name("evaluate_selective_goal_v41.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v41.py")
TARGETS = (10545, 10546, 10528, 10508, 10513, 10530, 10538, 10539,
           10510, 10514, 10604, 10529, 10544, 10547, 10548, 10608, 10621)
VARIANT = "v41_low_visibility_abstention_and_dagger_arm_candidate"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure() -> None:
    v16.SOURCES["actor"] = ACTOR
    core.EVALUATOR = EVALUATOR
    core.SELECTOR = SELECTOR
    core.DEST = DEST
    core.EXPECTED_VARIANT = VARIANT


def freeze() -> dict:
    configure()
    prior = BASE / "v40/opened-gate-plan.json"
    previous = json.loads(prior.read_text())
    by_seed = {case["seed"]: case for case in previous["cases"]}
    cases = []
    for seed in TARGETS:
        old = by_seed[seed]
        cases.append({**old, "role": ("v40_regression" if seed in (10545, 10546)
                                     else "opened_retention_control")})
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(evaluator=sha(EVALUATOR), selector=sha(SELECTOR),
                   template=sha(core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v41-opened-screen-plan.v1",
        "role": "opened_actor_and_low_visibility_diagnostic_not_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v40_opened_plan_sha256": sha(prior),
        "v40_gate_stop_sha256": sha(BASE / "v40/opened-native/gate-stop.json"),
        "training_plan_sha256": sha(OUT / "actor-training-plan.json"),
        "core_runner_sha256": sha(Path(core.__file__)),
        "image_runner_sha256": sha(Path(image.__file__)),
        "route_runner_sha256": sha(Path(route.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "gate": "Both V40 regression starts and all fifteen diverse opened controls pass native contact, support, motor, image and source checks; then run a separate full 75-start retention gate. No fresh cases are consumed here.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V41 screen plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def expected_template(event: dict) -> bool:
    if not event["template_applied"] or event["orange_pixels"] < 24:
        return False
    correction = float(np.linalg.norm(event["template_correction_xy_mm"]))
    camera_x, camera_y = event["camera_world_xyz_m"][:2]
    return ((event["orange_pixels"] < 64 and correction >= 3.0) or
            (0.03 <= camera_x < 0.08 and camera_y < -0.08 and
             correction >= 2.0 and event["template_pixels_disagree"] <= 10))


def run_case(case: dict, plan: dict) -> dict:
    row = route.run_case(case, plan)
    report = json.loads((ROOT / row["result"]).read_text())
    entry = next(e for e in report["image_goal_capture_events"] if e["time_s"] == 7.5)
    row["faults"] = [f for f in row["faults"] if f != "V39 observable template rule mismatch"]
    if entry["template_used_for_control"] != expected_template(entry):
        row["faults"].append("V41 observable template rule mismatch")
    if report["reach_checkpoint_sha256"] != plan["source_sha256"]["actor"]:
        row["faults"].append("V41 actor identity mismatch")
    row["template_abstained_low_visibility"] = entry["template_abstained_low_visibility"]
    row["pass"] = not row["faults"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    rows: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["seed"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[c["seed"]] for c in plan["cases"] if c["seed"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v41-opened-screen.v1",
                "role": "opened_actor_and_low_visibility_diagnostic_not_fresh",
                "body_role": "rejected_historical_direct_frame_sandbox",
                "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "pass", "contact_s", "entry_orange_pixels",
                               "template_used_for_control", "faults")}), flush=True)


if __name__ == "__main__":
    main()
