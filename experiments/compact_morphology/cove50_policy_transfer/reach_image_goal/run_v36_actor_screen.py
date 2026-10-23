"""Paired native screen of V36 arm weights with the V35 observable RGB rule."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
    run_v35_wider_rgb_screen as core,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v36"
PLAN = OUT / "actor-screen-plan.json"
DEST = OUT / "actor-screen"
ACTOR = OUT / "actor.pt"
TRAINING = OUT / "training-report.json"
SOURCES = (
    BASE / "v21/opened-native/matrix.json",
    BASE / "v32/opened-native/matrix.json",
    BASE / "v32/prior-failure-screen/matrix.json",
    BASE / "v35/wider-rgb-screen/matrix.json",
)
CASE_PLANS = (
    BASE / "v21/opened-gate-plan.json",
    BASE / "v32/opened-gate-plan.json",
    BASE / "v32/prior-failure-screen-plan.json",
)
TARGETS = (10530, 10538, 10534, 10539, 10513, 10533,
           10608, 10612, 10620, 10621)
BASELINE_COHORT = {
    10530: 2, 10538: 3, 10534: 2, 10539: 2, 10513: 2,
    10533: 0, 10608: 3, 10612: 3, 10620: 1, 10621: 1,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    training = json.loads(TRAINING.read_text())
    if training["actor_sha256"] != sha(ACTOR):
        raise ValueError("V36 saved actor differs from training report")
    matrices = [json.loads(path.read_text()) for path in SOURCES]
    if any(not matrix["complete"] for matrix in matrices):
        raise ValueError("Need complete earlier native matrices")
    baseline = [{row["seed"]: row for row in matrix["cases"]} for matrix in matrices]
    cases = {row["seed"]: row for path in CASE_PLANS
             for row in json.loads(path.read_text())["cases"]}
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(ACTOR), evaluator=sha(core.EVALUATOR),
                   selector=sha(core.SELECTOR), template=sha(core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v36-actor-screen-plan.v1",
        "role": "opened_new_actor_and_wider_RGB_paired_diagnostic_not_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "actor_training_plan_sha256": sha(OUT / "actor-training-plan.json"),
        "actor_training_report_sha256": sha(TRAINING),
        "source_matrix_sha256": {str(path.relative_to(ROOT)): sha(path) for path in SOURCES},
        "source_case_plan_sha256": {str(path.relative_to(ROOT)): sha(path) for path in CASE_PLANS},
        "source_sha256": sources,
        "core_runner_sha256": sha(Path(core.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(TARGETS),
        "cases": [
            {"seed": seed, "role": "former_failure" if seed in (10530,10534,10539,10513) else
                              "image_rescue_retention" if seed == 10538 else "pass_control",
             "baseline_pass": baseline[BASELINE_COHORT[seed]][seed]["pass"],
             "baseline_contact_s": baseline[BASELINE_COHORT[seed]][seed]["contact_s"],
             "baseline_trace": baseline[BASELINE_COHORT[seed]][seed]["trace"],
             **{key: cases[seed][key] for key in
                ("rail_v", "delay_ms", "toy_offset_x_m", "toy_offset_y_m", "gaze_head_yaw_rad")}}
            for seed in TARGETS
        ],
        "qualification": "This opened paired screen cannot promote the actor. One-second contact, support, motor and RGB gates apply; all previous passes and disjoint fresh starts remain required.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists():
        if PLAN.read_bytes() != encoded:
            raise ValueError("Frozen V36 actor screen differs from source")
    else:
        PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    v16.SOURCES["actor"] = ACTOR
    core.DEST = DEST
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(core.run_case, case, plan): case["seed"]
                   for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v36-actor-screen.v1",
                      "role": "opened_new_actor_and_wider_RGB_paired_diagnostic_not_fresh",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN),
                      "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered),
                      "pass_count": sum(row["pass"] for row in ordered),
                      "rescues": [row["seed"] for row in ordered if row["pass"] and not row["baseline_pass"]],
                      "regressions": [row["seed"] for row in ordered if row["baseline_pass"] and not row["pass"]],
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "pass", "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
