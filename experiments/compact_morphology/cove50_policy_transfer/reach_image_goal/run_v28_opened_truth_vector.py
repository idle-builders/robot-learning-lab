"""Screen the full post-score truth-informed RGB goal vector on an opened miss.

The vector is prescribed from scoring-only simulator truth after the V21 score.
This is mechanism diagnosis, never automatic inference or a fresh score.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v22_opened_failure_probes as helper,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V21 = BASE / "v21"
OUT = BASE / "v28"
PLAN = OUT / "opened-full-truth-vector-plan.json"
DEST = OUT / "opened-full-truth-vector"
EVALUATOR = Path(__file__).with_name("evaluate_truth_vector_v28.py")
SEEDS = (10604, 10601, 10602)
PROBES = (("truth_exact", 3.58, -4.19),
          ("truth_margin", 4.0, -5.0))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_plan() -> dict:
    score_path = V21 / "fresh-reserved/matrix.json"
    challenge_path = V21 / "fresh-challenge-plan.json"
    prior_path = BASE / "v27/opened-post-actor-roll/matrix.json"
    score = json.loads(score_path.read_text())
    challenge = json.loads(challenge_path.read_text())
    prior = json.loads(prior_path.read_text())
    if (not score["complete"] or score["case_count"] != 24 or
            score["pass_count"] != 22 or score["promotion_gate_pass"] or
            score["plan_sha256"] != sha(challenge_path)):
        raise ValueError("Need sealed failed V21 score")
    rows = {row["seed"]: row for row in score["cases"]}
    if ([rows[seed]["pass"] for seed in SEEDS] != [False, True, True] or
            any(rows[seed]["selection_mode"] != "v21_scoped_y_locked"
                for seed in SEEDS)):
        raise ValueError("Need opened scoped-Y miss and pass controls")
    if (not prior["complete"] or prior["case_count"] != 6 or
            prior["rescues"] or prior["regressions"]):
        raise ValueError("Need complete failed V27 physical-target screen")
    cases = {case["seed"]: case for case in challenge["cases"]}
    sources = helper.v16.SOURCES
    pointer = json.loads(sources["pointer"].read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical rejected-body diagnostic only")
    tasks = [{**cases[seed], "probe": label, "forced_mode": "v16_locked",
              "extra_goal_bias_body_xy_mm": [x, y],
              "baseline_result": rows[seed]["result"],
              "baseline_pass": rows[seed]["pass"],
              "baseline_contact_s": rows[seed]["contact_s"]}
             for seed in SEEDS for label, x, y in PROBES]
    return {
        "schema": "robot-learning-lab.cove50-v28-opened-full-truth-vector-plan.v1",
        "role": "postscore_opened_full_truth_informed_goal_screen_not_inference_or_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v21_fresh_matrix_sha256": sha(score_path),
        "v21_fresh_plan_sha256": sha(challenge_path),
        "v27_opened_matrix_sha256": sha(prior_path),
        "source_sha256": {name: sha(path) for name, path in sources.items()},
        "teacher_evaluator_sha256": sha(EVALUATOR),
        "helper_sha256": sha(Path(helper.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "paired_pre_reach_tolerance": {"qpos_max_abs": 5e-5,
                                        "action_max_abs": 1e-3,
                                        "rgb_entry_max_abs_m": 5e-4},
        "truth_label_provenance": {
            "opened_seed": 10604,
            "body_frame_image_minus_truth_xy_mm_at_reach_entry": [-3.58, 4.19],
            "needed_negative_of_error_xy_mm": [3.58, -4.19],
            "derivation": "Scored V21 entry RGB estimate minus simulator cube center, rotated into the reached pelvis frame at trace index 375 after the fresh score was sealed. Screen one nearly exact correction and one larger contact-margin vector within a 6 mm per-axis diagnostic bound. This truth is never an inference input.",
        },
        "tasks": tasks,
        "qualification": "Two simulator-truth-informed goal biases on one opened failure and two prior passes. Even a successful probe only identifies a perception/action sensitivity; a camera-observable correction with varied retention and a separately frozen disjoint native test would be required for promotion.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = frozen_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    expected = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists():
        if PLAN.read_bytes() != expected:
            raise ValueError("Frozen V28 full-truth diagnostic plan differs")
    else:
        PLAN.write_bytes(expected)
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    helper.DEST = DEST
    helper.EVALUATOR = EVALUATOR
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(helper.run_task, task, plan): task
                   for task in plan["tasks"]}
        for future in as_completed(futures):
            row = future.result()
            report = json.loads((ROOT / row["result"]).read_text())
            if report.get("goal_bias_provenance") != (
                    "v28_postscore_simulator_truth_informed_counterfactual_not_deployable"):
                raise ValueError("Truth-vector provenance missing")
            rows[(row["seed"], row["probe"])] = row
            ordered = [rows[(task["seed"], task["probe"])]
                       for task in plan["tasks"]
                       if (task["seed"], task["probe"]) in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v28-opened-full-truth-vector-native.v1",
                "role": plan["role"],
                "plan_sha256": sha(PLAN),
                "complete": len(ordered) == len(plan["tasks"]),
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "rescues": [[item["seed"], item["probe"]] for item in ordered
                            if item["pass"] and not item["baseline_pass"]],
                "regressions": [[item["seed"], item["probe"]] for item in ordered
                                if not item["pass"] and item["baseline_pass"]],
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
