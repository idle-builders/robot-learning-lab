"""Screen saved arm actors and terminal routes on opened positive-X cases."""

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
OUT = BASE / "v25"
PLAN = OUT / "opened-actor-route-plan.json"
DEST = OUT / "opened-actor-route"
SEEDS = (10604, 10601, 10602)
PROBES = (("v10_height", "v16_v10_height_locked"),
          ("v10_locked", "v16_v10_locked"),
          ("specialist", "v16_specialist_locked"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_plan() -> dict:
    score_path = V21 / "fresh-reserved/matrix.json"
    challenge_path = V21 / "fresh-challenge-plan.json"
    score = json.loads(score_path.read_text())
    challenge = json.loads(challenge_path.read_text())
    if (not score["complete"] or score["case_count"] != 24 or
            score["pass_count"] != 22 or score["promotion_gate_pass"] or
            score["plan_sha256"] != sha(challenge_path)):
        raise ValueError("Need sealed failed V21 score")
    rows = {row["seed"]: row for row in score["cases"]}
    if ([rows[seed]["pass"] for seed in SEEDS] != [False, True, True] or
            any(rows[seed]["selection_mode"] != "v21_scoped_y_locked"
                for seed in SEEDS)):
        raise ValueError("Need opened scoped-Y miss and controls")
    cases = {case["seed"]: case for case in challenge["cases"]}
    sources = helper.v16.SOURCES
    pointer = json.loads(sources["pointer"].read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical rejected-body diagnostic only")
    tasks = [{**cases[seed], "probe": label, "forced_mode": mode,
              "extra_goal_bias_body_xy_mm": [0.0, 0.0],
              "baseline_result": rows[seed]["result"],
              "baseline_pass": rows[seed]["pass"],
              "baseline_contact_s": rows[seed]["contact_s"]}
             for seed in SEEDS for label, mode in PROBES]
    return {
        "schema": "robot-learning-lab.cove50-v25-opened-actor-route-plan.v1",
        "role": "postscore_opened_forced_saved_actor_route_screen_not_automatic_or_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v21_fresh_matrix_sha256": sha(score_path),
        "v21_fresh_plan_sha256": sha(challenge_path),
        "source_sha256": {name: sha(path) for name, path in sources.items()},
        "teacher_evaluator_sha256": sha(helper.EVALUATOR),
        "helper_sha256": sha(Path(helper.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "paired_pre_reach_tolerance": {"qpos_max_abs": 5e-5,
                                        "action_max_abs": 1e-3,
                                        "rgb_entry_max_abs_m": 5e-4},
        "tasks": tasks,
        "qualification": "Three saved actor/terminal-route alternatives forced on an opened positive-X miss and two passing controls. Any rescue needs an observable router, varied retention and separately frozen disjoint native testing; this screen is not an automatic policy or new-body evidence.",
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
            raise ValueError("Frozen V25 actor-route plan differs")
    else:
        PLAN.write_bytes(expected)
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    helper.DEST = DEST
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(helper.run_task, task, plan): task
                   for task in plan["tasks"]}
        for future in as_completed(futures):
            row = future.result()
            rows[(row["seed"], row["probe"])] = row
            ordered = [rows[(task["seed"], task["probe"])]
                       for task in plan["tasks"]
                       if (task["seed"], task["probe"]) in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v25-opened-actor-route-native.v1",
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
