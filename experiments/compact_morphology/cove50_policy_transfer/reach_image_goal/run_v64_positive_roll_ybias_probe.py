"""Prescribe nearby image-goal offsets on the V62 positive-entry roll miss."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
    run_v41_opened_screen as screen,
    run_v57_contact_actor_screen as v57,
    run_v60_route_probe as v60,
    run_v61_roll_ybias_probe as probe,
)


ROOT = Path(__file__).resolve().parents[4]
OUT = v60.BASE / "v64"
PLAN = OUT / "positive-roll-prescribed-y-bias-plan.json"
DEST = OUT / "opened-positive-roll-y-bias"
CASE_ID = "positive-10-roll-high"
BIASES_M = (-.006, -.005, -.004, -.003, -.002, 0.0, .002)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    baseline_path = v60.DEST / "seed-11301-positive-10-roll-high.json"
    baseline = json.loads(baseline_path.read_text())
    if baseline["phases"]["reach"]["continuous_finger_contact_s"] >= 1:
        raise ValueError("Expected V60 positive-entry raised-roll contact miss")
    case = next(row for row in json.loads(v60.PLAN.read_text())["cases"]
                if row["id"] == CASE_ID)
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(v57.ACTOR), evaluator=sha(probe.EVALUATOR),
                   selector=sha(probe.SELECTOR), template=sha(screen.core.TEMPLATE),
                   prescribed_runner=sha(Path(probe.__file__)))
    plan = {
        "schema": "robot-learning-lab.cove50-v64-positive-roll-y-bias-plan.v1",
        "role": "opened_experimenter_prescribed_goal_offset_search_not_controller",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources, "runner_sha256": sha(Path(__file__)),
        "v60_plan_sha256": sha(v60.PLAN),
        "v60_failed_report_sha256": sha(baseline_path),
        "case_count": len(BIASES_M),
        "cases": [{**case, "diagnostic_y_bias_m": bias} for bias in BIASES_M],
        "gate": "Evaluate seven exact raised-roll positive-entry native visits with all support, rated effort, >=1 s fingertip contact, floor, self-contact, clipping and <=10 mm RGB gates. Seek at least one full-gate contact teacher with >=1.2 s margin. Prescribed offsets are not an automatic controller, fresh score or new-body qualification.",
        "design": "An early V62 native replay showed that the -4 mm front-route rule raises this visit's contact from the V60 0.60 s baseline to 0.92 s but does not pass. Hold seed, toy, motor, gaze, route, V57 arm and observed RGB entry fixed, sweeping -6, -5, -4, -3, -2, 0 and +2 mm body-Y offsets. Simulator toy truth remains scoring-only.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V64 positive-roll plan differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 3:
        raise ValueError("Use one to three native evaluators alongside the V62 route screen")
    DEST.mkdir(parents=True, exist_ok=True)
    probe.DEST = DEST
    probe.CASE_ID = CASE_ID
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe.run_case, case, plan): case["diagnostic_y_bias_m"]
                   for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["diagnostic_y_bias_m"]] = row
            ordered = [rows[case["diagnostic_y_bias_m"]] for case in plan["cases"]
                       if case["diagnostic_y_bias_m"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v64-positive-roll-y-bias.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
                "teacher_found": any(row["pass"] and row["contact_s"] >= 1.2
                                     for row in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "entry_goal_error_mm_scoring_only",
                               "faults")}), flush=True)


if __name__ == "__main__":
    main()
