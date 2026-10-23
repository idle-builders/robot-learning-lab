"""Stage the complete opened V45 retention set before another fresh score."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v41_fresh_challenge as fresh,
    run_v41_opened_gate as old,
    run_v45_combined_screen as screen,
)

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v45"
PLAN = OUT / "opened-retention-plan.json"
DEST = OUT / "opened-retention"
STAGE_SIZE = 6
CASE_FIELDS = ("rail_v", "delay_ms", "toy_offset_x_m", "toy_offset_y_m",
               "gaze_head_yaw_rad", "pitch_offset_rad", "roll_clearance_rad")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    screened_path = screen.DEST / "matrix.json"
    screened = json.loads(screened_path.read_text())
    if (not screened["complete"] or screened["case_count"] != 27 or
            screened["pass_count"] != 27 or screened["plan_sha256"] != sha(screen.PLAN)):
        raise ValueError("V45 27-case development screen must pass first")
    screen_plan = json.loads(screen.PLAN.read_text())
    if screen_plan["source_sha256"]["runner"] != sha(Path(screen.__file__)):
        raise ValueError("V45 screen runner changed")
    opened_plan = json.loads(old.PLAN.read_text())
    fresh_plan = json.loads(fresh.PLAN.read_text())
    if len(opened_plan["cases"]) != 75 or len(fresh_plan["cases"]) != 32:
        raise ValueError("Expected 75 older and 32 formerly fresh starts")
    opened_rows = {r["seed"]: r for r in json.loads((old.DEST / "matrix.json").read_text())["cases"]}
    fresh_rows = {r["seed"]: r for r in json.loads((fresh.DEST / "matrix.json").read_text())["cases"]}
    cases = []
    for source, role, rows in ((fresh_plan["cases"], "reopened_v41_fresh", fresh_rows),
                               (opened_plan["cases"], "older_retention", opened_rows)):
        for case in source:
            cases.append({**case, "pitch_offset_rad": .105, "roll_clearance_rad": -.25,
                          "baseline_v41_contact_s": rows[case["seed"]]["contact_s"],
                          "development_role": role})
    cases += screen.variance_cases()
    if len(cases) != 119 or len({c["seed"] for c in cases}) != 119:
        raise ValueError("Expected 119 unique opened/development starts")
    screened_cases = {c["seed"]: c for c in screen_plan["cases"]}
    for case in cases:
        prior = screened_cases.get(case["seed"])
        if prior and any(case[key] != prior[key] for key in CASE_FIELDS):
            raise ValueError(f"Screen and retention case differ: {case['seed']}")
    if len(screened_cases) != 27 or not set(screened_cases) <= {c["seed"] for c in cases}:
        raise ValueError("V45 screen cases are not all in retention")
    plan = {
        "schema": "robot-learning-lab.cove50-v45-opened-retention-plan.v1",
        "role": "full_opened_and_varied_native_retention_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": screen_plan["source_sha256"],
        "v45_screen_plan_sha256": sha(screen.PLAN),
        "v45_screen_matrix_sha256": sha(screened_path),
        "v41_opened_plan_sha256": sha(old.PLAN),
        "v41_fresh_plan_sha256": sha(fresh.PLAN),
        "runner_sha256": sha(Path(__file__)),
        "stage_size": STAGE_SIZE,
        "reuse_identical_screened_native_visits": list(screened_cases),
        "case_count": len(cases), "cases": cases,
        "gate": "All 119 unique native visits must pass: 75 old retention, 32 opened V41 fresh, and 12 new varied development starts. Reuse the 27 exactly matched V45 screen visits once after source and case identity checks. Stop after the first failed six-case stage. Only a complete all-pass retention gate can permit a new disjoint score. Route pitch/clearance remain at the V41 checkpoint's exact declared values.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V45 retention plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def write_matrix(plan: dict, rows: dict[int, dict], stopped: bool) -> dict:
    ordered = [rows[c["seed"]] for c in plan["cases"] if c["seed"] in rows]
    matrix = {
        "schema": "robot-learning-lab.cove50-v45-opened-retention.v1",
        "role": "full_opened_and_varied_native_retention_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN), "complete": len(ordered) == 119,
        "stopped_after_failed_stage": stopped,
        "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
        "screen_reused_count": sum(r["seed"] in plan["reuse_identical_screened_native_visits"]
                                   for r in ordered),
        "regressions_from_v41_pass": [r["seed"] for r in ordered
                                      if not r["pass"] and r["baseline_v41_contact_s"] is not None
                                      and r["baseline_v41_contact_s"] >= 1],
        "cases": ordered,
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= STAGE_SIZE:
        raise ValueError("Use one to six native evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    screened = json.loads((screen.DEST / "matrix.json").read_text())
    screened_rows = {r["seed"]: r for r in screened["cases"]}
    screen.DEST = DEST
    rows: dict[int, dict] = {}
    for start in range(0, len(plan["cases"]), STAGE_SIZE):
        block = plan["cases"][start:start + STAGE_SIZE]
        for case in block:
            if case["seed"] in screened_rows:
                rows[case["seed"]] = screened_rows[case["seed"]]
        pending = [case for case in block if case["seed"] not in screened_rows]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(screen.run_case, case, plan) for case in pending]
            for future in as_completed(futures):
                row = future.result()
                rows[row["seed"]] = row
                print(json.dumps({key: row[key] for key in
                                  ("seed", "pass", "contact_s", "faults")}), flush=True)
        failed = any(not rows[case["seed"]]["pass"] for case in block)
        matrix = write_matrix(plan, rows, failed)
        print(f"stage {start // STAGE_SIZE + 1}: {matrix['pass_count']}/{matrix['case_count']}", flush=True)
        if failed:
            break


if __name__ == "__main__":
    main()
