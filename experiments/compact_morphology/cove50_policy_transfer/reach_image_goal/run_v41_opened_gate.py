"""Staged native retention gate after the V41 opened screen."""

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
    run_v41_opened_screen as screen,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v41"
PLAN = OUT / "opened-gate-plan.json"
DEST = OUT / "opened-native"
STAGE_SIZE = 6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    screen.configure()
    previous_path = BASE / "v40/opened-gate-plan.json"
    previous = json.loads(previous_path.read_text())
    screened_path = OUT / "opened-screen/matrix.json"
    screened = json.loads(screened_path.read_text())
    if not screened["complete"] or screened["case_count"] != 17 or screened["pass_count"] != 17:
        raise ValueError("V41 17-case opened screen must pass before retention")
    if screened["plan_sha256"] != sha(screen.PLAN):
        raise ValueError("V41 screen plan identity mismatch")
    if len(previous["cases"]) != 75:
        raise ValueError("Expected 75 older/recent starts")
    screened_cases = json.loads(screen.PLAN.read_text())["cases"]
    by_seed = {case["seed"]: case for case in previous["cases"]}
    for case in screened_cases:
        old = by_seed[case["seed"]]
        for key in ("rail_v", "delay_ms", "toy_offset_x_m", "toy_offset_y_m",
                    "gaze_head_yaw_rad"):
            if case[key] != old[key]:
                raise ValueError(f"V41 screened case changed {key}: {case['seed']}")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(evaluator=sha(screen.EVALUATOR), selector=sha(screen.SELECTOR),
                   template=sha(core.TEMPLATE))
    if json.loads(screen.PLAN.read_text())["source_sha256"] != sources:
        raise ValueError("V41 screened source identities changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v41-opened-gate-plan.v1",
        "role": "opened_51_plus_24_retention_not_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v40_opened_plan_sha256": sha(previous_path),
        "v41_screen_plan_sha256": sha(screen.PLAN),
        "v41_screen_matrix_sha256": sha(screened_path),
        "core_runner_sha256": sha(Path(core.__file__)),
        "screen_runner_sha256": sha(Path(screen.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "stage_size": STAGE_SIZE,
        "reuse_identical_screened_native_visits": [case["seed"] for case in screened_cases],
        "case_count": 75, "cases": previous["cases"],
        "gate": "All 75 unique opened native starts must pass, including both V40 regressions. Reuse the 17 exactly matched native screen visits with identical checkpoint, evaluator, parameters and source hashes; run the other 58 starts. Stop at the first failed six-case stage. Only a complete gate permits the sealed disjoint fresh score.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V41 full plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def write_matrix(plan: dict, rows: dict[int, dict], stopped: bool) -> dict:
    ordered = [rows[c["seed"]] for c in plan["cases"] if c["seed"] in rows]
    matrix = {
        "schema": "robot-learning-lab.cove50-v41-opened-gate.v1",
        "role": "opened_51_plus_24_retention_not_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN), "complete": len(ordered) == 75,
        "stopped_after_failed_stage": stopped,
        "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
        "screen_reused_count": sum(r["seed"] in plan["reuse_identical_screened_native_visits"]
                                   for r in ordered),
        "rescues": [r["seed"] for r in ordered if r["pass"] and not r["baseline_pass"]],
        "regressions": [r["seed"] for r in ordered if not r["pass"] and r["baseline_pass"]],
        "cases": ordered,
    }
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
    core.DEST = DEST
    DEST.mkdir(parents=True, exist_ok=True)
    rows: dict[int, dict] = {}
    screened = json.loads((OUT / "opened-screen/matrix.json").read_text())
    screened_rows = {row["seed"]: row for row in screened["cases"]}
    for start in range(0, len(plan["cases"]), STAGE_SIZE):
        block = plan["cases"][start:start + STAGE_SIZE]
        pending = [case for case in block if case["seed"] not in screened_rows]
        for case in block:
            if case["seed"] in screened_rows:
                row = screened_rows[case["seed"]]
                rows[row["seed"]] = row
                print(json.dumps({"seed": row["seed"], "pass": row["pass"],
                                  "contact_s": row["contact_s"],
                                  "source": "identical_native_screen_visit"}), flush=True)
        write_matrix(plan, rows, False)
        if pending:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(screen.run_case, case, plan): case["seed"] for case in pending}
                for future in as_completed(futures):
                    row = future.result()
                    rows[row["seed"]] = row
                    write_matrix(plan, rows, False)
                    print(json.dumps({key: row[key] for key in
                                      ("seed", "pass", "contact_s", "faults")}), flush=True)
        if any(not rows[c["seed"]]["pass"] for c in block):
            matrix = write_matrix(plan, rows, True)
            print(f"stopped after failed stage: {matrix['pass_count']}/{matrix['case_count']}", flush=True)
            break


if __name__ == "__main__":
    main()
