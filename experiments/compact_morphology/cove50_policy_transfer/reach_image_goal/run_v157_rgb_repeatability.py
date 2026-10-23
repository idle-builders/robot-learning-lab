"""Freeze exact new-seed RGB repeats before V156 camera outcomes are visible.

One center placement per V156 seed family is repeated on the same native CUDA
runtime. These are correlated camera observations, never extra contact cases.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v98_entry_exposure as entry_runner,
    run_v156_independent_seed_entry_families as v156,
    v118_observable_selector as observations,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v157"
PLAN = OUT / "exact-rgb-repeatability-plan.json"
DEST = OUT / "native-entry-only"
MATRIX = DEST / "matrix.json"
V156_PLAN = BASE / "v156/independent-seed-entry-family-plan.json"
V156_MATRIX = BASE / "v156/native-entry-only/matrix.json"
V156_SUMMARY = BASE / "v156/source-checked-entry-family-summary.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict, *, frozen: bool = False) -> None:
    encoded = json.dumps(value, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists() and path.read_text() != encoded:
        raise ValueError(f"Frozen V157 plan changed: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def freeze() -> dict:
    if not PLAN.exists() and V156_MATRIX.exists():
        raise ValueError("Freeze repeat plan before seeing any V156 entry result")
    source = v156.freeze()
    reserve = json.loads(RESERVE.read_text())
    pointer = json.loads(POINTER.read_text())
    if (source["case_count"] != 168 or source["family_count"] != 24 or
            source["contains_contact_visits"] or
            source["source_sha256"]["runner"] != sha(Path(v156.__file__)) or
            source["source_sha256"]["entry_runner"] !=
            sha(Path(entry_runner.__file__)) or
            reserve["status"] != "sealed_no_contact_visits" or
            pointer["selected_design_id"] is not None):
        raise ValueError("V156 source, body pointer, or sealed reserve changed")
    centers = [case for case in source["cases"]
               if case["toy_delta_x_m"] == 0 and
               case["toy_delta_y_m"] == 0]
    if (len(centers) != 24 or
            len({case["family_id"] for case in centers}) != 24 or
            len({case["seed"] for case in centers}) != 24):
        raise ValueError("Expected one exact center per independent seed family")
    cases = [{**case, "id": f"repeat-{case['seed']}-center",
              "source_v156_entry_id": case["id"]} for case in centers]
    plan = {
        "schema": "robot-learning-lab.cove50-v157-exact-rgb-repeatability-plan.v1",
        "role": "same_seed_exact_camera_repeatability_diagnostic_not_contact_or_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**source["source_sha256"],
                          "runner": sha(Path(__file__)),
                          "v156_plan": sha(V156_PLAN),
                          "v156_runner": sha(Path(v156.__file__)),
                          "entry_runner": sha(Path(entry_runner.__file__)),
                          "untouched_score_reserve": sha(RESERVE)},
        "case_count": 24, "seed_family_count": 24,
        "contains_contact_visits": False,
        "prior_visibility": (
            "Frozen after the V152 seed24526 one-pixel RGB mismatch and its "
            "posthoc CUDA repeat, before any V156 native entry result. "
            "No V147 independent score visit has been opened."),
        "method": (
            "Repeat exactly one preselected zero-offset center toy placement "
            "from each of the 24 V156 new-seed families on the same CUDA "
            "runtime. Keep seed, motor voltage, delay, gaze, pitch, roll and "
            "toy XY identical. Compare the first used controlled RGB image, "
            "orange mask, numeric goal, route and camera pose to its original "
            "V156 entry. Do not choose a placement after seeing V156 routes, "
            "count repeats as independent seeds, or infer contact success."),
        "cases": cases,
    }
    write_json(PLAN, plan, frozen=True)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    row = entry_runner.run_case(case, plan)
    result = ROOT / row["result"]
    trace = result.with_suffix(".npz")
    report = json.loads(result.read_text())
    observed = observations.observed_features(report) if row["entry_available"] else None
    if row["entry_available"] and observed["selection_mode"] != row["selection_mode"]:
        raise ValueError(f"V157 observed route differs: {case['id']}")
    return {**row, "family_id": case["family_id"],
            "stratum": case["stratum"],
            "source_v156_entry_id": case["source_v156_entry_id"],
            "observed_features": observed,
            "result_sha256": sha(result), "trace_sha256": sha(trace)}


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"]
               if case["id"] in rows]
    value = {
        "schema": "robot-learning-lab.cove50-v157-exact-rgb-repeatability-matrix.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "case_count": len(ordered), "planned_count": 24,
        "complete": len(ordered) == 24,
        "observed_route_counts": dict(sorted(Counter(
            row["selection_mode"] for row in ordered).items())),
        "contains_contact_outcomes": False,
        "cases": ordered,
    }
    write_json(MATRIX, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError("Use one or two CUDA camera evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"V157 froze 24 exact new-seed RGB repeats; sha256={sha(PLAN)}")
        return
    v156_matrix = json.loads(V156_MATRIX.read_text())
    v156_summary = json.loads(V156_SUMMARY.read_text())
    refs = plan["source_sha256"]
    by_id = {row["id"]: row for row in v156_matrix["cases"]}
    if (refs["runner"] != sha(Path(__file__)) or
            refs["v156_plan"] != sha(V156_PLAN) or
            refs["v156_runner"] != sha(Path(v156.__file__)) or
            refs["entry_runner"] != sha(Path(entry_runner.__file__)) or
            refs["untouched_score_reserve"] != sha(RESERVE) or
            v156_matrix["plan_sha256"] != sha(V156_PLAN) or
            not v156_matrix["complete"] or v156_matrix["case_count"] != 168 or
            not v156_summary["complete"] or
            v156_summary["source_checked_entry_count"] != 168 or
            v156_summary["source_sha256"]["matrix"] != sha(V156_MATRIX) or
            any(case["source_v156_entry_id"] not in by_id for case in plan["cases"])):
        raise ValueError("V156 complete source-checked entries required before repeats")
    DEST.mkdir(parents=True, exist_ok=True)
    old = json.loads(MATRIX.read_text()) if MATRIX.exists() else None
    if old and old["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V157 matrix belongs to another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    entry_runner.DEST = DEST
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case
                   for case in plan["cases"] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V157 {row['id']}: {row['selection_mode']} "
                  f"{matrix['case_count']}/24 exact repeats", flush=True)


if __name__ == "__main__":
    main()
