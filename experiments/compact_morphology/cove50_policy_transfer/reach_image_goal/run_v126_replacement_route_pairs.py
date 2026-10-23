"""Opened route comparison after V122's replacement-host image-entry mismatch.

V122 remains failed under its original frozen tolerances. This diagnostic
uses the same five starts and source evaluators, with a predeclared 20 um
image-entry allowance that covers the first observed replacement-host gap.
It is not an independent score or a policy promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
    run_v122_near_route_pairs as prior,
)


ROOT = prior.ROOT
BASE = prior.BASE
OUT = BASE / "v126"
PLAN = OUT / "replacement-route-pairs-plan.json"
DEST = OUT / "native-replacement-route-pairs"
MATRIX = DEST / "matrix.json"
PARENT_SHA = "cf9fac8fbf677893ac580016388d173ab21c42d170520654b8ef67366b1e31a0"
TOLERANCES = {
    "rounded_entry_error_xy_m": 0.000020,
    "rounded_camera_world_xyz_m": 0.000020,
    "estimated_goal_world_xyz_m": 0.000020,
    "native_entry_qpos_max_abs": 0.000005,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def freeze() -> dict:
    parent = prior.freeze()
    if sha(prior.PLAN) != PARENT_SHA:
        raise ValueError("Original V122 plan changed")
    first = prior.DEST / "seed-19026-near-image-x-automatic.json"
    second = prior.DEST / "seed-19026-near-image-x-forced.json"
    if not first.is_file() or not second.is_file():
        raise ValueError("Need the stopped V122 first-pair evidence")
    plan = json.loads(json.dumps(parent))
    plan.update(
        schema="robot-learning-lab.cove50-v126-replacement-route-pairs-plan.v1",
        role="opened_post_observation_replacement_runtime_diagnostic_not_score",
        parent_v122_plan_sha256=PARENT_SHA,
        parent_v122_first_pair_reports={
            "automatic": {"path": str(first.relative_to(ROOT)), "sha256": sha(first)},
            "forced": {"path": str(second.relative_to(ROOT)), "sha256": sha(second)},
        },
        entry_tolerances=TOLERANCES,
        amendment_reason=(
            "V122 first pair on replacement RTX 4090 had native entry qpos gap "
            "1.967e-6 and identical rounded camera/pixel count, but automatic "
            "versus forced image error differed by up to 9 um and estimated "
            "goal by 10 um. This exceeded V122's 2/1 um image allowances; V122 "
            "remains failed. Freeze a 20 um image allowance before any V126 "
            "visit, retain the 5 um native-state and exact pixel requirements."
        ),
        decision_gate=(
            "Diagnostic only: compare all five paired contacts and report "
            "full-gate or margin losses, entry differences and near-boundary "
            "sensitivity. A gain here may guide new training data; it cannot "
            "repair V122, establish a sealed score or qualify the body."
        ),
    )
    plan["source_sha256"]["recovery_runner"] = sha(Path(__file__))
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("V126 diagnostic plan or source changed")
    PLAN.write_bytes(encoded)
    return plan


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    result = prior.write_matrix(plan, rows)
    result["schema"] = "robot-learning-lab.cove50-v126-replacement-route-pairs.v1"
    result["parent_v122_plan_sha256"] = PARENT_SHA
    result["entry_tolerances"] = TOLERANCES
    write_json(MATRIX, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    print(f"V126 diagnostic plan sha256={sha(PLAN)}", flush=True)
    if args.freeze_only:
        return
    prior.PLAN, prior.DEST, prior.MATRIX = PLAN, DEST, MATRIX
    DEST.mkdir(parents=True, exist_ok=True)
    old = json.loads(MATRIX.read_text()) if MATRIX.exists() else None
    if old and old["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V126 matrix belongs to another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    original = native.EVALUATOR
    for case in plan["cases"]:
        if case["id"] in rows:
            continue
        forced = case["diagnostic_force_specialist"]
        native.EVALUATOR = prior.FORCED_EVALUATOR if forced else original
        try:
            row = native.run_case(case, {"source_sha256": plan["source_sha256"][
                "forced" if forced else "baseline"]}, DEST)
        finally:
            native.EVALUATOR = original
        report = json.loads((ROOT / row["result"]).read_text())
        if bool(report.get("diagnostic_force_specialist", False)) != forced:
            raise ValueError(f"V126 route identity mismatch: {case['id']}")
        row["diagnostic_force_specialist"] = forced
        rows[case["id"]] = row
        result = write_matrix(plan, rows)
        print(f"V126 {case['id']}: {row['contact_s']:.2f} s, "
              f"{'pass' if row['pass'] else 'fail'}; {result['pair_count']} pairs", flush=True)
    result = write_matrix(plan, rows)
    print(f"V126 diagnostic {result['pair_count']}/{plan['pair_count']} pairs; "
          f"near gains {result['near_gain_half_second_count']}; "
          f"full regressions {result['full_gate_regression_count']}", flush=True)


if __name__ == "__main__":
    main()
