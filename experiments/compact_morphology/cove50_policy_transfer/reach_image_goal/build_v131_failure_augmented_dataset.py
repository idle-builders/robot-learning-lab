"""Add opened failure-seed perturbations without claiming seed independence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    build_v118_observable_pairs as previous,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
NEW_SEEDS = BASE / "v129/challenging-observable-pairs.json"
PLAN = BASE / "v130/opened-failure-perturbation-plan.json"
MATRIX = BASE / "v130/native-opened-failure-perturbations/matrix.json"
RESERVE = BASE / "v128/disjoint-contact-score-plan.json"
OUT = BASE / "v131/failure-augmented-observable-pairs.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    original = json.loads(NEW_SEEDS.read_text())
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    reserve = json.loads(RESERVE.read_text())
    if (not original["complete"] or len(original["pairs"]) != 89 or
            not matrix["complete"] or matrix["case_count"] != 24 or
            matrix["plan_sha256"] != sha(PLAN) or
            reserve["status"] != "sealed_no_contact_visits" or
            len(reserve["cases"]) != 24):
        raise ValueError("Complete source pairs and sealed V128 reserve required")
    planned = {row["id"]: row for row in plan["cases"]}
    if {row["id"] for row in matrix["cases"]} != set(planned):
        raise ValueError("V130 matrix differs from the frozen plan")
    reserve_seeds = {row["seed"] for row in reserve["cases"]}
    if ({row["seed_audit_only"] for row in original["pairs"]} |
            {row["seed"] for row in plan["cases"]}) & reserve_seeds:
        raise ValueError("Training seed overlaps independent reserve")
    pairs = list(original["pairs"])
    for row in matrix["cases"]:
        case = planned[row["id"]]
        if any(row[key] != case[key] for key in (
                "seed", "route_variant", "rail_v", "delay_ms",
                "toy_offset_x_m", "toy_offset_y_m", "gaze_head_yaw_rad")):
            raise ValueError(f"Native V130 row differs from plan: {row['id']}")
        pair = previous.make_pair(
            "V130_opened_failure_seed_physical_perturbation", row, row,
            row["baseline_result"], row["shifted_result"],
            row["baseline_contact_s"], row["shifted_contact_s"],
            row["baseline_pass"], row["shifted_pass"])
        pair["case_id_audit_only"] = row["id"]
        pair["condition_audit_only"] = {
            "anchor_opened_failure_seed": case["anchor_opened_failure_seed"],
            "toy_offset_xy_m": [case["toy_offset_x_m"], case["toy_offset_y_m"]],
            "route_variant": case["route_variant"],
        }
        pairs.append(pair)
    if len(pairs) != 113 or len({row["seed_audit_only"] for row in pairs}) != 89:
        raise ValueError("Expected 113 pairs in 89 seed groups")
    counts = {
        "pair_count": len(pairs), "distinct_seed_count": 89,
        "opened_reused_seed_pair_count": 24,
        "new_baseline_pass_count": matrix["baseline_pass_count"],
        "new_shifted_pass_count": matrix["shifted_pass_count"],
        "new_rescue_count": matrix["rescue_count"],
        "new_regression_count": matrix["regression_count"],
    }
    dataset = {
        "schema": "robot-learning-lab.cove50-v131-failure-augmented-pairs.v1",
        "role": "opened_seed_reuse_training_only_not_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "complete": True,
        "source_sha256": {"builder": sha(Path(__file__)),
                          "v129_dataset": sha(NEW_SEEDS),
                          "v130_plan": sha(PLAN), "v130_matrix": sha(MATRIX),
                          "v128_sealed_reserve": sha(RESERVE)},
        "feature_contract": original["feature_contract"],
        "crossfit_group_rule": ("When withholding a training example, withhold "
                                "all examples with its seed; the 24 V130 "
                                "physical variations reuse three opened V101 "
                                "seed identities and are not independent."),
        "counts": counts,
        "pairs": pairs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dataset, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Existing V131 dataset differs from checked sources")
    OUT.write_text(encoded)
    print(f"{OUT} sha256={sha(OUT)} counts={counts}")


if __name__ == "__main__":
    main()
