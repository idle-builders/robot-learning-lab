"""Join complete V127 native pairs to the prior observed-RGB training data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    build_v118_observable_pairs as previous,
    run_v127_challenging_pair_data as challenge,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OLD = BASE / "v118/observable-pairs.json"
PLAN = BASE / "v127/challenging-paired-development-plan.json"
MATRIX = BASE / "v127/native-challenging-pairs/matrix.json"
RESERVE = BASE / "v128/disjoint-contact-score-plan.json"
OUT = BASE / "v129/challenging-observable-pairs.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    old = json.loads(OLD.read_text())
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    reserved = json.loads(RESERVE.read_text())
    if (not old["complete"] or len(old["pairs"]) != 65 or
            not matrix["complete"] or matrix["case_count"] != 24 or
            matrix["plan_sha256"] != sha(PLAN) or
            matrix["reserved_plan_sha256"] != sha(RESERVE) or
            reserved["status"] != "sealed_no_contact_visits" or
            len(reserved["cases"]) != 24):
        raise ValueError("Complete source pairs and sealed disjoint reserve required")
    expected = {row["id"]: row for row in plan["cases"]}
    if {row["id"] for row in matrix["cases"]} != set(expected):
        raise ValueError("V127 native matrix differs from the frozen plan")
    old_seeds = {row["seed_audit_only"] for row in old["pairs"]}
    development_seeds = {row["seed"] for row in plan["cases"]}
    reserve_seeds = {row["seed"] for row in reserved["cases"]}
    if (old_seeds & development_seeds or old_seeds & reserve_seeds or
            development_seeds & reserve_seeds or
            {challenge.physical_key(row) for row in plan["cases"]} &
            {challenge.physical_key(row) for row in reserved["cases"]}):
        raise ValueError("Development or reserve conditions overlap")
    pairs = list(old["pairs"])
    for row in matrix["cases"]:
        case = expected[row["id"]]
        if (row["seed"] != case["seed"] or
                row["route_variant"] != case["route_variant"] or
                row["rail_v"] != case["rail_v"] or
                row["delay_ms"] != case["delay_ms"] or
                row["toy_offset_x_m"] != case["toy_offset_x_m"] or
                row["toy_offset_y_m"] != case["toy_offset_y_m"]):
            raise ValueError(f"Native pair differs from frozen case: {row['id']}")
        pair = previous.make_pair(
            "V127_new_seed_failure_neighborhood", row, row,
            row["baseline_result"], row["shifted_result"],
            row["baseline_contact_s"], row["shifted_contact_s"],
            row["baseline_pass"], row["shifted_pass"])
        pair["condition_audit_only"] = {
            "anchor_opened_failure_seed": case["anchor_opened_failure_seed"],
            "toy_offset_xy_m": [case["toy_offset_x_m"], case["toy_offset_y_m"]],
            "route_variant": case["route_variant"],
        }
        pairs.append(pair)
    counts = {
        "pair_count": len(pairs),
        "new_pair_count": matrix["case_count"],
        "new_baseline_pass_count": matrix["baseline_pass_count"],
        "new_shifted_pass_count": matrix["shifted_pass_count"],
        "new_rescue_count": matrix["rescue_count"],
        "new_regression_count": matrix["regression_count"],
        "full_gate_rescue_count": sum(not row["baseline_pass"] and row["shifted_pass"]
                                      for row in pairs),
        "full_gate_regression_count": sum(row["baseline_pass"] and not row["shifted_pass"]
                                          for row in pairs),
    }
    dataset = {
        "schema": "robot-learning-lab.cove50-v129-challenging-observable-pairs.v1",
        "role": "complete_opened_training_only_no_sealed_contact_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "complete": True,
        "source_sha256": {"builder": sha(Path(__file__)), "old_dataset": sha(OLD),
                          "v127_plan": sha(PLAN), "v127_matrix": sha(MATRIX),
                          "v128_sealed_reserve": sha(RESERVE)},
        "feature_contract": old["feature_contract"],
        "counts": counts,
        "pairs": pairs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dataset, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Existing V129 training dataset differs from checked sources")
    OUT.write_text(encoded)
    print(f"{OUT} sha256={sha(OUT)} counts={counts}")


if __name__ == "__main__":
    main()
