"""Freeze and score one disjoint v16 native RGB reach challenge."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import run_v16_opened_matrix as opened


V16 = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v16"
DEST = V16 / "fresh-reserved"

# Design strata are intended placements, not controller selections. The selected
# branch is measured from each native entry image after the score is opened.
OFFSETS = [
    ("deep_corner", -.0105, -.0045), ("deep_corner", -.0110, -.0055),
    ("deep_corner", -.0115, -.0065), ("deep_corner", -.0120, -.0045),
    ("deep_corner", -.0100, -.0060), ("deep_corner", -.0125, -.0050),
    ("deep_corner", -.0110, -.0070), ("deep_corner", -.0120, -.0065),
    ("deep_corner", -.0105, -.0075),
    ("goal_bias_region", -.0035, -.0060), ("goal_bias_region", -.0040, -.0055),
    ("goal_bias_region", -.0045, -.0065), ("goal_bias_region", -.0030, -.0050),
    ("goal_bias_region", -.0050, -.0060), ("goal_bias_region", -.0040, -.0070),
    ("goal_bias_region", -.0035, -.0045),
    ("specialist_region", .0025, .0045), ("specialist_region", .0030, .0050),
    ("specialist_region", .0035, .0055), ("specialist_region", .0020, .0060),
    ("specialist_region", .0040, .0040), ("specialist_region", .0030, .0065),
    ("specialist_region", .0025, .0055),
    ("live_rgb_region", -.0015, -.0040), ("live_rgb_region", -.0025, .0000),
    ("live_rgb_region", .0010, -.0050), ("live_rgb_region", -.0030, .0010),
    ("live_rgb_region", .0020, -.0040), ("live_rgb_region", -.0020, -.0020),
    ("broad_control", -.0070, -.0070), ("broad_control", -.0060, .0060),
    ("broad_control", .0060, -.0060), ("broad_control", .0060, .0060),
    ("broad_control", .0000, -.0060), ("broad_control", -.0020, .0040),
    ("broad_control", .0050, .0000),
]


def frozen_plan() -> dict:
    gate_path = V16 / "opened-native/matrix.json"
    gate = json.loads(gate_path.read_text())
    opened_path = V16 / "opened-native-plan.json"
    if (gate["plan_sha256"] != opened.sha(opened_path) or not gate["complete"] or
            gate["case_count"] != 30 or gate["pass_count"] != 30):
        raise ValueError("Fresh v16 score requires complete 30/30 opened gate")
    if len(OFFSETS) != 36:
        raise ValueError("Expected thirty-six disjoint fresh placements")
    cases = [{"origin": "v16_fresh_reserved", "design_stratum": stratum,
              "seed": 10401 + i, "rail_v": 11 if i % 2 == 0 else 12,
              "delay_ms": (0, 5, 10)[i % 3], "toy_offset_x_m": x,
              "toy_offset_y_m": y, "gaze_head_yaw_rad": -1.5}
             for i, (stratum, x, y) in enumerate(OFFSETS)]
    return {
        "schema": "robot-learning-lab.cove50-v16-fresh-native-plan.v1",
        "role": "single_disjoint_fresh_score_on_historical_rejected_body",
        "opened_gate_sha256": opened.sha(gate_path),
        "opened_plan_sha256": opened.sha(opened_path),
        "source_sha256": {name: opened.sha(path) for name, path in opened.SOURCES.items()},
        "opened_runner_sha256": opened.sha(Path(opened.__file__)),
        "fresh_runner_sha256": opened.sha(Path(__file__)),
        "design": "Nine deep corners, seven goal-bias placements, seven specialist placements, six live-RGB placements and seven broad controls; 18 each at 11/12 V and twelve each at 0/5/10 ms delay. Branch exposure is measured after replay, not guaranteed by placement label.",
        "promotion_gate": "36/36 full native passes, with at least two actual selections each of the height, bias, specialist and live branches; otherwise retain v14 as current scored controller and treat v16 as an incomplete candidate.",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Use one to three independent native evaluators")
    plan = frozen_plan()
    DEST.mkdir(parents=True, exist_ok=True)
    plan_path = V16 / "fresh-reserved-plan.json"
    planned = (json.dumps(plan, indent=2) + "\n").encode()
    if plan_path.exists():
        if plan_path.read_bytes() != planned:
            raise ValueError("Frozen fresh plan differs from source or opened gate")
    else:
        plan_path.write_bytes(planned)
    if args.freeze_only:
        print(f"frozen {plan_path} sha256={opened.sha(plan_path)}")
        return
    opened.DEST = DEST
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(opened.run_case, case, plan): case for case in plan["cases"]}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            row["design_stratum"] = case["design_stratum"]
            rows[case["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v16-fresh-native.v1",
                "role": "single_disjoint_fresh_score_on_historical_rejected_body",
                "plan_sha256": opened.sha(plan_path),
                "complete": len(ordered) == 36,
                "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
