"""Predeclare disjoint v21 RGB reach challenge before opened gate outcome."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v21"
OPENED = OUT / "opened-gate-plan.json"
PLAN = OUT / "fresh-challenge-plan.json"
EVALUATOR = Path(__file__).with_name("evaluate_adaptive_goal_v21.py")
RUNNER = Path(__file__).with_name("run_v21_fresh_challenge.py")
STRATA = {
    "positive_x": ((.006, .004), (.008, .006), (.004, .008), (.009, .005),
                   (.005, .010), (.007, .007)),
    "high_y": ((-.001, .008), (.001, .011), (.003, .009), (-.003, .010),
               (.004, .011), (.0, .012)),
    "negative_corner": ((-.011, -.006), (-.009, -.008), (-.013, -.005),
                        (-.008, -.004), (-.012, -.009), (-.010, -.003)),
    "ordinary": ((-.005, .002), (-.002, -.003), (.001, .003),
                 (-.006, -.001), (.002, -.002), (-.003, .005)),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    opened = json.loads(OPENED.read_text())
    if (len(opened["cases"]) != 51 or
            opened["role"] != "opened_native_development_gate_not_fresh_score"):
        raise ValueError("Need frozen 51-case opened gate before fresh-plan freeze")
    pointer = json.loads(v16.SOURCES["pointer"].read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("This challenge is for the rejected historical body only")
    prior_seeds = {case["seed"] for case in opened["cases"]}
    cases = []
    for stratum, offsets in STRATA.items():
        for x, y in offsets:
            idx = len(cases)
            cases.append({
                "seed": 10601 + idx,
                "intended_stratum": stratum,
                "rail_v": 11 if idx % 2 == 0 else 12,
                "delay_ms": (0, 5, 10)[idx % 3],
                "toy_offset_x_m": x,
                "toy_offset_y_m": y,
                "gaze_head_yaw_rad": -1.5 if idx % 2 == 0 else -1.3,
            })
    if (len(cases) != 24 or len({case["seed"] for case in cases}) != 24 or
            prior_seeds.intersection(case["seed"] for case in cases) or
            {v: sum(case["rail_v"] == v for case in cases) for v in (11, 12)} !=
            {11: 12, 12: 12} or
            {d: sum(case["delay_ms"] == d for case in cases) for d in (0, 5, 10)} !=
            {0: 8, 5: 8, 10: 8}):
        raise ValueError("Fresh challenge coverage or motor balance changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v21-disjoint-fresh-challenge-plan.v1",
        "role": "disjoint_native_score_if_opened_gate_passes",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "opened_plan_sha256": sha(OPENED),
        "source_sha256": {name: sha(path) for name, path in v16.SOURCES.items()
                          if name != "evaluator"} | {"evaluator": sha(EVALUATOR)},
        "source_sha256_plan": sha(Path(__file__)),
        "runner_sha256": sha(RUNNER),
        "case_count": len(cases),
        "cases": cases,
        "promotion_gate": "All 24 disjoint native starts must pass the complete contact, support, effort, floor, clipping and RGB gates, with at least four actual automatic v21 scoped-Y selections. Any miss rejects promotion. No result qualifies a future selected body.",
        "qualification": "New 106xx seeds and toy offsets, balanced 11/12 V, 0/5/10 ms delay and two qualified slow gaze targets. Strata cover positive-X, high-Y, negative corner and ordinary starts; actual RGB entry, not intended stratum, chooses the branch.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    expected = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists():
        if PLAN.read_bytes() != expected:
            raise ValueError("Frozen v21 disjoint fresh plan differs from sources")
    else:
        PLAN.write_bytes(expected)
    print(f"frozen {PLAN} sha256={sha(PLAN)} cases={len(cases)}")


if __name__ == "__main__":
    main()
