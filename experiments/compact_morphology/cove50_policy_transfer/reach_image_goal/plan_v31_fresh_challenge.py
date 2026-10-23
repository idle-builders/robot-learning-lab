"""Freeze disjoint V31 toy and motor conditions before opened gate outcome."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
)

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v31"
PLAN = OUT / "fresh-challenge-plan.json"
OPENED = OUT / "opened-gate-plan.json"
PREVIOUS = BASE / "v21/fresh-challenge-plan.json"
EVALUATOR = Path(__file__).with_name("evaluate_template_goal_v31.py")
TEMPLATE = Path(__file__).with_name("template_virtual_rgb_goal_v30.py")
RUNNER = Path(__file__).with_name("run_v31_fresh_challenge.py")
STRATA = {
    "positive_x": ((.007, .003), (.010, .005), (.006, .007), (.009, .009),
                   (.004, .006), (.011, .004), (.008, .010), (.005, .008)),
    "high_y": ((-.002, .010), (.002, .013), (.004, .010), (-.004, .011),
               (.001, .014), (.005, .012), (-.001, .013), (.003, .012)),
    "negative_corner": ((-.014, -.006), (-.010, -.010), (-.012, -.004), (-.008, -.007),
                        (-.015, -.003), (-.009, -.011), (-.013, -.008), (-.011, -.002)),
    "ordinary": ((-.004, .001), (-.001, -.004), (.003, .002), (-.007, -.002),
                 (.001, -.004), (-.002, .006), (-.006, .004), (.004, -.001)),
}
RAIL_PATTERN = (10, 11, 12, 11, 12, 10, 12, 11)
DELAY_PATTERN = (0, 5, 10, 10, 5, 0, 5, 10)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    opened = json.loads(OPENED.read_text())
    previous = json.loads(PREVIOUS.read_text())
    pointer = json.loads(v16.SOURCES["pointer"].read_text())
    if (len(opened["cases"]) != 24 or
            opened["role"] != "automatic_opened_template_gaze_height_router_gate" or
            pointer["status"] != "rejected_no_selected_mechanical_design"):
        raise ValueError("V31 fresh plan requires frozen opened gate on historical body")
    cases = []
    for stratum, offsets in STRATA.items():
        for index, (toy_x, toy_y) in enumerate(offsets):
            cases.append({"seed": 10701 + len(cases),
                          "intended_stratum": stratum,
                          "rail_v": RAIL_PATTERN[index],
                          "delay_ms": DELAY_PATTERN[index],
                          "toy_offset_x_m": toy_x, "toy_offset_y_m": toy_y,
                          "gaze_head_yaw_rad": -1.5})
    prior = {case["seed"] for case in opened["cases"]} | {case["seed"] for case in previous["cases"]}
    if (len(cases) != 32 or len({case["seed"] for case in cases}) != 32 or
            prior & {case["seed"] for case in cases} or
            {v: sum(case["rail_v"] == v for case in cases) for v in (10, 11, 12)} !=
            {10: 8, 11: 12, 12: 12} or
            {v: sum(case["delay_ms"] == v for case in cases) for v in (0, 5, 10)} !=
            {0: 8, 5: 12, 10: 12}):
        raise ValueError("Fresh seed, toy, supply, or delay coverage changed")
    sources = {key: sha(path) for key, path in v16.SOURCES.items() if key != "evaluator"}
    sources.update({"evaluator": sha(EVALUATOR), "template": sha(TEMPLATE)})
    plan = {"schema": "robot-learning-lab.cove50-v31-disjoint-fresh-challenge-plan.v1",
            "role": "disjoint_native_score_if_opened_gate_passes",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "opened_plan_sha256": sha(OPENED),
            "previous_fresh_plan_sha256": sha(PREVIOUS),
            "source_sha256": sources,
            "planner_sha256": sha(Path(__file__)), "runner_sha256": sha(RUNNER),
            "case_count": len(cases), "cases": cases,
            "promotion_gate": "All 32 new starts pass complete contact, support, effort, floor, clipping and image gates; at least four V10-height and four scoped-Y branch selections actually occur. One miss rejects historical-body promotion. No future-body qualification.",
            "qualification": "New 107xx seeds and new toy offsets; balanced 10/11/12 V and 0/5/10 ms delay across positive-X, high-Y, negative-corner and ordinary regions. Exact synthetic camera/robot/cube model on rejected Direct Frame."}
    expected = json.dumps(plan, indent=2) + "\n"
    if PLAN.exists() and PLAN.read_text() != expected:
        raise ValueError("Frozen V31 fresh plan differs from current sources")
    if not PLAN.exists():
        PLAN.write_text(expected)
    print(f"frozen {PLAN} sha256={sha(PLAN)} cases={len(cases)}")


if __name__ == "__main__":
    main()
