"""Freeze varied opened-case neighborhoods for the next RGB reach lesson.

These 105xx starts are development data derived from already-opened 104xx
failures and controls. They cannot serve as a disjoint promotion score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v18"
FRESH = BASE / "v16/fresh-reserved/matrix.json"
FRESH_PLAN = BASE / "v16/fresh-reserved-plan.json"
FORCED = BASE / "v17/posthoc-forced-branch/summary.json"
BIAS = BASE / "v17/posthoc-hard-goal-bias/summary.json"
POINTER = ROOT / "robot/ape-current-design.json"

FAILURE_CENTERS = {
    10401: "v16_v10_height_locked",
    10409: "v10_locked_plus_x2mm",
    10417: "v14_locked_minus_y2mm",
    10421: "v16_live",
    10422: "v16_bias_y_locked",
    10433: "v16_specialist_locked",
}
PASS_CONTROLS = (10403, 10404, 10410, 10418, 10419, 10423)
JITTER_BODY_XY_MM = ((-2, 0), (2, 0), (0, -2), (0, 2))
# The native RGB evaluator admits the two previously qualified slow-gaze
# targets only. Motor-start and toy variance still change the attained view.
HEAD_YAW_JITTER_RAD = (0.0, .2, 0.0, .2)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    fresh = json.loads(FRESH.read_text())
    if fresh["case_count"] != 36 or fresh["pass_count"] != 30 or not fresh["complete"]:
        raise ValueError("Expected sealed v16 fresh score")
    scored = {row["seed"]: row for row in fresh["cases"]}
    for seed in FAILURE_CENTERS:
        if scored[seed]["pass"]:
            raise ValueError(f"Failure center became a pass: {seed}")
    for seed in PASS_CONTROLS:
        if not scored[seed]["pass"]:
            raise ValueError(f"Pass control became a failure: {seed}")
    forced = json.loads(FORCED.read_text())
    bias = json.loads(BIAS.read_text())
    if (forced["rescued_seeds"] != [10401, 10421, 10422, 10433] or
            bias["rescued_miss_seeds"] != [10409, 10417]):
        raise ValueError("Opened teacher probes changed")
    pointer = json.loads(POINTER.read_text())
    if (pointer["status"] != "rejected_no_selected_mechanical_design" or
            pointer["selected_design_id"] is not None):
        raise ValueError("This curriculum is for the rejected historical body only")
    planned = {row["seed"]: row for row in json.loads(FRESH_PLAN.read_text())["cases"]}
    cases = []
    centers = tuple(FAILURE_CENTERS) + PASS_CONTROLS
    for origin_seed in centers:
        source = planned[origin_seed]
        for j, ((dx, dy), dh) in enumerate(zip(JITTER_BODY_XY_MM,
                                                 HEAD_YAW_JITTER_RAD)):
            idx = len(cases)
            cases.append({
                "origin": "derived_opened_v16_fresh_case",
                "role": "failure_neighborhood" if origin_seed in FAILURE_CENTERS
                        else "pass_retention_neighborhood",
                "origin_seed": origin_seed,
                "seed": 10501 + idx,
                "rail_v": 11 if idx % 2 == 0 else 12,
                "delay_ms": (0, 5, 10)[idx % 3],
                "toy_offset_x_m": round(source["toy_offset_x_m"] + dx / 1000, 6),
                "toy_offset_y_m": round(source["toy_offset_y_m"] + dy / 1000, 6),
                "gaze_head_yaw_rad": round(source["gaze_head_yaw_rad"] + dh, 4),
                "toy_jitter_body_xy_mm": [dx, dy],
                "assigned_teacher_probe": FAILURE_CENTERS.get(origin_seed),
            })
    if (len(cases) != 48 or len({row["seed"] for row in cases}) != 48 or
            {v: sum(row["rail_v"] == v for row in cases) for v in (11, 12)} !=
            {11: 24, 12: 24} or
            {d: sum(row["delay_ms"] == d for row in cases) for d in (0, 5, 10)} !=
            {0: 16, 5: 16, 10: 16}):
        raise ValueError("Variance balance changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v18-variance-development-plan.v1",
        "role": "opened_development_not_disjoint_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v16_fresh_matrix_sha256": sha(FRESH),
        "v16_fresh_plan_sha256": sha(FRESH_PLAN),
        "v17_forced_summary_sha256": sha(FORCED),
        "v17_goal_bias_summary_sha256": sha(BIAS),
        "pointer_sha256": sha(POINTER),
        "source_sha256": sha(Path(__file__)),
        "centers": list(centers),
        "failure_teacher_probes": FAILURE_CENTERS,
        "body_xy_jitter_mm": [list(xy) for xy in JITTER_BODY_XY_MM],
        "head_yaw_jitter_rad": HEAD_YAW_JITTER_RAD,
        "case_count": len(cases),
        "cases": cases,
        "qualification": "This varied neighborhood is derived from opened 104xx cases to collect native teacher/contact evidence. It cannot be used as a fresh promotion score. Automatic inference must use observable inputs and a later independently frozen disjoint set.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "development-variance-plan.json"
    expected = json.dumps(plan, indent=2).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != expected:
            raise ValueError("Frozen v18 development plan changed")
    else:
        path.write_bytes(expected)
    print(f"frozen {path} sha256={sha(path)} cases={len(cases)}")


if __name__ == "__main__":
    main()
