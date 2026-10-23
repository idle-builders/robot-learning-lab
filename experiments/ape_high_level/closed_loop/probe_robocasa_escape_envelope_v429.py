"""Idealized straight-line escape envelope after the V423 proximity stop.

This is a counterfactual feasibility bound, not an actor policy or proof a
walking ape can reverse at these speeds. It tests the complete prop trajectory
and all fixed kitchen geometry with an evaluator-only 50 cm cylinder.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_layout2_routes_v409 import evaluator_scene, sha256
from run_robocasa_moving_prop_v423 import MOVE_END_S, obstacle_xy


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-escape-envelope-v429"
BASE = ROOT / "artifacts/ape-robocasa-moving-prop-v423/report.json"
STEP_S = .01
REQUIRED_GAP_M = .05


def simulate(start_xy: np.ndarray, start_time: float, speed_m_s: float,
             direction_xy: np.ndarray | None = None) -> dict:
    model, data, ape_body, ape_geom = evaluator_scene([4.95, -3.34])
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    times = np.arange(start_time, MOVE_END_S + STEP_S / 2, STEP_S)
    times = np.unique(np.r_[times, MOVE_END_S])
    direction = np.asarray([-1., 0.] if direction_xy is None else direction_xy, dtype=float)
    direction /= np.linalg.norm(direction)
    trajectory = []
    for time_s in times:
        # An ideal immediate, straight escape; -x is V429's reverse case.
        ape_xy = start_xy + direction * speed_m_s * (time_s - start_time)
        prop_xy = obstacle_xy(float(time_s))
        model.body_pos[ape_body, :2] = ape_xy
        model.body_pos[prop_body, :2] = prop_xy
        mujoco.mj_forward(model, data)
        prop_gap = float(mujoco.mj_geomDistance(model, data, ape_geom, prop_geom, .6, None))
        fixed_gap = .2
        fixed_geom = None
        for other in range(model.ngeom):
            if other in (ape_geom, prop_geom) or model.geom(other).name.startswith("floor_"):
                continue
            if np.linalg.norm(data.geom_xpos[other, :2] - ape_xy) > (
                    model.geom_rbound[other] + model.geom_rbound[ape_geom] + .2):
                continue
            gap = float(mujoco.mj_geomDistance(model, data, ape_geom, other, .2, None))
            if gap < fixed_gap:
                fixed_gap, fixed_geom = gap, model.geom(other).name
        trajectory.append({"time_s": round(float(time_s), 4),
                           "ape_xy_m": ape_xy.tolist(), "prop_xy_m": prop_xy.tolist(),
                           "prop_signed_gap_m": prop_gap,
                           "fixed_signed_gap_capped_m": fixed_gap,
                           "nearest_fixture": fixed_geom})
    return {"ideal_reverse_speed_m_s": speed_m_s,
            "direction_xy": direction.tolist(),
            "minimum_prop_gap_m": min(row["prop_signed_gap_m"] for row in trajectory),
            "minimum_fixed_gap_capped_m": min(row["fixed_signed_gap_capped_m"] for row in trajectory),
            "prop_margin_pass": all(row["prop_signed_gap_m"] >= REQUIRED_GAP_M for row in trajectory),
            "fixed_margin_pass": all(row["fixed_signed_gap_capped_m"] >= REQUIRED_GAP_M for row in trajectory),
            "trace": trajectory}


def main() -> None:
    base = json.loads(BASE.read_text())
    stop = base["episodes"][1]
    start = np.asarray(stop["end_true_xy_m"], dtype=float)
    start_time = float(stop["end_time_s"])
    speeds = [0., .25, .5, .6, .65, .7, .75, 1.]
    cases = [simulate(start, start_time, speed) for speed in speeds]
    report = {"source_sha256": sha256(Path(__file__)),
              "base_report_sha256": sha256(BASE),
              "start_xy_m": start.tolist(), "start_time_s": start_time,
              "scope": "ideal immediate straight -x reverse, 10ms geometry replay, no acceleration/braking/skill or actor access to ground truth",
              "required_gap_m": REQUIRED_GAP_M, "cases": cases}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for case in cases:
        print(case["ideal_reverse_speed_m_s"],
              round(case["minimum_prop_gap_m"], 4),
              round(case["minimum_fixed_gap_capped_m"], 4),
              case["prop_margin_pass"], case["fixed_margin_pass"], flush=True)


if __name__ == "__main__":
    main()
