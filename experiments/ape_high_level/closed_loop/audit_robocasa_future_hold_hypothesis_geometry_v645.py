"""Evaluator-only fixture check of V641's counterfactual hazard hypotheses."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_dynamic_suite_v440 import prop_fixture_gap
import run_robocasa_memory_future_hold_v641 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_person_joined_v497 as joined
import run_robocasa_memory_return_v599 as memory


SOURCE = Path(__file__)
OUT = trial.OUT
MODES = (("straight", None, 0.),
         ("left_after_0.6", .6, -np.pi / 2.),
         ("right_after_0.6", .6, np.pi / 2.),
         ("left_after_1.2", 1.2, -np.pi / 2.),
         ("right_after_1.2", 1.2, np.pi / 2.))


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    model = joined.scene_with_owner_and_prop(crossing.START)
    data = mujoco.MjData(model)
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    ape_geom = model.geom("audit_ape_geom").id
    rows = []
    for ep in report["episodes"]:
        summary = ep["memory_return"]
        if summary["run_label"] != "late_screen_held":
            continue
        decision = next(row for row in ep["trace"]
                        if row["stage"] == "memory_held_hazard_decision")
        track = decision["actor_track"]
        center = np.asarray(track["center_xy_m"])
        velocity = np.asarray(track["velocity_xy_m_s"])
        for name, turn_after, angle in MODES:
            c, s = np.cos(angle), np.sin(angle)
            turned_velocity = np.array([c * velocity[0] - s * velocity[1],
                                        s * velocity[0] + c * velocity[1]])
            minimum = (.3, None, None)
            for elapsed in np.arange(0., trial.POST_STOP_UNTIL_S -
                                      decision["time_s"] + .001, .02):
                age = decision["time_s"] + elapsed - track["last_seen_s"]
                if turn_after is None or age <= turn_after:
                    prop_xy = center + velocity * age
                else:
                    prop_xy = (center + velocity * turn_after +
                               turned_velocity * (age - turn_after))
                model.body_pos[prop_body, :2] = prop_xy
                mujoco.mj_forward(model, data)
                gap, name_of_geom = prop_fixture_gap(
                    model, data, prop_geom, ape_geom, prop_xy)
                if gap < minimum[0]:
                    minimum = (gap, elapsed, name_of_geom)
            rows.append({"source": summary["source"],
                         "forecast_hypothesis": name,
                         "minimum_scene_gap_m": round(float(minimum[0]), 5),
                         "minimum_at_elapsed_s": round(float(minimum[1]), 4),
                         "nearest_geom": minimum[2],
                         "clear_of_scene_5cm": minimum[0] >= .05})
    result = {
        "schema": "robot-learning-lab.robocasa-future-hypothesis-geometry-diagnostic.v1",
        "scope": "Posthoc evaluator-only geometry. This screen is not available to the actor and does not verify dynamic behavior or select a sensor.",
        "v641_report_sha256": memory.digest(report_path),
        "source_sha256": memory.digest(SOURCE),
        "sample_s": .02,
        "rows": rows,
    }
    (OUT / "hypothesis-geometry-diagnostic.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
