"""Development-only swept response envelope for V716's early range cue."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_layout2_routes_v409 import clearance, sha256
from held_pose_screen import along_path
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_layout2_owner_inference_monitor_v715 as monitor
import run_robocasa_layout2_owner_inference_trigger_v716 as trial
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side
import run_robocasa_moving_prop_v423 as moving


SOURCE = Path(__file__)
OUT = trial.OUT / "escape-envelope-v720.json"
DECLARATION = trial.OUT / "escape-envelope-v720-declaration.json"
MAP = side.ROOT / "artifacts/ape-robocasa-layout2-joint-station-v712/actor-terminal-grid.npz"
GOAL = np.array([1.1, -3.95])
SPEEDS_M_S = (.145, .2, .25, .3, .4)
DISPATCH_DELAYS_S = (0., .1, .2, .4)
DT = .02


def scene():
    model = person.scene_with_person(side.START, layout_id=2, style_id=1,
                                     clutter_mode=0)
    torso = interaction.TORSO_XY + [-.3, 0.]
    hand = interaction.SHOW_HAND + [-.3, 0., 0.]
    person.set_owner_pose(model, torso, hand, show_can=True)
    direction = (hand[:2] - torso) / np.linalg.norm(hand[:2] - torso)
    model.body_pos[model.body("shown_can").id, :2] = hand[:2] + .14 * direction
    return model, mujoco.MjData(model)


def main():
    trial_report = json.loads((trial.OUT / "report.json").read_text())
    episode = trial_report["episode"]
    tracker = json.loads((trial.OUT / "tracker-replay-v719.json").read_text())
    dispatch_cue_s = tracker["cases"]["moving"]["first_development_interrupt_s"]
    assert dispatch_cue_s is not None
    declaration = {"schema": "robot-learning-lab.layout2-inference-escape-envelope-declaration.v1",
                   "trial_report_sha256": sha256(trial.OUT / "report.json"),
                   "tracker_replay_sha256": sha256(trial.OUT / "tracker-replay-v719.json"),
                   "actor_map_source_sha256": sha256(MAP),
                   "cue_time_s": dispatch_cue_s,
                   "candidate_goal_xy_m": GOAL.tolist(),
                   "candidate_goal_selection": "development choice after V716: southern known-free corridor, not an actor-selected or held-out refuge",
                   "speeds_m_s": SPEEDS_M_S,
                   "dispatch_delays_s": DISPATCH_DELAYS_S,
                   "route_guard": "20ms raw 30cm nearest-range check; freeze at first stop",
                   "evaluation": "full 23s 20ms MuJoCo scene gap, 5cm threshold, owner at shown pose, source-body cylinder; evaluator truth never commands move",
                   "source_sha256": sha256(SOURCE)}
    DECLARATION.write_text(json.dumps(declaration, indent=2) + "\n")
    with np.load(MAP) as saved:
        grid = route.SensorGrid()
        grid.free[:] = saved["free"]
        grid.occupied[:] = saved["occupied"]
    estimated = np.asarray(episode["final_estimated_xy_m"])
    actual = np.asarray(episode["final_actual_xy_m"])
    old_margin = route.MARGIN_M
    route.MARGIN_M = .15
    try:
        requested, plan_status, cell_path = grid.plan(estimated, GOAL)
    finally:
        route.MARGIN_M = old_margin
    assert plan_status == "known_goal" and requested is not None
    estimated_path = np.asarray([estimated] + [grid.xy(cell) for cell in cell_path[1:]])
    actual_path = actual + (estimated_path - estimated) * (1. + .30 / 5.89)
    path_length = float(np.sum(np.linalg.norm(np.diff(actual_path, axis=0), axis=1)))
    rows = []
    for prop_profile in ("stationary", "moving"):
        for speed in SPEEDS_M_S:
            for delay in DISPATCH_DELAYS_S:
                model, data = scene()
                ape_body = model.body("audit_ape").id
                ape_geom = model.geom("audit_ape_geom").id
                prop_body = model.body("route_obstacle").id
                stopped_time = None
                covered = 0.
                minimum = {"gap_m": float("inf"), "time_s": None, "nearest_geom": None}
                first_below_5cm = None
                end_xy = actual.copy()
                for t in np.arange(0., episode["hold_until_s"] + DT / 2., DT):
                    if prop_profile == "stationary":
                        prop = side.START
                    else:
                        fraction = np.clip((t - trial_report["capture_trigger_time_s"] - .15) /
                                           2.5, 0., 1.)
                        prop = side.START + fraction * (monitor.END - side.START)
                    model.body_pos[prop_body, :2] = prop
                    if (t >= dispatch_cue_s + delay and stopped_time is None and
                            covered < path_length):
                        mujoco.mj_forward(model, data)
                        raw = moving.raw_nearest_range(model, data, end_xy)
                        if raw < moving.GUARD_CENTER_RANGE_M:
                            stopped_time = round(float(t), 4)
                        else:
                            covered = min(path_length, covered + speed * DT)
                            end_xy = along_path(actual_path, covered)
                    gap, nearest = clearance(model, data, ape_body, ape_geom, end_xy)
                    if gap < minimum["gap_m"]:
                        minimum = {"gap_m": gap, "time_s": round(float(t), 4),
                                   "nearest_geom": nearest}
                    if gap < .05 and first_below_5cm is None:
                        first_below_5cm = round(float(t), 4)
                rows.append({"prop_profile": prop_profile,
                             "speed_m_s": speed, "dispatch_delay_s": delay,
                             "first_raw_guard_stop_s": stopped_time,
                             "path_completed": covered >= path_length - 1.e-6,
                             "end_xy_m": end_xy.tolist(),
                             "minimum_scene_gap": minimum,
                             "first_below_5cm_s": first_below_5cm})
    result = {"schema": "robot-learning-lab.layout2-inference-escape-envelope.v1",
              "declaration_sha256": sha256(DECLARATION),
              "source_sha256": sha256(SOURCE),
              "actor_map_plan_status": plan_status,
              "actor_map_path_xy_m": estimated_path.tolist(),
              "modeled_actual_path_xy_m": actual_path.tolist(),
              "modeled_path_length_m": path_length,
              "development_cue_time_s": dispatch_cue_s,
              "not_a_live_actor_policy": True,
              "rows": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print({profile: [(row["speed_m_s"], row["dispatch_delay_s"],
                      round(row["minimum_scene_gap"]["gap_m"], 3),
                      row["path_completed"])
                     for row in rows if row["prop_profile"] == profile and
                     row["first_below_5cm_s"] is None]
           for profile in ("stationary", "moving")}, flush=True)


if __name__ == "__main__":
    main()
