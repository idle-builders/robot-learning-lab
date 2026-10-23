"""V331 same-packet rear-shin range ablation for V330 in-bound hazards.

Five actor sensor layouts share exactly the same noisy/dropout captures. A
separate evaluator replay scores hold/left-pitch/right-pitch outcomes after
the actor has chosen; no scene label enters the alarm or action forecast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from direct_structural_rooms_v270 import selected_visual
from protective_action_gate import compare
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import (
    ALARM_S, END_S, FLEX_TIME_S, START_S, actor_alarm, actual_pose, estimated_pose,
)
from audit_rear_action_holdout_v330 import CASE_TAIL, set_scene


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v331"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
V330 = ROOT / "artifacts/ape-closed-loop-cove50-v330/rear-action-holdout-audit.json"
LAYOUTS = {
    "front_four": ("front",),
    "rear_two": ("left_rear", "right_rear"),
    "front_four_plus_left_rear": ("front", "left_rear"),
    "front_four_plus_right_rear": ("front", "right_rear"),
    "front_four_plus_both_rear": ("front", "left_rear", "right_rear"),
}
ARRAYS = {"front": "front_points_xyz_m",
          "left_rear": "left_rear_points_xyz_m",
          "right_rear": "right_rear_points_xyz_m"}


def evaluator_pitch(room: LateCorridorRoom, prior: dict,
                    base_y: float, side_name: str) -> dict:
    address = int(room.model.joint(f"{side_name}_shoulder_pitch").qposadr[0])
    side = 1 if prior["side_evaluator_only"] == "left" else -1
    closest = (float("inf"), "", "")
    for time_s in np.linspace(START_S, END_S, 41):
        set_scene(room, float(time_s), side=side,
                  x_offset_m=prior["prop_x_offset_m_evaluator_only"],
                  height_m=prior["prop_height_m_evaluator_only"],
                  speed_m_s=prior["prop_speed_m_s_evaluator_only"],
                  base_y=base_y)
        room.data.qpos[18:27] = 0.
        room.data.qpos[address] = -.8 * np.clip(
            (time_s - ALARM_S) / FLEX_TIME_S, 0., 1.)
        room._forward()
        closest = min(closest, room._current_obstacle_clearance())
    return {"minimum_scene_clearance_m": round(closest[0], 5),
            "closest_scene_geoms": [closest[1], closest[2]]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(V330.read_text())
    anchor_path = V323 / "seed170-present/seed170-present/episode-trace.json"
    anchor = json.loads(anchor_path.read_text())
    actual, estimated = actual_pose(anchor), estimated_pose(anchor)
    rows = []
    for earlier in prior["rows"]:
        if earlier["prop_speed_m_s_evaluator_only"] > .25:
            continue
        room = LateCorridorRoom(case=(earlier["seed"], *CASE_TAIL),
                                pose_source="command", branch=OUT)
        try:
            room.layout(earlier["seed"], absent=False, furnished=True)
            room.actual[:] = actual
            room.estimated[:] = estimated
            robot = RobotGeometry(room.model)
            captures = []
            for entry in earlier["actor_captures"]:
                path = ROOT / entry["path"]
                if sha(path) != entry["sha256"]:
                    raise AssertionError(f"Changed V330 actor capture: {path}")
                with np.load(path) as frame:
                    captures.append({name: frame[key].copy()
                                     for name, key in ARRAYS.items()} |
                                    {"actor_qpos": frame["actor_qpos"].copy()})
            truth = {"hold": earlier["evaluator_outcomes"]["hold"],
                     "left_shoulder_pitch": evaluator_pitch(
                         room, earlier, actual[1], "left"),
                     "right_shoulder_pitch": evaluator_pitch(
                         room, earlier, actual[1], "right")}
            layouts = {}
            for layout, components in LAYOUTS.items():
                first = np.concatenate([captures[0][name] for name in components])
                second = np.concatenate([captures[1][name] for name in components])
                alarm = actor_alarm(first, second, estimated)
                forecast = ({"status": "no_alarm", "chosen": "hold"}
                            if alarm["side"] is None else
                            compare(second, estimated,
                                    captures[1]["actor_qpos"],
                                    alarm["side"], robot))
                action = ("hold" if forecast["chosen"] == "hold" else
                          f"{alarm['side']}_shoulder_pitch")
                layouts[layout] = {"actor_alarm": alarm,
                                   "actor_action_forecast": forecast,
                                   "actor_chosen_action": action,
                                   "evaluator_minimum_scene_clearance_m": truth[action][
                                       "minimum_scene_clearance_m"]}
            rows.append({"case": earlier["case"],
                         "prop_speed_m_s_evaluator_only": earlier[
                             "prop_speed_m_s_evaluator_only"],
                         "evaluator_hold_clearance_m": truth["hold"][
                             "minimum_scene_clearance_m"],
                         "evaluator_counterfactual_pitch_clearance_m": {
                             "left": truth["left_shoulder_pitch"][
                                 "minimum_scene_clearance_m"],
                             "right": truth["right_shoulder_pitch"][
                                 "minimum_scene_clearance_m"]},
                         "layouts": layouts,
                         "v330_actor_capture_sha256": [e["sha256"] for e in
                                                       earlier["actor_captures"]]})
        finally:
            room.close()
    summary = {}
    for name in LAYOUTS:
        hazardous = [r for r in rows if r["evaluator_hold_clearance_m"] < .05]
        summary[name] = {
            "cases": len(rows), "hazardous_hold_cases": len(hazardous),
            "hazardous_detected": sum(r["layouts"][name]["actor_alarm"][
                "side"] is not None for r in hazardous),
            "hazardous_rescued_above_5cm": sum(r["layouts"][name][
                "evaluator_minimum_scene_clearance_m"] >= .05 for r in hazardous),
            "harmful_actions_from_safe_hold": sum(
                r["evaluator_hold_clearance_m"] >= .05 and
                r["layouts"][name]["evaluator_minimum_scene_clearance_m"] < .05
                for r in rows),
            "shoulder_requests": sum(r["layouts"][name][
                "actor_chosen_action"] != "hold" for r in rows)}
    report = {"scope": "V331 same-packet actor ablation of original four forward/side shin imagers, new two rear-outward imagers, one-sided additions, and all six. Only V330's 96 in-bound 18/24cm/s cases are scored; 30cm/s floor intrusions cannot be rescued by arm posture. Counterfactual scene clearance is evaluator-only and never selects actor actions. Fixed pose and sensor-package limits remain.",
              "source_sha256": sha(SOURCE), "v330_audit_sha256": sha(V330),
              "selected_visual_sha256": sha(selected_visual()),
              "layouts": LAYOUTS, "summary": summary, "rows": rows}
    (OUT / "rear-sensor-ablation-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
