"""V330 frozen rear-shin/alarm/action gate on new local geometry and speeds.

This uses new furnished seeds and prop offset/height/speed combinations not
used to select the V329 rear view or hold-versus-shoulder rule. It is still a
fixed-pose kinematic continuation, not a live owner search or motor test.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_protective_postures_v327 import self_clearance
from audit_shin_range_v190 import scan as shin_scan
from direct_structural_rooms_v270 import selected_visual
from protective_action_gate import compare
from rear_shin_range import scan as rear_scan
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import (
    ALARM_S, END_S, FLEX_TIME_S, START_S, actor_alarm,
    actual_pose, estimated_pose, estimated_shin_points, panel,
)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v330"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (186, 188)
CASE_TAIL = ("present", .30, 9.8, 10.4, 10.8, 12.8)
X_OFFSETS_M = (-.08, -.02, .08, .12)
HEIGHTS_M = (.07, .11, .15)
SPEEDS_M_S = (.18, .24, .30)
SHOULDER_TARGET_RAD = -.8


def set_scene(room: LateCorridorRoom, time_s: float, *, side: int,
              x_offset_m: float, height_m: float, speed_m_s: float,
              base_y: float) -> None:
    room.time_s = time_s
    room._update_moving_obstacle()
    y_relative = .45 - speed_m_s * max(0., time_s - START_S)
    y = base_y + side * (y_relative - base_y)
    gid = room.model.geom("loop_stool_geom").id
    room.model.geom_size[gid, 2] = height_m / 2
    room.model.body_pos[room.model.body("loop_stool").id] = (
        x_offset_m, y, height_m / 2)
    room._forward()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    anchor_path = V323 / "seed170-present/seed170-present/episode-trace.json"
    anchor = json.loads(anchor_path.read_text())
    actual, estimated = actual_pose(anchor), estimated_pose(anchor)
    rows = []
    for seed in SEEDS:
        for side in (1, -1):
            side_name = "left" if side == 1 else "right"
            for speed_m_s in SPEEDS_M_S:
                for height_m in HEIGHTS_M:
                    for x_offset_m in X_OFFSETS_M:
                        label = (f"seed{seed}-{side_name}-v{round(speed_m_s * 100):02d}"
                                 f"-h{round(height_m * 100):02d}"
                                 f"-x{round(x_offset_m * 100):+03d}")
                        branch = OUT / label
                        branch.mkdir(parents=True, exist_ok=True)
                        room = LateCorridorRoom(case=(seed, *CASE_TAIL),
                                                pose_source="command", branch=branch)
                        try:
                            room.layout(seed, absent=False, furnished=True)
                            room.actual[:] = actual
                            room.estimated[:] = estimated
                            robot = RobotGeometry(room.model)
                            captures, clouds = [], []
                            for index, time_s in enumerate((START_S, ALARM_S)):
                                set_scene(room, time_s, side=side,
                                          x_offset_m=x_offset_m, height_m=height_m,
                                          speed_m_s=speed_m_s, base_y=actual[1])
                                base, base_counts = shin_scan(
                                    room, robot, {"case": f"v330-{label}-{index}-front"})
                                rear_clouds = []
                                rear_counts = {}
                                for sensor_side, sensor_name in ((1, "left"), (-1, "right")):
                                    extra, counts = rear_scan(
                                        room, sensor_side,
                                        f"v330-{label}-{index}-{sensor_name}-rear")
                                    rear_clouds.append(extra)
                                    rear_counts[sensor_name] = counts
                                front_estimated = estimated_shin_points(
                                    base["four"], actual, estimated)
                                left_rear_estimated = estimated_shin_points(
                                    rear_clouds[0], actual, estimated)
                                right_rear_estimated = estimated_shin_points(
                                    rear_clouds[1], actual, estimated)
                                points = np.concatenate((front_estimated,
                                                         left_rear_estimated,
                                                         right_rear_estimated))
                                path = branch / f"range-{index}.npz"
                                np.savez_compressed(path, points_xyz_m=points,
                                                    front_points_xyz_m=front_estimated,
                                                    left_rear_points_xyz_m=left_rear_estimated,
                                                    right_rear_points_xyz_m=right_rear_estimated,
                                                    estimated_pose_xyyaw=estimated,
                                                    actor_qpos=room.data.qpos.copy(),
                                                    capture_time_s=time_s)
                                captures.append({"path": str(path.relative_to(ROOT)),
                                                 "sha256": sha(path),
                                                 "front_points": len(base["four"]),
                                                 "rear_points": {name: counts[
                                                     "environment_points"] for name, counts
                                                     in rear_counts.items()},
                                                 "front_scan_counts": base_counts,
                                                 "rear_scan_counts": rear_counts})
                                clouds.append(points)
                            alarm = actor_alarm(clouds[0], clouds[1], estimated)
                            forecast = ({"status": "no_alarm", "chosen": "hold"}
                                        if alarm["side"] is None else
                                        compare(clouds[1], estimated,
                                                room.data.qpos.copy(),
                                                alarm["side"], robot))
                            chosen = forecast["chosen"]
                            outcomes = {}
                            frames = {}
                            video_case = label in (
                                "seed186-left-v24-h15-x-02",
                                "seed186-left-v30-h15-x-02")
                            for policy in ("hold", "actor_choice"):
                                joint = (f"{alarm['side']}_shoulder_pitch"
                                         if policy == "actor_choice" and
                                         chosen == "shoulder_pitch" else None)
                                address = (int(room.model.joint(joint).qposadr[0])
                                           if joint else None)
                                scene_min = (float("inf"), "", "")
                                self_min = (float("inf"), "", "")
                                rendered = []
                                for index, time_s in enumerate(np.linspace(START_S, END_S, 41)):
                                    set_scene(room, float(time_s), side=side,
                                              x_offset_m=x_offset_m, height_m=height_m,
                                              speed_m_s=speed_m_s, base_y=actual[1])
                                    room.data.qpos[18:27] = 0.
                                    if address is not None:
                                        fraction = np.clip(
                                            (time_s - ALARM_S) / FLEX_TIME_S, 0., 1.)
                                        room.data.qpos[address] = (
                                            SHOULDER_TARGET_RAD * fraction)
                                    room._forward()
                                    gap, body, obstacle = room._current_obstacle_clearance()
                                    scene_min = min(scene_min, (gap, body, obstacle))
                                    self_min = min(self_min,
                                                   self_clearance(room, side_name))
                                    if video_case and index % 5 == 0:
                                        rendered.append(panel(np.asarray(room.capture_external()),
                                            f"{label} | {('hold' if policy == 'hold' else chosen)} | "
                                            f"t={time_s:.2f}s | gap={gap:.3f}m"))
                                outcomes[policy] = {
                                    "minimum_scene_clearance_m": round(scene_min[0], 5),
                                    "closest_scene_geoms": [scene_min[1], scene_min[2]],
                                    "minimum_self_clearance_m": round(self_min[0], 5),
                                    "closest_self_geoms": [self_min[1], self_min[2]],
                                    "requested_joint": joint}
                                frames[policy] = rendered
                            video_path = None
                            if video_case:
                                video_path = branch / "hold-vs-actor-choice.mp4"
                                with imageio.get_writer(video_path, fps=6,
                                                        codec="libx264", quality=7) as writer:
                                    for left, right in zip(frames["hold"],
                                                           frames["actor_choice"]):
                                        writer.append_data(np.concatenate((left, right), axis=1))
                            row = {"case": label, "seed": seed,
                                   "side_evaluator_only": side_name,
                                   "prop_height_m_evaluator_only": height_m,
                                   "prop_x_offset_m_evaluator_only": x_offset_m,
                                   "prop_speed_m_s_evaluator_only": speed_m_s,
                                   "actor_captures": captures,
                                   "actor_alarm": alarm,
                                   "actor_action_forecast": forecast,
                                   "evaluator_outcomes": outcomes,
                                   "video_path": (str(video_path.relative_to(ROOT))
                                                  if video_path else None),
                                   "video_sha256": sha(video_path) if video_path else None}
                            rows.append(row)
                            print(label, alarm["side"], chosen,
                                  outcomes["hold"]["minimum_scene_clearance_m"],
                                  outcomes["actor_choice"]["minimum_scene_clearance_m"],
                                  flush=True)
                        finally:
                            room.close()
    def score(group: list[dict]) -> dict:
        hazards = [r for r in group if r["evaluator_outcomes"]["hold"][
            "minimum_scene_clearance_m"] < .05]
        return {"cases": len(group), "hazardous_hold_cases": len(hazards),
                "hazardous_detected": sum(r["actor_alarm"]["side"] is not None
                                          for r in hazards),
                "hazardous_rescued_above_5cm": sum(
                    r["evaluator_outcomes"]["actor_choice"][
                        "minimum_scene_clearance_m"] >= .05 for r in hazards),
                "harmful_actions_from_safe_hold": sum(
                    r["evaluator_outcomes"]["hold"]["minimum_scene_clearance_m"] >= .05
                    and r["evaluator_outcomes"]["actor_choice"][
                        "minimum_scene_clearance_m"] < .05 for r in group),
                "any_self_overlap": sum(r["evaluator_outcomes"]["actor_choice"][
                    "minimum_self_clearance_m"] < 0. for r in group)}
    report = {"scope": "Frozen V329 rear-outward shin sensor and observed-point hold-versus-shoulder rule on two new furnished/light seeds, four new x offsets, three new heights and three approach speeds. 18/24cm/s are inside the action gate's 25cm/s assumption; 30cm/s deliberately exceeds it. Selected Direct Frame base remains fixed at the retained V323 pose; this is kinematic moving-prop stress, not complete owner-loop or native motor proof.",
              "source_sha256": sha(SOURCE),
              "rear_sensor_source_sha256": sha(Path(rear_scan.__code__.co_filename)),
              "action_gate_source_sha256": sha(Path(compare.__code__.co_filename)),
              "selected_visual_sha256": sha(selected_visual()),
              "v323_anchor_trace_sha256": sha(anchor_path),
              "actual_pose_evaluator_only": actual.tolist(),
              "actor_estimated_pose_xyyaw": estimated.tolist(),
              "design": {"seeds": SEEDS, "x_offsets_m": X_OFFSETS_M,
                         "heights_m": HEIGHTS_M, "speeds_m_s": SPEEDS_M_S,
                         "sides": ("left", "right")},
              "summary": {"in_bound": score([r for r in rows if r[
                  "prop_speed_m_s_evaluator_only"] <= .25]),
                          "out_of_bound": score([r for r in rows if r[
                              "prop_speed_m_s_evaluator_only"] > .25])},
              "rows": rows}
    (OUT / "rear-action-holdout-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
