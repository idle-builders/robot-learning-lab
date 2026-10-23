"""V328 frozen actor shin alarm + shoulder-pitch request on new geometry.

The response was selected after V326/V327; x offsets and heights here were
not used to choose it. This is still a kinematic skill surrogate.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_protective_postures_v327 import self_clearance
from audit_shin_intrusion_geometry_v326 import vary_prop
from audit_shin_range_v190 import scan as shin_scan
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import (
    ALARM_S, END_S, FLEX_TIME_S, START_S, actor_alarm,
    actual_pose, estimated_pose, estimated_shin_points, panel,
)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v328"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (170, 172)
X_OFFSETS_M = (-.15, -.05, .05, .15)
HEIGHTS_M = (.06, .10, .14)
SHOULDER_TARGET_RAD = -.8


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        trace_path = (V323 / f"seed{seed}-present" /
                      f"seed{seed}-present/episode-trace.json")
        trace = json.loads(trace_path.read_text())
        actual, estimated = actual_pose(trace), estimated_pose(trace)
        for mirrored in (False, True):
            side = "right" if mirrored else "left"
            for height_m in HEIGHTS_M:
                for x_offset_m in X_OFFSETS_M:
                    label = (f"seed{seed}-{side}-h{round(height_m * 100):02d}"
                             f"-x{round(x_offset_m * 100):+03d}")
                    branch = OUT / label
                    branch.mkdir(parents=True, exist_ok=True)
                    room = LateCorridorRoom(case=tuple(trace["case"]),
                                            pose_source="command", branch=branch)
                    try:
                        room.layout(seed, absent=False, furnished=True)
                        room.actual[:] = actual
                        room.estimated[:] = estimated
                        robot = RobotGeometry(room.model)
                        captures, clouds = [], []
                        for index, time_s in enumerate((START_S, ALARM_S)):
                            vary_prop(room, time_s, mirrored=mirrored,
                                      x_offset_m=x_offset_m, height_m=height_m,
                                      base_y=actual[1])
                            cloud, counts = shin_scan(
                                room, robot, {"case": f"v328-{label}-{index}"})
                            points = estimated_shin_points(
                                cloud["four"], actual, estimated)
                            path = branch / f"shin-{index}.npz"
                            np.savez_compressed(path, points_xyz_m=points,
                                                estimated_pose_xyyaw=estimated,
                                                actor_qpos=room.data.qpos.copy(),
                                                capture_time_s=time_s)
                            captures.append({"path": str(path.relative_to(ROOT)),
                                             "sha256": sha(path),
                                             "point_count": len(points),
                                             "scan_counts": counts})
                            clouds.append(points)
                        alarm = actor_alarm(clouds[0], clouds[1], estimated)
                        outcomes = {}
                        frames = {}
                        video_case = (seed == SEEDS[0] and not mirrored and
                                      height_m == .14 and x_offset_m == .15)
                        for policy in ("hold", "alarm_shoulder_pitch"):
                            joint = (f"{alarm['side']}_shoulder_pitch"
                                     if alarm["side"] and policy == "alarm_shoulder_pitch"
                                     else None)
                            address = (int(room.model.joint(joint).qposadr[0])
                                       if joint else None)
                            scene_min = (float("inf"), "", "")
                            self_min = (float("inf"), "", "")
                            rendered = []
                            for index, time_s in enumerate(np.linspace(START_S, END_S, 41)):
                                vary_prop(room, float(time_s), mirrored=mirrored,
                                          x_offset_m=x_offset_m, height_m=height_m,
                                          base_y=actual[1])
                                room.data.qpos[18:27] = 0.
                                if address is not None:
                                    fraction = np.clip(
                                        (time_s - ALARM_S) / FLEX_TIME_S, 0., 1.)
                                    room.data.qpos[address] = SHOULDER_TARGET_RAD * fraction
                                room._forward()
                                gap, body, obstacle = room._current_obstacle_clearance()
                                scene_min = min(scene_min, (gap, body, obstacle))
                                self_min = min(self_min, self_clearance(room, side))
                                if video_case and index % 5 == 0:
                                    rendered.append(panel(np.asarray(room.capture_external()),
                                        f"{label} | {policy} | t={time_s:.2f}s | gap={gap:.3f}m"))
                            outcomes[policy] = {
                                "minimum_scene_clearance_m": round(scene_min[0], 5),
                                "closest_scene_geoms": [scene_min[1], scene_min[2]],
                                "minimum_self_clearance_m": round(self_min[0], 5),
                                "closest_self_geoms": [self_min[1], self_min[2]],
                                "requested_joint": joint,
                                "requested_target_rad": (SHOULDER_TARGET_RAD
                                                         if joint else None)}
                            frames[policy] = rendered
                        video_path = None
                        if video_case:
                            video_path = branch / "hold-vs-alarm-shoulder-pitch.mp4"
                            with imageio.get_writer(video_path, fps=6, codec="libx264",
                                                    quality=7) as writer:
                                for left, right in zip(frames["hold"],
                                                       frames["alarm_shoulder_pitch"]):
                                    writer.append_data(np.concatenate((left, right), axis=1))
                        row = {"case": label, "seed": seed,
                               "mirrored_evaluator_setup": mirrored,
                               "prop_height_m_evaluator_only": height_m,
                               "prop_x_offset_m_evaluator_only": x_offset_m,
                               "actor_alarm": alarm,
                               "actor_captures": captures,
                               "evaluator_outcomes": outcomes,
                               "video_path": (str(video_path.relative_to(ROOT))
                                              if video_path else None),
                               "video_sha256": sha(video_path) if video_path else None,
                               "v323_trace_sha256": sha(trace_path)}
                        rows.append(row)
                        print(label, alarm["side"],
                              outcomes["hold"]["minimum_scene_clearance_m"],
                              outcomes["alarm_shoulder_pitch"][
                                  "minimum_scene_clearance_m"], flush=True)
                    finally:
                        room.close()
    hazards = [r for r in rows if r["evaluator_outcomes"]["hold"][
        "minimum_scene_clearance_m"] < .05]
    report = {"scope": "Frozen V324 actor two-shin-scan alarm plus V327-selected side shoulder-pitch -0.8rad request, tested on previously unused x offsets and prop heights at the two retained blocked poses. 20ms kinematic scene/self geometry is evaluator-only. This is not native skill feasibility or a full owner-search run.",
              "source_sha256": sha(SOURCE),
              "actor_alarm_source_sha256": sha(Path(actor_alarm.__code__.co_filename)),
              "selected_visual_sha256": sha(selected_visual()),
              "design": {"seeds": SEEDS, "x_offsets_m": X_OFFSETS_M,
                         "heights_m": HEIGHTS_M, "sides": ("left", "right")},
              "summary": {"cases": len(rows), "hazardous_hold_cases": len(hazards),
                          "hazardous_detected": sum(r["actor_alarm"]["side"] is not None
                                                    for r in hazards),
                          "hazardous_rescued_scene_5cm_self_nonoverlap": sum(
                              r["evaluator_outcomes"]["alarm_shoulder_pitch"][
                                  "minimum_scene_clearance_m"] >= .05 and
                              r["evaluator_outcomes"]["alarm_shoulder_pitch"][
                                  "minimum_self_clearance_m"] >= 0.
                              for r in hazards)},
              "rows": rows}
    (OUT / "shoulder-response-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
