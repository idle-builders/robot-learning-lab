"""V326 frozen V324 alarm/arm request across predeclared local prop geometry.

Both actor scans and 20ms evaluator continuations use the selected Direct
Frame. Actor decisions see noisy shin returns and retained estimated pose;
scene geometry and clearance are evaluator-only labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_shin_range_v190 import scan as shin_scan
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import (
    ALARM_S, ELBOW_FLEX_RAD, END_S, FLEX_TIME_S, START_S,
    actor_alarm, actual_pose, estimated_pose, estimated_shin_points, panel, scene,
)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v326"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (170, 172)
X_OFFSETS_M = (-.10, 0., .10)
PROP_HEIGHTS_M = (.04, .08, .12)


def vary_prop(room: LateCorridorRoom, time_s: float, *, mirrored: bool,
              x_offset_m: float, height_m: float, base_y: float) -> None:
    scene(room, time_s, moving=True, mirrored=mirrored, base_y=base_y)
    gid = room.model.geom("loop_stool_geom").id
    room.model.geom_size[gid, 2] = height_m / 2
    y = float(room.model.body_pos[room.model.body("loop_stool").id][1])
    room.model.body_pos[room.model.body("loop_stool").id] = (
        x_offset_m, y, height_m / 2)
    room._forward()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        trace_path = (V323 / f"seed{seed}-present" /
                      f"seed{seed}-present/episode-trace.json")
        trace = json.loads(trace_path.read_text())
        actual, estimated = actual_pose(trace), estimated_pose(trace)
        for mirrored in (False, True):
            for height_m in PROP_HEIGHTS_M:
                for x_offset_m in X_OFFSETS_M:
                    label = (f"seed{seed}-" + ("right" if mirrored else "left") +
                             f"-h{round(height_m * 100):02d}-x{round(x_offset_m * 100):+03d}")
                    branch = OUT / label
                    branch.mkdir(parents=True, exist_ok=True)
                    room = LateCorridorRoom(case=tuple(trace["case"]),
                                            pose_source="command", branch=branch)
                    try:
                        room.layout(seed, absent=False, furnished=True)
                        room.actual[:] = actual
                        room.estimated[:] = estimated
                        robot = RobotGeometry(room.model)
                        captures = []
                        clouds = []
                        for index, time_s in enumerate((START_S, ALARM_S)):
                            vary_prop(room, time_s, mirrored=mirrored,
                                      x_offset_m=x_offset_m, height_m=height_m,
                                      base_y=actual[1])
                            cloud, counts = shin_scan(
                                room, robot, {"case": f"v326-{label}-{index}"})
                            points = estimated_shin_points(
                                cloud["four"], actual, estimated)
                            path = branch / f"shin-{index}.npz"
                            np.savez_compressed(path, points_xyz_m=points,
                                                estimated_pose_xyyaw=estimated,
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
                                      height_m == .12 and x_offset_m == .10)
                        for policy in ("hold", "alarm_elbow_flex"):
                            minimum = (float("inf"), None, "", "")
                            rendered = []
                            joint = (f"{alarm['side']}_elbow_pitch"
                                     if alarm["side"] and policy == "alarm_elbow_flex"
                                     else None)
                            address = (int(room.model.joint(joint).qposadr[0])
                                       if joint else None)
                            for index, time_s in enumerate(np.linspace(START_S, END_S, 41)):
                                vary_prop(room, float(time_s), mirrored=mirrored,
                                          x_offset_m=x_offset_m,
                                          height_m=height_m, base_y=actual[1])
                                room.data.qpos[18:27] = 0.
                                if address is not None:
                                    fraction = np.clip(
                                        (time_s - ALARM_S) / FLEX_TIME_S, 0., 1.)
                                    room.data.qpos[address] = ELBOW_FLEX_RAD * fraction
                                room._forward()
                                gap, body, obstacle = room._current_obstacle_clearance()
                                if gap < minimum[0]:
                                    minimum = (gap, float(time_s), body, obstacle)
                                if video_case and index % 5 == 0:
                                    rendered.append(panel(np.asarray(room.capture_external()),
                                        f"{label} | {policy} | t={time_s:.2f}s | gap={gap:.3f}m"))
                            outcomes[policy] = {
                                "minimum_evaluator_clearance_m": round(minimum[0], 5),
                                "time_of_minimum_s": round(minimum[1], 4),
                                "closest_robot_geom": minimum[2],
                                "closest_obstacle_geom": minimum[3],
                                "requested_joint": joint}
                            frames[policy] = rendered
                        video_path = None
                        if video_case:
                            video_path = branch / "hold-vs-alarm-flex.mp4"
                            with imageio.get_writer(video_path, fps=6, codec="libx264",
                                                    quality=7) as writer:
                                for left, right in zip(frames["hold"],
                                                       frames["alarm_elbow_flex"]):
                                    writer.append_data(np.concatenate((left, right), axis=1))
                        row = {"case": label, "seed": seed,
                               "mirrored_evaluator_setup": mirrored,
                               "prop_height_m_evaluator_only": height_m,
                               "prop_x_offset_m_evaluator_only": x_offset_m,
                               "actor_alarm": alarm, "actor_captures": captures,
                               "evaluator_outcomes": outcomes,
                               "video_path": (str(video_path.relative_to(ROOT))
                                              if video_path else None),
                               "video_sha256": sha(video_path) if video_path else None,
                               "v323_trace_sha256": sha(trace_path)}
                        rows.append(row)
                        print(label, alarm["side"],
                              outcomes["hold"]["minimum_evaluator_clearance_m"],
                              outcomes["alarm_elbow_flex"]["minimum_evaluator_clearance_m"],
                              flush=True)
                    finally:
                        room.close()
    hazardous = [row for row in rows if row["evaluator_outcomes"]["hold"][
        "minimum_evaluator_clearance_m"] < .05]
    report = {"scope": "Frozen V324 two-scan 3cm/0.2s shin alarm and kinematic side-elbow flexion on two retained Direct Frame blocked poses, left/right approach, three x offsets and three low-prop heights. Local hazard geometry is deliberately changed, with all dimensions fixed before reading results. Evaluator truth only scores the 20ms continuation; no native contact or actuator claim.",
              "source_sha256": sha(SOURCE),
              "v324_source_sha256": sha(Path(actor_alarm.__code__.co_filename)),
              "selected_visual_sha256": sha(selected_visual()),
              "design": {"seeds": SEEDS, "x_offsets_m": X_OFFSETS_M,
                         "heights_m": PROP_HEIGHTS_M, "sides": ("left", "right")},
              "summary": {"cases": len(rows),
                          "hazardous_hold_cases": len(hazardous),
                          "hazardous_detected": sum(r["actor_alarm"]["side"] is not None
                                                    for r in hazardous),
                          "hazardous_rescued_above_5cm": sum(
                              r["evaluator_outcomes"]["alarm_elbow_flex"][
                                  "minimum_evaluator_clearance_m"] >= .05
                              for r in hazardous)},
              "rows": rows}
    (OUT / "shin-intrusion-geometry-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
