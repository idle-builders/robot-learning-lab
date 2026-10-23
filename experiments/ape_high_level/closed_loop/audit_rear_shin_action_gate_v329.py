"""V329 rear-shin view plus actor-ranked hold/shoulder response.

This diagnoses the V328 rearward blind patch and harmful false response on
the same geometry set. New geometry and full live owner integration remain
future tests; scene truth is used only for the post-decision evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_protective_postures_v327 import self_clearance
from audit_shin_intrusion_geometry_v326 import vary_prop
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
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v329"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
V328 = ROOT / "artifacts/ape-closed-loop-cove50-v328/shoulder-response-audit.json"
SHOULDER_TARGET_RAD = -.8


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(V328.read_text())
    rows = []
    for earlier in prior["rows"]:
        seed = earlier["seed"]
        label = earlier["case"]
        branch = OUT / label
        branch.mkdir(parents=True, exist_ok=True)
        trace_path = (V323 / f"seed{seed}-present" /
                      f"seed{seed}-present/episode-trace.json")
        trace = json.loads(trace_path.read_text())
        actual, estimated = actual_pose(trace), estimated_pose(trace)
        room = LateCorridorRoom(case=tuple(trace["case"]),
                                pose_source="command", branch=branch)
        try:
            room.layout(seed, absent=False, furnished=True)
            room.actual[:] = actual
            room.estimated[:] = estimated
            robot = RobotGeometry(room.model)
            captures, fused = [], []
            for index, time_s in enumerate((START_S, ALARM_S)):
                vary_prop(room, time_s,
                          mirrored=earlier["mirrored_evaluator_setup"],
                          x_offset_m=earlier["prop_x_offset_m_evaluator_only"],
                          height_m=earlier["prop_height_m_evaluator_only"],
                          base_y=actual[1])
                old_path = ROOT / earlier["actor_captures"][index]["path"]
                if sha(old_path) != earlier["actor_captures"][index]["sha256"]:
                    raise AssertionError(f"Changed V328 actor input: {old_path}")
                with np.load(old_path) as frame:
                    original = frame["points_xyz_m"].copy()
                    qpos = frame["actor_qpos"].copy()
                rear_points = []
                rear_counts = {}
                for side, name in ((1, "left"), (-1, "right")):
                    points, counts = rear_scan(
                        room, side, f"v329-{label}-{index}-{name}")
                    rear_points.append(estimated_shin_points(
                        points, actual, estimated))
                    rear_counts[name] = counts
                combined = np.concatenate((original, *rear_points))
                path = branch / f"shin-fused-{index}.npz"
                np.savez_compressed(path, points_xyz_m=combined,
                                    actor_qpos=qpos,
                                    estimated_pose_xyyaw=estimated,
                                    capture_time_s=time_s)
                captures.append({"path": str(path.relative_to(ROOT)),
                                 "sha256": sha(path),
                                 "original_four_shin_points": len(original),
                                 "extra_rear_points": [len(p) for p in rear_points],
                                 "rear_scan_counts": rear_counts})
                fused.append(combined)
            alarm = actor_alarm(fused[0], fused[1], estimated)
            forecast = ({"status": "no_alarm", "chosen": "hold"}
                        if alarm["side"] is None else
                        compare(fused[1], estimated, qpos, alarm["side"], robot))
            chosen = forecast["chosen"]
            side = "right" if earlier["mirrored_evaluator_setup"] else "left"
            video_case = (label == "seed170-left-h14-x-05")
            outcomes, frames = {}, {}
            for policy in ("hold", "actor_choice"):
                joint = (f"{alarm['side']}_shoulder_pitch"
                         if policy == "actor_choice" and chosen == "shoulder_pitch"
                         else None)
                address = (int(room.model.joint(joint).qposadr[0])
                           if joint else None)
                minimum = (float("inf"), "", "")
                self_minimum = (float("inf"), "", "")
                rendered = []
                for index, time_s in enumerate(np.linspace(START_S, END_S, 41)):
                    vary_prop(room, float(time_s),
                              mirrored=earlier["mirrored_evaluator_setup"],
                              x_offset_m=earlier["prop_x_offset_m_evaluator_only"],
                              height_m=earlier["prop_height_m_evaluator_only"],
                              base_y=actual[1])
                    room.data.qpos[18:27] = 0.
                    if address is not None:
                        fraction = np.clip((time_s - ALARM_S) / FLEX_TIME_S,
                                           0., 1.)
                        room.data.qpos[address] = SHOULDER_TARGET_RAD * fraction
                    room._forward()
                    gap, body, obstacle = room._current_obstacle_clearance()
                    minimum = min(minimum, (gap, body, obstacle))
                    self_minimum = min(self_minimum, self_clearance(room, side))
                    if video_case and index % 5 == 0:
                        rendered.append(panel(np.asarray(room.capture_external()),
                            f"{label} | {('hold' if policy == 'hold' else chosen)} | "
                            f"t={time_s:.2f}s | gap={gap:.3f}m"))
                outcomes[policy] = {
                    "minimum_evaluator_scene_clearance_m": round(minimum[0], 5),
                    "closest_scene_geoms": [minimum[1], minimum[2]],
                    "minimum_evaluator_self_clearance_m": round(self_minimum[0], 5),
                    "closest_self_geoms": [self_minimum[1], self_minimum[2]],
                    "requested_joint": joint}
                frames[policy] = rendered
            video_path = None
            if video_case:
                video_path = branch / "hold-vs-rear-range-action.mp4"
                with imageio.get_writer(video_path, fps=6, codec="libx264",
                                        quality=7) as writer:
                    for left, right in zip(frames["hold"], frames["actor_choice"]):
                        writer.append_data(np.concatenate((left, right), axis=1))
            row = {"case": label, "seed": seed,
                   "mirrored_evaluator_setup": earlier["mirrored_evaluator_setup"],
                   "prop_height_m_evaluator_only": earlier["prop_height_m_evaluator_only"],
                   "prop_x_offset_m_evaluator_only": earlier["prop_x_offset_m_evaluator_only"],
                   "actor_captures": captures,
                   "actor_alarm": alarm,
                   "actor_action_forecast": forecast,
                   "evaluator_outcomes": outcomes,
                   "video_path": (str(video_path.relative_to(ROOT))
                                  if video_path else None),
                   "video_sha256": sha(video_path) if video_path else None,
                   "v328_case_input_sha256": [c["sha256"] for c in earlier["actor_captures"]]}
            rows.append(row)
            print(label, alarm["side"], chosen,
                  outcomes["hold"]["minimum_evaluator_scene_clearance_m"],
                  outcomes["actor_choice"]["minimum_evaluator_scene_clearance_m"],
                  flush=True)
        finally:
            room.close()
    hazards = [r for r in rows if r["evaluator_outcomes"]["hold"][
        "minimum_evaluator_scene_clearance_m"] < .05]
    harmful = [r for r in rows if r["evaluator_outcomes"]["hold"][
        "minimum_evaluator_scene_clearance_m"] >= .05 and
        r["evaluator_outcomes"]["actor_choice"][
            "minimum_evaluator_scene_clearance_m"] < .05]
    report = {"scope": "V329 development replay on V328 geometry: added rear-outward 112.5deg 8x8 first-hit range fan on each shin, fused with retained noisy four-shin actor points. A bounded 25cm/s point forecast compares holding against a -0.8rad shoulder-pitch action using only robot collision primitives and observed points. Evaluator truth scores the chosen 20ms kinematic continuation after actor choice. This is not an independent heldout result, sensor-package qualification or native skill proof.",
              "source_sha256": sha(SOURCE),
              "rear_sensor_source_sha256": sha(Path(rear_scan.__code__.co_filename)),
              "action_gate_source_sha256": sha(Path(compare.__code__.co_filename)),
              "v328_audit_sha256": sha(V328),
              "selected_visual_sha256": sha(selected_visual()),
              "summary": {"cases": len(rows), "hazardous_hold_cases": len(hazards),
                          "hazardous_detected": sum(r["actor_alarm"]["side"] is not None
                                                    for r in hazards),
                          "hazardous_rescued_above_5cm": sum(
                              r["evaluator_outcomes"]["actor_choice"][
                                  "minimum_evaluator_scene_clearance_m"] >= .05
                              for r in hazards),
                          "harmful_actions_from_safe_hold": len(harmful)},
              "rows": rows}
    (OUT / "rear-shin-action-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / Path(rear_scan.__code__.co_filename).name).write_bytes(
        Path(rear_scan.__code__.co_filename).read_bytes())
    (OUT / Path(compare.__code__.co_filename).name).write_bytes(
        Path(compare.__code__.co_filename).read_bytes())


if __name__ == "__main__":
    main()
