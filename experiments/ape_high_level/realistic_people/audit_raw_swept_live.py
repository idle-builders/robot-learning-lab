"""Audit live raw-scan veto timing, action effect, and independent geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from audit_viewpoint_route import box_clearance, evaluate
from raw_swept_gate import assess_swept_low_hits


def audit(run: Path) -> dict:
    actions = json.loads((run / "evidence/online-action-trace.json").read_text())["actions"]
    route = evaluate(run / "evidence")
    scene = json.loads((run / "evidence/scene-info.json").read_text())
    monitor_path = run / "evidence/continuous-vision-monitor.json"
    monitor = json.loads(monitor_path.read_text())["frames"] if monitor_path.exists() else []
    range_frames = json.loads((run / "evidence/range-fan-trace.json").read_text())["frames"]
    angles = [beam["angle_deg"] for beam in range_frames[0]["planes"][0]["beams"]]
    low_scan_times = [row["timeline_time_s"] for row in range_frames]
    obstacle_size = scene["route_obstacle_evaluator_only"]["size_xy_m"]
    held_gap = None
    held_hit_alerts = []
    live_held_alarm_frames = []
    for row in monitor:
        frame = row["frame"]
        if row.get("decision_scan_time_s") is not None:
            low_scan_times.append(row["decision_scan_time_s"])
        if row.get("held_raw_alarm"):
            live_held_alarm_frames.append(frame)
            command = actions[frame]["command_for_next_frame"]
            assert command["forward_step_m"] <= 0, (frame, command)
            assert not command.get("body_yaw_delta_deg"), (frame, command)
        # Inference advances the world before this frame's command. The eye
        # remains at the preceding command's executed pose throughout it.
        eye = (scene["camera_eye_xyz_m"][:2] if frame == 0 else
               actions[frame - 1]["proxy_eye_xyz_m_evaluator_only"][:2])
        for sample in row["samples"]:
            if "0" in sample["range_m_by_plane"]:
                low_scan_times.append(sample["time_s"])
            # Saved monitor nulls mix clear and invalid rays, so only a
            # coherent finite occupied patch can be assessed here.
            low_hits = sample["range_m_by_plane"].get("0")
            if low_hits and assess_swept_low_hits(low_hits, angles, step_m=0.,
                    step_heading_rel_deg=0., clearance_m=.24)["veto"]:
                held_hit_alerts.append({"frame": frame,
                    "time_s": sample["time_s"]})
            cube = sample.get("route_cube_center_xy_m_evaluator_only")
            if cube is None:
                continue
            gap = box_clearance(eye, cube, obstacle_size)
            if held_gap is None or gap < held_gap["clearance_m"]:
                held_gap = {"clearance_m": gap, "frame": frame,
                            "time_s": sample["time_s"]}
    veto_rows = []
    vetoed_step_evaluator = []
    unknown_rows = []
    positive_step_rows = []
    full_detector_owner_frames = 0
    flow_owner_frames = 0
    identity_hold_frames = []
    memory_gaze_frames = []
    for row in actions:
        frame = row["frame"]
        command = row["command_for_next_frame"]
        actor = row["actor"]
        actionable = actor.get("intent") in (
            "look_to_owner", "approach_bounded", "hold_near_person")
        full_detector_owner_frames += (actionable and
                                       actor.get("chosen_proposal") is not None)
        flow_owner_frames += (actionable and
                              actor.get("observation_method") == "rgbd_flow" and
                              actor.get("track", {}).get("status") == "tracked")
        if actor.get("intent") in ("hold_ask_show_again", "hold_reobserve"):
            identity_hold_frames.append(frame)
        if actor.get("memory_gaze_only"):
            memory_gaze_frames.append(frame)
        evidence = row.get("raw_swept_evidence")
        if evidence is not None:
            assert evidence["proposed_route_intent"] == "step_to_waypoint"
            assert 0 <= row["decision_time_s"] - evidence["scan_time_s"] < .1
            if evidence["veto"]:
                veto_rows.append(frame)
                assert command["forward_step_m"] <= 0, (frame, command)
                eye = row["proxy_eye_xyz_m_evaluator_only"][:2]
                cube = row["route_cube_center_xy_m_evaluator_only"]
                heading = math.radians(row["route_body_heading_deg_commanded"])
                step = evidence["step_m"]
                proposed = [eye[0] + step * math.sin(heading),
                            eye[1] + step * math.cos(heading)]
                vetoed_step_evaluator.append({"frame": frame,
                    "held_gap_m": box_clearance(eye, cube, obstacle_size),
                    "proposed_step_gap_m": box_clearance(proposed, cube,
                                                         obstacle_size)})
            if evidence["status"] == "unknown":
                unknown_rows.append(frame)
                assert command["forward_step_m"] <= 0, (frame, command)
        if command["forward_step_m"] > 0:
            positive_step_rows.append(frame)
            if any(item.get("raw_swept_evidence") is not None for item in actions):
                assert evidence is not None and not evidence["veto"] and (
                    evidence["status"] != "unknown"), (frame, evidence)
    return {"scope": "live Isaac eye-proxy route; cube gap is evaluator only",
            "frames": len(actions),
            "full_detector_owner_frames": full_detector_owner_frames,
            "flow_owner_frames": flow_owner_frames,
            "owner_actionable_frames": full_detector_owner_frames + flow_owner_frames,
            "identity_hold_frames": identity_hold_frames,
            "memory_gaze_frames": memory_gaze_frames,
            "raw_swept_veto_frames": veto_rows,
            "vetoed_step_evaluator_only": vetoed_step_evaluator,
            "raw_swept_unknown_frames": unknown_rows,
            "positive_step_frames": positive_step_rows,
            "minimum_cube_box_clearance_m": route["minimum_cube_box_clearance_m"],
            "minimum_cube_box_clearance_during_vision": held_gap,
            "held_occupied_patch_alert_samples": held_hit_alerts,
            "live_held_alarm_frames": live_held_alarm_frames,
            "max_low_scan_gap_sim_s": max((right - left for left, right in
                zip(sorted(low_scan_times), sorted(low_scan_times)[1:])), default=None),
            "temporary_proxy_clearance_m": route["temporary_proxy_clearance_m"],
            "proxy_clearance_pass": route["temporary_proxy_clearance_pass"],
            "proxy_clearance_during_vision_pass": (held_gap is not None and
                held_gap["clearance_m"] >= route["temporary_proxy_clearance_m"]),
            "commanded_forward_m": route["commanded_forward_m"],
            "commanded_reverse_m": route["commanded_reverse_m"],
            "final_owner_distance_m": route["final_owner_distance_m"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = audit(args.run)
    (args.run / "raw-swept-live-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
