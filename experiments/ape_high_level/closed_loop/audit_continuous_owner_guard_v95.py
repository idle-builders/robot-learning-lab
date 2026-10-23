"""Independent V95 packet-alarm and 20ms response-geometry replay."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from articulated_person_scene import FAR_HAND_XYZ_M, add_person, set_person_pose
import direct_structural_rooms_v270 as structural
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import (OUT as V92,
                                                  OfferDriftRoom)
from run_continuous_owner_guard_v95 import (CLOSING_THRESHOLD_M_PER_0P2S,
                                            MIN_POINT_GAP_M, OUT, SIDE_STEP_M,
                                            SOURCE)
from run_live_hazard_guard_v332 import ParkedLateRoom
from run_shin_intrusion_posture_v324 import actor_alarm


AUDIT_SOURCE = Path(__file__)


def main() -> None:
    report = json.loads((OUT / "live-moving-owner-offer-audit.json").read_text())
    assert report["source_sha256"] == sha(SOURCE)
    assert len(report["rows"]) == 4
    checks = 2
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    results = []
    try:
        for row in report["rows"]:
            condition, seed = row["condition"], row["case"][0]
            branch = (OUT / condition / f"seed{seed}-present" /
                      f"seed{seed}-present")
            trace_path = ROOT / row["trace_path"]
            assert sha(trace_path) == row["trace_sha256"]
            trace = json.loads(trace_path.read_text())
            original_trace = json.loads((
                V92 / condition / f"seed{seed}-present" /
                f"seed{seed}-present/episode-trace.json").read_text())
            def observable_prefix(source):
                return [(event.get("action"), event.get("label"),
                         event.get("actor_input_sha256"),
                         event.get("evaluator_actual_pose_xyyaw"))
                        for event in source["controller_events"]
                        if event.get("action") != "route_help_resolution"]
            old_prefix = observable_prefix(original_trace)
            new_prefix = observable_prefix(trace)
            assert old_prefix[:len(new_prefix)] == new_prefix
            checks += 1
            events = [event["event"] for event in
                      trace["owner_intervention_events"]]
            data_path = branch / "continuous-owner-guard.json"
            monitor = json.loads(data_path.read_text())
            frames = monitor["frames"]
            assert len(frames) == 13
            checks += 3
            packets = []
            for index, frame in enumerate(frames):
                path = ROOT / frame["packet_path"]
                assert sha(path) == frame["packet_sha256"]
                with np.load(path) as packet:
                    assert abs(float(packet["capture_time_s"]) -
                               frame["capture_time_s"]) < 1e-5
                    packets.append({key: packet[key].copy()
                                    for key in packet.files})
                if index:
                    assert abs(frame["capture_time_s"] -
                               frames[index-1]["capture_time_s"] - .1) < 1e-5
                checks += 2
            alarms = []
            for index in range(2, len(frames)):
                replay = actor_alarm(
                    packets[index-2]["front_four_points_xyz_m"],
                    packets[index]["front_four_points_xyz_m"],
                    packets[index]["estimated_pose_xyyaw"])
                closing = replay["side_returns"]["left"]["closing_0p2s_m"]
                assert closing == frames[index]["left_closing_0p2s_m"]
                candidate = (closing is not None and
                             closing >= CLOSING_THRESHOLD_M_PER_0P2S)
                if candidate and not alarms:
                    alarms.append(frames[index]["capture_time_s"])
                assert frames[index]["actor_alarm"] == (
                    candidate and len(alarms) == 1 and
                    frames[index]["capture_time_s"] == alarms[0])
                checks += 2
            video_path = ROOT / row["video_path"]
            assert sha(video_path) == row["video_sha256"]
            video = imageio.get_reader(video_path)
            try:
                first = video.get_data(0)
                last = video.get_data(video.count_frames()-1)
                assert first.shape == last.shape and first.ndim == 3
            finally:
                video.close()
            checks += 2
            if condition == "moving":
                assert row["status"] == "owner_turn_near_body_unresolved"
                assert row["owner_questions"] == 1
                assert events == ["route_help_requested"]
                assert alarms == [16.4]
                assert monitor["alarm"]["capture_time_s"] == 16.4
                assert monitor["response"]["status"] == (
                    "research_right_side_step_provisional")
                assert monitor["response"]["gate"]["unknown_free_ray_volume"]
                checks += 7
                # Recompute the point-only candidate gate from retained actor
                # packet 16.6, keeping actor estimated pose separate from truth.
                packet = packets[4]
                room = OfferDriftRoom(case=tuple(row["case"]),
                                      pose_source="command", branch=OUT / "audit")
                try:
                    room.layout(seed, absent=False, furnished=True)
                    qpos = packet["actor_qpos"]
                    pose = packet["estimated_pose_xyyaw"]
                    right = packet["right_rear_points_xyz_m"]
                    robot = RobotGeometry(room.model)
                    minimum_point = float("inf")
                    for fraction in np.linspace(0., 1., 9):
                        candidate_pose = pose.copy()
                        candidate_pose[1] -= SIDE_STEP_M*fraction
                        robot.set_state(robot.pose_qpos(qpos,
                                                       candidate_pose))
                        minimum_point = min(minimum_point, float(np.min(
                            robot.solid_distances(right,
                                                  robot.primitives()))))
                    assert abs(minimum_point - monitor["response"]["gate"][
                        "minimum_swept_right_rear_point_gap_m"]) < .0001
                    assert minimum_point >= MIN_POINT_GAP_M
                    checks += 2
                    # Reconstruct the evaluator-only continuous response with
                    # saved actual qpos. The task never receives these gaps.
                    room.data.qpos[:] = qpos
                    start_actual = np.array((qpos[0], qpos[1], 0.))
                    room.head = float(qpos[room.head_addr])
                    room.estimated[:] = pose
                    set_person_pose(room.model, FAR_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    minimum_scene = float("inf")
                    for tick in range(41):
                        time_s = 16.6 + .02*tick
                        fraction = min(1., (time_s-16.6)/.4)
                        room.actual[:] = start_actual
                        room.actual[1] -= SIDE_STEP_M*.55*fraction
                        room.time_s = time_s
                        room._update_moving_obstacle()
                        room._forward()
                        minimum_scene = min(minimum_scene,
                                            room._current_obstacle_clearance()[0])
                        checks += 1
                    assert abs(minimum_scene - row[
                        "evaluator_path_min_clearance_m"]) < .0001
                    assert minimum_scene >= .05
                    checks += 2
                finally:
                    room.close()
            else:
                assert row["status"] == "owner_assisted_visual_reacquisition"
                assert row["owner_questions"] == 2
                assert "scripted_owner_presentation" in events
                assert "scripted_owner_identity_answer" in events
                assert alarms == [] and monitor["alarm"] is None
                assert row["evaluator_path_min_clearance_m"] >= .05
                checks += 6
            assert not row["evaluator_motion_interrupted"]
            checks += 1
            results.append({
                "condition": condition, "seed": seed,
                "status": row["status"],
                "first_alarm_s": alarms[0] if alarms else None,
                "minimum_scene_gap_m": row["evaluator_path_min_clearance_m"],
                "video": str(video_path),
            })
    finally:
        structural.structural_model = old_model
    result = {
        "schema": "robot-learning-lab.ape-continuous-owner-guard-audit.v1",
        "passed": True, "checks": checks,
        "source_sha256": sha(AUDIT_SOURCE),
        "results": results,
        "scope": "Replays owner-phase alarms from saved noisy actor points, independently recomputes the point-only side gate and 20ms MuJoCo response gap, verifies task semantics and decodes complete episode videos. The reposition is a research kinematic continuation; full observed-free volume, asynchronous inference and physical selected-body response are not qualified.",
    }
    path = OUT / "independent-audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path, checks, "checks passed")


if __name__ == "__main__":
    main()
