"""V95: keep an actor low-range interrupt alive through the owner-help phase.

The V92 furnished route and articulated owner remain. Four front/side shin
views drive a 10Hz moving-surface cue; rear-right range points gate a bounded
rightward response. The point gate is provisional: its packets omit free rays,
so it cannot certify the full exit volume or a physical skill.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from articulated_person_scene import add_person
from audit_shin_range_v190 import scan as front_scan
import direct_structural_rooms_v270 as structural
from rear_shin_range import scan as rear_scan
from robot_geometry import RobotGeometry
from run import ROOT, sha
import run_blocked_owner_help_v317 as owner_module
import run_live_articulated_owner_v343 as v343
import run_live_moving_owner_offer_v338 as prior
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_shin_intrusion_posture_v324 import (actor_alarm,
                                             estimated_shin_points)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v95"
CASES = tuple(case for case in owner_module.CASES
              if case[0] in (170, 172) and case[1] == "present")
SAMPLE_INTERVAL_S = .1
CLOSING_THRESHOLD_M_PER_0P2S = .018
SIDE_STEP_M = .08
MIN_POINT_GAP_M = .06
MIN_RIGHT_REAR_POINTS = 8


class OwnerPhaseMonitor:
    def __init__(self, room, branch: Path, seed: int) -> None:
        self.room = room
        self.branch = branch
        self.seed = seed
        self.robot = RobotGeometry(room.model)
        self.frames = []
        self.points = []
        self.alarm = None
        self.next_capture_s = room.time_s
        self.original_forward = room._forward
        self.capture()
        self.next_capture_s += SAMPLE_INTERVAL_S

        def monitored_forward():
            self.original_forward()
            if self.room.time_s + 1e-6 >= self.next_capture_s:
                self.capture()
                self.next_capture_s += SAMPLE_INTERVAL_S

        room._forward = monitored_forward

    def capture(self) -> None:
        room = self.room
        index = len(self.frames)
        front, counts = front_scan(
            room, self.robot,
            {"case": f"v95-seed{self.seed}-tick{index}-front"})
        right, right_counts = rear_scan(
            room, -1, f"v95-seed{self.seed}-tick{index}-right-rear")
        front_points = estimated_shin_points(
            front["four"], room.actual.copy(), room.estimated.copy())
        right_points = estimated_shin_points(
            right, room.actual.copy(), room.estimated.copy())
        self.points.append((front_points, right_points))
        motion = (actor_alarm(self.points[index-2][0], front_points,
                              room.estimated.copy()) if index >= 2 else None)
        closing = (motion["side_returns"]["left"]["closing_0p2s_m"]
                   if motion else None)
        new_alarm = (self.alarm is None and closing is not None and
                     closing >= CLOSING_THRESHOLD_M_PER_0P2S)
        if new_alarm:
            self.alarm = {"capture_time_s": round(room.time_s, 4),
                          "capture_index": index,
                          "side": "left",
                          "closing_0p2s_m": closing}
            # The V317 owner callback checks this after its scripted wait.
            # The outer V95 callback restores the evaluator contact flag and
            # replaces that callback's generic interruption status.
            room.motion_interrupted = True
        path = self.branch / f"owner-guard-{index:02d}.npz"
        np.savez_compressed(
            path, front_four_points_xyz_m=front_points,
            right_rear_points_xyz_m=right_points,
            estimated_pose_xyyaw=room.estimated.copy(),
            actor_qpos=room.data.qpos.copy(),
            capture_time_s=room.time_s)
        self.frames.append({
            "capture_time_s": round(room.time_s, 4),
            "packet_path": str(path.relative_to(ROOT)),
            "packet_sha256": sha(path),
            "front_points": len(front_points),
            "right_rear_points": len(right_points),
            "front_counts": counts,
            "right_rear_counts": right_counts,
            "left_closing_0p2s_m": closing,
            "actor_alarm": new_alarm,
            "evaluator_gap_m": round(room._current_obstacle_clearance()[0], 5),
        })

    def point_gate(self) -> dict:
        front, right = self.points[-1]
        pose = self.room.estimated.copy()
        qpos = self.room.data.qpos.copy()
        minimum = float("inf")
        if len(right):
            for fraction in np.linspace(0., 1., 9):
                candidate = pose.copy()
                candidate[1] -= SIDE_STEP_M * fraction
                self.robot.set_state(self.robot.pose_qpos(qpos, candidate))
                minimum = min(minimum, float(np.min(
                    self.robot.solid_distances(
                        right, self.robot.primitives()))))
        allowed = len(right) >= MIN_RIGHT_REAR_POINTS and minimum >= MIN_POINT_GAP_M
        return {"status": "provisional_point_only",
                "right_rear_points": len(right),
                "minimum_swept_right_rear_point_gap_m": round(minimum, 5),
                "point_margin_m": MIN_POINT_GAP_M,
                "allowed": bool(allowed),
                "unknown_free_ray_volume": True}

    def restore(self) -> None:
        self.room._forward = self.original_forward

    def save(self, response: dict) -> dict:
        payload = {
            "schema": "robot-learning-lab.ape-continuous-owner-guard.v1",
            "seed": self.seed,
            "alarm": self.alarm,
            "frames": self.frames,
            "response": response,
            "scope": "Four front/side first-hit shin point clouds at 10Hz drive anonymous left-side closing cue; separate right-rear points form only a provisional side-step point gate. Captures continue through scripted owner motion and the step/hold. Evaluator gaps are scoring only."
        }
        path = self.branch / "continuous-owner-guard.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                "alarm": self.alarm, "response": response}


def monitored_owner_motion(room, branch, video, seed, mode, condition):
    wrapped = v343.move_person_during_wait(
        room, branch, video, seed, mode, condition)
    room.v95_monitor = OwnerPhaseMonitor(room, branch, seed)
    return wrapped


def main() -> None:
    original = (prior.OUT, prior.add_owner_hand,
                prior.move_owner_during_wait, prior.LateCorridorRoom,
                prior.CASES, owner_module.OWNER_SHOW_XYZ_M,
                owner_module.search.run_case)
    captured = []

    def intercept_run_case(*args, **kwargs):
        callback = kwargs["on_route_blocked"]
        seed = args[0]

        def monitored_callback(room, observe, branch, video):
            response = callback(room, observe, branch, video)
            monitor = room.v95_monitor
            if monitor.alarm is not None:
                room.motion_interrupted = False
                gate = monitor.point_gate()
                started = room.time_s
                if gate["allowed"]:
                    # Research-only kinematic continuation: the existing ray
                    # packet omits observed-free volume, so run_case must not
                    # interpret this as an actor-authorized recovery step.
                    start_actual = room.actual.copy()
                    start_estimated = room.estimated.copy()
                    start_time = room.time_s
                    for step in range(1, 21):
                        fraction = step / 20
                        room.actual[:] = start_actual
                        room.actual[1] -= SIDE_STEP_M * .55 * fraction
                        room.estimated[:] = start_estimated
                        room.estimated[1] -= SIDE_STEP_M * fraction
                        room.time_s = start_time + .4 * fraction
                        room._update_moving_obstacle()
                        room._forward()
                        scene = room._current_obstacle_clearance()
                        room.path_min_clearance = min(room.path_min_clearance,
                                                      scene)
                        if scene[0] < 0:
                            room.motion_interrupted = True
                            break
                        if step % 5 == 0:
                            rgb, _ = room.capture("lower_rgb", depth=False)
                            video.append_data(owner_module.panel(
                                room, rgb,
                                f"V95 right-side response t{room.time_s:.1f}"))
                    room.wait()
                    rgb, _ = room.capture("lower_rgb", depth=False)
                    video.append_data(owner_module.panel(
                        room, rgb, f"V95 held after response t{room.time_s:.1f}"))
                prior_owner_status = response["status"]
                response.update({
                    "status": "owner_turn_near_body_unresolved",
                    "owner_questions": 1,
                    "owner_answer": "not_asked",
                    "autonomous_find": False,
                    "reason": "front-side low-motion cue interrupted owner show",
                    "prior_owner_callback_status": prior_owner_status,
                })
                actor_response = {
                    "status": "research_right_side_step_provisional" if gate["allowed"]
                              else "no_point_supported_side_step",
                    "gate": gate, "start_time_s": round(started, 4),
                    "end_time_s": round(room.time_s, 4),
                    "evaluator_minimum_scene_gap_m": round(
                        room.path_min_clearance[0], 5),
                    "evaluator_contact": room.motion_interrupted,
                }
                if gate["allowed"]:
                    response["research_kinematic_reposition"] = True
                    response["evaluator_preparation_min_scene_gap_m"] = round(
                        room.path_min_clearance[0], 5)
            else:
                actor_response = {"status": "no_owner_phase_alarm"}
            monitor.restore()
            saved = monitor.save(actor_response)
            captured.append({"seed": seed, "condition": branch.parts[-3],
                             "monitor": saved})
            response["continuous_owner_guard"] = saved
            return response

        kwargs["on_route_blocked"] = monitored_callback
        kwargs["allow_research_reposition"] = True
        return original[-1](*args, **kwargs)

    try:
        prior.OUT = OUT
        prior.add_owner_hand = add_person
        prior.move_owner_during_wait = monitored_owner_motion
        prior.LateCorridorRoom = OfferDriftRoom
        prior.CASES = CASES
        owner_module.OWNER_SHOW_XYZ_M = v343.OWNER_SHOW_XYZ_M
        owner_module.search.run_case = intercept_run_case
        prior.main()
    finally:
        (prior.OUT, prior.add_owner_hand,
         prior.move_owner_during_wait, prior.LateCorridorRoom,
         prior.CASES, owner_module.OWNER_SHOW_XYZ_M,
         owner_module.search.run_case) = original
    path = OUT / "live-moving-owner-offer-audit.json"
    report = json.loads(path.read_text())
    report.update({
        "schema": "robot-learning-lab.ape-continuous-owner-guard.v1",
        "source_sha256": sha(SOURCE),
        "v343_source_sha256": sha(Path(v343.__file__)),
        "capture_count": len(captured),
        "scope": "V92 furnished articulated owner and low-prop drift with 10Hz four-front-shin owner-phase monitor, separate right-rear point-only side gate, early owner-task interruption, and a kinematic 8cm right-side response when the provisional point gate passes. The route guard and prior search remain from V343. Unknown free rays, physical stop, selected body and fresh owner resumption remain unqualified.",
    })
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(path)


if __name__ == "__main__":
    main()
