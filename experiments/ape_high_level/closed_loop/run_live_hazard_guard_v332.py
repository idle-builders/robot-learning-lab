"""V332: run the rear-range/action guard inside the live owner search.

The unmodified V321 owner route, active head viewpoints, target detector and
forward-step gate run in the same room after this guarded blocked-route wait.
The guard sees noisy joint-mounted range points, own qpos and estimated pose;
future scene distance is evaluator-only. Shoulder motion is a qpos surrogate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_shin_range_v190 import scan as front_scan
from depth_quality import reject_isolated_depth_spikes
from direct_structural_rooms_v270 import selected_visual
from protective_action_gate import compare
from rear_shin_range import scan as rear_scan
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_blocked_forward_viewpoint_v321 import panel, main as run_owner
import run_blocked_forward_viewpoint_v321 as owner_module
from run_late_corridor_intrusion_v323 import (FINAL_Y_M, INTRUSION_START_S,
                                              LateCorridorRoom)
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_shin_intrusion_posture_v324 import (actor_alarm,
                                              estimated_shin_points)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v332"
MAX_BLOCKED_MONITOR_S = 1.6
SAMPLE_INTERVAL_S = .2
SHOULDER_TARGET_RAD = -.8
ACTION_S = .4


class ParkedLateRoom(LateCorridorRoom):
    """Same earlier search, but park the late low prop at 45 cm."""

    def _update_moving_obstacle(self) -> None:
        super()._update_moving_obstacle()
        if self.motion_side is None or self.time_s < INTRUSION_START_S:
            return
        self.obstacle_xy[3] = (0., .45)
        self.model.body_pos[self.model.body("loop_stool").id] = (
            0., .45, STOOL_SIZE[2])


class LiveGuard:
    def __init__(self, room, branch: Path, video, seed: int,
                 label: str) -> None:
        self.room = room
        self.branch = branch
        self.video = video
        self.seed = seed
        self.label = label
        self.robot = RobotGeometry(room.model)
        self.frames: list[dict] = []
        self.decisions: list[dict] = []
        self.evaluator_path_min = (float("inf"), "", "")
        self.response: dict | None = None
        self.start_time_s = room.time_s

    def capture(self) -> dict:
        room = self.room
        index = len(self.frames)
        front, front_counts = front_scan(
            room, self.robot, {"case": f"v332-{self.label}-{index}-front"})
        components = [front["four"]]
        rear_counts = {}
        for side, name in ((1, "left"), (-1, "right")):
            points, counts = rear_scan(
                room, side, f"v332-{self.label}-{index}-{name}-rear")
            components.append(points)
            rear_counts[name] = counts
        actual = room.actual.copy()  # sensor adapter only
        estimated = room.estimated.copy()
        sensed = estimated_shin_points(
            np.concatenate(components), actual, estimated)
        path = self.branch / f"guard-range-{index:02d}.npz"
        np.savez_compressed(path, points_xyz_m=sensed,
                            estimated_pose_xyyaw=estimated,
                            actor_qpos=room.data.qpos.copy(),
                            capture_time_s=room.time_s)
        entry = {"index": index, "capture_time_s": round(room.time_s, 4),
                 "actor_packet_path": str(path.relative_to(ROOT)),
                 "actor_packet_sha256": sha(path),
                 "front_points": len(front["four"]),
                 "rear_points": [len(p) for p in components[1:]],
                 "front_counts": front_counts,
                 "rear_counts": rear_counts,
                 "estimated_pose_xyyaw": estimated.tolist()}
        self.frames.append(entry)
        if index:
            previous = self.frames[-2]
            with np.load(ROOT / previous["actor_packet_path"]) as packet:
                older = packet["points_xyz_m"].copy()
            alarm = actor_alarm(older, sensed, estimated)
            decision = ({"status": "no_alarm", "chosen": "hold"}
                        if alarm["side"] is None else
                        compare(sensed, estimated, room.data.qpos.copy(),
                                alarm["side"], self.robot))
            entry["actor_alarm"] = alarm
            entry["actor_action_forecast"] = decision
            self.decisions.append({"capture_index": index,
                                   "capture_time_s": room.time_s,
                                   "actor_alarm": alarm,
                                   "actor_action_forecast": decision})
            if decision["chosen"] == "shoulder_pitch" and self.response is None:
                self.response = {"status": "protective_action_requested",
                                 "side": alarm["side"],
                                 "trigger_time_s": room.time_s,
                                 "capture_index": index,
                                 "actor_action_forecast": decision}
        return entry

    def probe(self, substep: int, completed: float) -> dict | None:
        room = self.room
        if (not self.frames or room.time_s - self.frames[-1]["capture_time_s"]
                >= SAMPLE_INTERVAL_S - 1e-6):
            self.capture()
            if self.response is not None:
                return {"status": "protective_action_requested",
                        "capture_index": self.response["capture_index"]}
        return None

    def render(self, label: str) -> None:
        rgb, _ = self.room.capture("lower_rgb", depth=False)
        self.video.append_data(panel(
            self.room, rgb, f"V332 {self.label} | {label} | t{self.room.time_s:.1f}"))

    def apply_request(self) -> None:
        if self.response is None:
            return
        room = self.room
        joint = f"{self.response['side']}_shoulder_pitch"
        address = int(room.model.joint(joint).qposadr[0])
        first = float(room.data.qpos[address])
        start_time = room.time_s
        nearest = (float("inf"), "", "")
        for step in range(1, 21):
            fraction = step / 20
            room.time_s = start_time + ACTION_S * fraction
            room._update_moving_obstacle()
            room.data.qpos[address] = first + (SHOULDER_TARGET_RAD - first) * fraction
            room._forward()
            gap = room._current_obstacle_clearance()
            nearest = min(nearest, gap)
            room.path_min_clearance = min(room.path_min_clearance, gap)
            if gap[0] < 0:
                room.motion_interrupted = True
                break
            if step in (5, 10, 15, 20):
                self.render(f"protective {joint} {fraction:.0%}")
        self.evaluator_path_min = nearest
        self.response.update({"status": "protective_surrogate_completed"
                              if not room.motion_interrupted else
                              "protective_surrogate_contact",
                              "requested_joint": joint,
                              "requested_target_rad": SHOULDER_TARGET_RAD,
                              "start_time_s": start_time,
                              "end_time_s": room.time_s,
                              "evaluator_minimum_action_clearance_m": round(nearest[0], 5),
                              "evaluator_closest_geoms": [nearest[1], nearest[2]]})

    def run(self) -> dict:
        room = self.room
        self.capture()
        self.render("route blocked; watching")
        while room.time_s < self.start_time_s + MAX_BLOCKED_MONITOR_S - 1e-6:
            room.wait(safety_probe=self.probe)
            self.render("monitoring low motion")
            if room.motion_interrupted:
                break
            if self.response is not None:
                self.apply_request()
                break
        if self.response is None:
            status = ("evaluator_contact_without_alarm" if room.motion_interrupted
                      else "no_protective_action")
        else:
            status = self.response["status"]
        return {"status": status,
                "start_time_s": self.start_time_s,
                "end_time_s": room.time_s,
                "sample_count": len(self.frames),
                "response": self.response,
                "evaluator_path_min_clearance_m": round(
                    room.path_min_clearance[0], 5)}

    def save(self, original_status: str) -> dict:
        trace_path = self.branch / "live-guard-trace.json"
        payload = {"scope": "Actor range monitoring and hold/shoulder comparison during the same V321 live owner search after route refusal. Simulated qpos response; scene clearance is evaluator-only.",
                   "label": self.label, "seed": self.seed,
                   "start_time_s": self.start_time_s,
                   "end_time_s": self.room.time_s,
                   "actor_frames": self.frames,
                   "actor_decisions": self.decisions,
                   "response": self.response,
                   "owner_continuation_status": original_status,
                   "evaluator_final_model_clearance_m": round(
                       self.room._current_obstacle_clearance()[0], 5),
                   "final_actor_joint_qpos": self.room.data.qpos.tolist(),
                   "evaluator_minimum_action_clearance_m": (
                       round(self.evaluator_path_min[0], 5)
                       if np.isfinite(self.evaluator_path_min[0]) else None),
                   "evaluator_motion_interrupted": self.room.motion_interrupted}
        trace_path.write_text(json.dumps(payload, indent=2) + "\n")
        return {"trace_path": str(trace_path.relative_to(ROOT)),
                "trace_sha256": sha(trace_path),
                "monitor_status": (self.response["status"] if self.response
                                   else "no_protective_action"),
                "sample_count": len(self.frames)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    original_room = owner_module.HeldoutMovingRoom
    original_run_case = owner_module.search.run_case
    try:
        for condition, room_type in (("moving", LateCorridorRoom),
                                     ("parked", ParkedLateRoom)):
            guard_summaries = []

            def guarded_run_case(*args, **kwargs):
                base_callback = kwargs["on_route_blocked"]
                seed = args[0]

                def guarded_callback(room, observe, branch, video):
                    guard = LiveGuard(room, branch, video, seed,
                                      f"seed{seed}-{condition}")
                    guard.run()
                    result = base_callback(room, observe, branch, video)
                    summary = guard.save(result["status"])
                    guard_summaries.append(summary)
                    result["hazard_guard"] = summary
                    return result

                kwargs["on_route_blocked"] = guarded_callback
                return original_run_case(*args, **kwargs)

            owner_module.HeldoutMovingRoom = room_type
            owner_module.search.run_case = guarded_run_case
            output_root = OUT / condition
            run_owner(output_root=output_root, source=SOURCE,
                      map_depth_filter=reject_isolated_depth_spikes,
                      experiment_label=f"V332-{condition}")
            prior_path = output_root / "blocked-forward-viewpoint-audit.json"
            prior = json.loads(prior_path.read_text())
            rows.extend({"condition": condition, **row}
                        for row in prior["rows"])
            if len(guard_summaries) != 2:
                raise AssertionError("Expected both owner episodes to reach blocked guard")
    finally:
        owner_module.HeldoutMovingRoom = original_room
        owner_module.search.run_case = original_run_case
    report = {"scope": "V332 live selected Direct Frame owner teach/find continuation with V321 search and guarded blocked-route monitoring, paired late moving versus parked low prop. The actor uses six noisy first-hit shin views, retained pose/encoders, and a bounded own-body point forecast; scene truth never gates a requested posture or V321 forward step. Shoulder response is direct qpos interpolation with 20ms scene updates, not a native policy. Starting the monitor at route refusal adds time before V321's active head/owner continuation; this is not yet an always-on guard during earlier search moves.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "v321_source_sha256": sha(Path(owner_module.__file__)),
              "rear_sensor_source_sha256": sha(Path(rear_scan.__code__.co_filename)),
              "action_gate_source_sha256": sha(Path(compare.__code__.co_filename)),
              "rows": rows}
    (OUT / "live-hazard-owner-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
