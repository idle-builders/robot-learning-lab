"""Use aimed head depth as a second moving-hazard source during range loss."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import actor_range_refuge_v744 as selector
import head_depth_compact_track_v764 as depth_track
import run_layout2_adaptive_renewed_hold_dropout_v761 as dropout
import run_layout2_adaptive_refuge_v744 as adaptive
import run_layout2_adaptive_exit_turn_v749 as turn
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
from screen_robocasa_owner_safe_stop_v494 import aim_camera


SOURCE = Path(__file__)
OUT = dropout.trial.adaptive.west.side.ROOT / "artifacts/ape-robocasa-layout2-dropout-head-depth-response-v765"
CAPTURE_INTERVAL_S = .2
MIN_DEPTH_TRACK_SPEED_M_S = .25
THREAT_HORIZON_S = 2.5


class DepthMonitor(dropout.DropoutMonitor):
    depth_action_enabled = False
    output_branch = OUT / "depth_off"

    def __init__(self, model, data, actual, script, grid, estimate):
        super().__init__(model, data, actual, script, grid, estimate)
        self.depth_renderer = None
        self.last_depth_capture_s = None
        self.depth_observations = []
        self.head_target_track = None
        tracks = grid.multi_history[-1]["tracks"] if grid.multi_history else []
        unknown = [track for track in tracks
                   if full.FIRST_SURFACE is None or
                   np.linalg.norm(np.asarray(track["center_xy_m"]) -
                                  full.FIRST_SURFACE) > .45]
        if unknown:
            self.head_target_track = min(unknown, key=lambda track:
                                         np.linalg.norm(
                                             np.asarray(track["center_xy_m"]) -
                                             estimate[:2]))
            aim_camera(model, "head", estimate[:2],
                       self.head_target_track["center_xy_m"], 20)

    def __call__(self, t):
        result = super().__call__(t)
        missing = result["range_track_stream_health"] == "missing"
        if missing:
            result["invalidate_image"] = True
            result["actor_source_health_gate"] = "reject_capture_on_missing_range_track"
        if not missing or self.head_target_track is None:
            return result
        if (self.last_depth_capture_s is not None and
                t - self.last_depth_capture_s < CAPTURE_INTERVAL_S - .001):
            return result
        if self.depth_renderer is None:
            self.depth_renderer = mujoco.Renderer(self.model, height=480, width=640)
        rgb, depth, wall = owner.capture(self.depth_renderer, self.data, "head", True)
        self.last_depth_capture_s = t
        folder = self.output_branch / "head-depth-packets"
        folder.mkdir(parents=True, exist_ok=True)
        number = len(self.depth_observations)
        rgb_path = folder / f"{number:02d}.png"
        depth_path = folder / f"{number:02d}.npy"
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth)
        head = self.model.camera("head").id
        observation = depth_track.observe(
            depth, self.data.cam_xpos[head],
            self.data.cam_xmat[head].reshape(3, 3),
            self.model.cam_fovy[head], self.estimate[:2],
            np.asarray(self.head_target_track["center_xy_m"]),
            float(self.head_target_track["radius_m"]))
        packet = {"capture_time_s": t, "capture_wall_s": wall,
                  "last_range_track": self.head_target_track,
                  "rgb_path": str(rgb_path), "rgb_sha256": route.sha256(rgb_path),
                  "depth_path": str(depth_path), "depth_sha256": route.sha256(depth_path),
                  "actor_observation": observation}
        result["actor_head_depth_packet"] = packet
        if observation is None:
            return result
        self.depth_observations.append((t, np.asarray(observation["center_xy_m"])))
        if len(self.depth_observations) < 3 or self.trigger_time_s is not None:
            return result
        first_t, first_center = self.depth_observations[-3]
        velocity = (self.depth_observations[-1][1] - first_center) / (t - first_t)
        center = self.depth_observations[-1][1]
        relative = center - self.estimate[:2]
        speed2 = float(np.dot(velocity, velocity))
        closest_t = float(np.clip(-np.dot(relative, velocity) / speed2,
                                  0., THREAT_HORIZON_S)) if speed2 > 1.e-6 else 0.
        gap = (float(np.linalg.norm(center + closest_t * velocity - self.estimate[:2])) -
               float(self.head_target_track["radius_m"]) -
               selector.BODY_RADIUS_M - selector.POSE_TRACK_RESERVE_M)
        result["actor_head_depth_motion"] = {
            "velocity_xy_m_s": velocity.tolist(), "speed_m_s": float(np.linalg.norm(velocity)),
            "closest_time_s": closest_t, "predicted_gap_lower_m": gap,
            "would_interrupt": (float(np.linalg.norm(velocity)) > MIN_DEPTH_TRACK_SPEED_M_S and
                                gap < selector.REQUIRED_GAP_M)}
        if (not self.depth_action_enabled or
                not result["actor_head_depth_motion"]["would_interrupt"]):
            return result
        track = {"id": self.head_target_track["id"],
                 "status": "confirmed_now",
                 "center_xy_m": center.tolist(),
                 "radius_m": self.head_target_track["radius_m"]}
        goal, evidence = selector.choose(
            self.grid, self.estimate[:2], track, velocity, full.FIRST_SURFACE,
            allow_best_effort=True)
        requested, status, path = ((None, "no_supported_response", []) if goal is None
                                   else route.SensorGrid.plan(self.grid, self.estimate[:2], goal))
        self.trigger_time_s = t
        self.hazard_id = track["id"]
        self.hazard_centers = self.depth_observations.copy()
        self.start_estimated = self.estimate[:2].copy()
        self.start_actual = self.actual.copy()
        self.response_goal = None if goal is None else goal.copy()
        self.plan_status = status
        if status == "known_goal" and requested is not None:
            self.estimated_path = np.asarray(
                [self.start_estimated] + [self.grid.xy(cell) for cell in path[1:]])
            self.actual_path = (self.start_actual +
                                (self.estimated_path - self.start_estimated) *
                                selector.ROUTE_SCALE)
            self.path_length = float(np.sum(np.linalg.norm(
                np.diff(self.actual_path, axis=0), axis=1)))
            self.dispatch_time_s = t
        result.update({"actor_response_trigger_time_s": t,
                       "actor_response_source": "head_depth_during_range_dropout",
                       "actor_goal_xy_m": None if goal is None else goal.tolist(),
                       "actor_response_selection": evidence,
                       "actor_plan_status": status,
                       "actor_response_qualification": evidence["status"],
                       "actor_estimated_path_xy_m": None if self.estimated_path is None
                           else self.estimated_path.tolist(),
                       "invalidate_image": True})
        return result


class DepthOffMonitor(DepthMonitor):
    output_branch = OUT / "depth_off"


class DepthOnMonitor(DepthMonitor):
    depth_action_enabled = True
    output_branch = OUT / "depth_on"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {
        "schema": "robot-learning-lab.layout2-dropout-head-depth-response.v1",
        "control_report_sha256": route.sha256(dropout.OUT / "report.json"),
        "same_range_track_outage_s": dropout.DROP_UNTIL_AFTER_CAPTURE_S,
        "profiles": ["depth_off", "depth_on"],
        "both_profiles_capture_identical_type_head_packets_and_invalidate_item_on_range_loss": True,
        "head_aim": "nearest non-person actor range track at owner-image capture",
        "head_depth_capture_interval_s": CAPTURE_INTERVAL_S,
        "depth_track": "prior-target-gated 3D surface, body-height mask, stride-3 depth pixels, 3-center velocity",
        "depth_threat": "speed >0.25m/s and 2.5s projected clearance <5cm after 10cm pose/track reserve",
        "response": "V744 actor-map radial timed route and renewable hold, explicitly unqualified best effort if no goal supports every hazard hypothesis",
        "source_sha256": route.sha256(SOURCE),
        "extractor_source_sha256": route.sha256(Path(depth_track.__file__))}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    for label, monitor in (("depth_off", DepthOffMonitor),
                           ("depth_on", DepthOnMonitor)):
        adaptive.main(prop_factory=turn.ExitTurnProp, out=OUT / label,
                      profile=f"V761 0.8s range-track outage; aimed head depth {label}",
                      monitor_factory=monitor)
        path = OUT / label / "report.json"
        report = json.loads(path.read_text())
        report["depth_response_declaration_sha256"] = route.sha256(OUT / "predeclared.json")
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
