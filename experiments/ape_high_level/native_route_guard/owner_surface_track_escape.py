"""V52: use anonymous timed surface motion to decide a held owner turn.

The actor fits a small cluster of fused low/mid ray endpoints near the nearest
return, estimates its velocity from consecutive packets, and tests the held
body center against the cluster's constant-velocity path. No evaluator object
position, identity, endpoint or scripted path enters the decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import owner_full_angle_escape as v51


v50 = v51.v50
runner = v51.runner
ROOT = v51.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v52-surface-track-escape"
CLUSTER_HALF_WIDTH_RAYS = 4
CLUSTER_RANGE_BAND_M = .18
MAX_ASSOCIATION_STEP_M = .30
MAX_TRACK_AGE_S = .35
PREDICTION_HORIZON_S = .60
HELD_CENTER_CLEARANCE_M = .21
MIN_TRACK_SPEED_MPS = .08


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cluster(packet, pose):
    valid = [(float(value), index) for index, value in enumerate(packet)
             if value is not None]
    if len(valid) < 36:
        return None, None, None, len(valid)
    nearest, ray = min(valid)
    indexes = [index for index in range(
        max(0, ray-CLUSTER_HALF_WIDTH_RAYS),
        min(len(packet), ray+CLUSTER_HALF_WIDTH_RAYS+1))
        if packet[index] is not None and
        packet[index] <= nearest+CLUSTER_RANGE_BAND_M]
    points = np.asarray([
        np.asarray(pose)+packet[index]*np.array([
            math.cos(math.radians(-180.+360.*index/len(packet))),
            math.sin(math.radians(-180.+360.*index/len(packet)))])
        for index in indexes])
    return np.mean(points, axis=0), nearest, ray, len(valid)


class SurfaceTrackEscape(v51.FullAngleEscape):
    def start(self, xy, time_s, packet):
        super().start(xy, time_s, packet)
        self.old_center = None
        self.old_center_s = None
        self.track_count = 0

    def observe(self, time_s, xy, packet):
        center, nearest, ray, valid = cluster(packet, xy)
        healthy = center is not None
        velocity = None
        predicted = None
        closest_at_s = None
        associated = False
        if healthy and self.old_center is not None:
            dt = float(time_s)-self.old_center_s
            delta = center-self.old_center
            associated = bool(0. < dt <= MAX_TRACK_AGE_S and
                              np.linalg.norm(delta) <= MAX_ASSOCIATION_STEP_M)
            if associated:
                velocity = delta/dt
                self.track_count += 1
                speed2 = float(np.dot(velocity, velocity))
                if speed2 >= MIN_TRACK_SPEED_MPS**2:
                    closest_at_s = float(np.clip(
                        -np.dot(center-np.asarray(xy), velocity)/speed2,
                        0., PREDICTION_HORIZON_S))
                    predicted = float(np.linalg.norm(
                        center+closest_at_s*velocity-np.asarray(xy)))
            else:
                self.track_count = 0
        if not healthy:
            self.track_count = 0
        if self.alarm_s is None and predicted is not None and \
                predicted < HELD_CENTER_CLEARANCE_M:
            future_surface = center+closest_at_s*velocity
            away = np.asarray(xy)-future_surface
            self.direction = away/np.linalg.norm(away)
            self.alarm_s = float(time_s)
            self.last_command_s = float(time_s)
        self.old_center = center
        self.old_center_s = float(time_s) if healthy else None
        self.last_packet = list(packet)
        self.last_packet_s = float(time_s)
        return {"alarm": self.alarm_s is not None,
                "capture_time_s": round(float(time_s), 5),
                "valid_rays": valid, "healthy": healthy,
                "nearest_range_m": round(nearest, 5) if healthy else None,
                "nearest_bearing_deg": (-180.+360.*ray/len(packet)
                                        if healthy else None),
                "cluster_center_xy_m": center.tolist() if healthy else None,
                "associated": associated, "track_count": self.track_count,
                "cluster_velocity_mps": (velocity.tolist() if velocity is not None
                                         else None),
                "predicted_center_to_surface_m": (round(predicted, 5)
                                                  if predicted is not None else None),
                "closest_at_s": closest_at_s,
                "escape_direction": self.direction.tolist() if self.alarm_s
                is not None else None}


def main():
    source = runner.original()
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = runner.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(runner.owner.TEACH_CROP).convert(
        "RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    original_geometry = runner.geometry_at
    original_packet = runner.observed_packet
    results = []
    try:
        runner.geometry_at = v51.v46.low_prop_geometry(original_geometry)
        runner.observed_packet = v51.v46.plane_packet(
            original_packet, (.05, .30))
        for path_name, end_xy in v50.PATHS.items():
            obstacle_fn = lambda t, end_xy=end_xy: v50.prop_at(
                t, checkpoint_s, end_xy)
            for case in v51.v47.CASES:
                history = []

                def retain_actor_packet(packet, time_s, stage):
                    if stage == "post_alarm_range_watch":
                        history.append({"time_s": float(time_s),
                                        "stage": stage, "packet": list(packet)})
                    return packet

                run = runner.run_case(
                    case, source, policy, scenario, vision, exemplar,
                    flank_probe_factory=lambda video:
                        v51.v41.LocalizedFlankCamera(video),
                    flank_response=runner.FLANK_RESPONSE.get(case),
                    require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                    response_controller=(v51.v41.JointHazardResponse()
                                         if case in runner.FLANK_RESPONSE else None),
                    pre_show_restage=v51.v47.HistoryMeetingPose(history),
                    owner_fovy_deg=50., range_transform=retain_actor_packet,
                    near_body_controller=SurfaceTrackEscape(),
                    output_root=OUT / path_name / case)
                rows = json.loads(Path(run["trace"]["path"]).read_text())
                commands = [row["near_body_command"] for row in rows
                            if row["stage"] == "second_inference_range_watch"
                            and row["near_body_command"] is not None and
                            row["near_body_command"]["commanded_step_m"] > 0]
                alarm = next((row for row in rows if row["stage"] ==
                              "near_body_escape_plan"), None)
                results.append({"path": path_name, "case": case,
                                "run": run,
                                "near_body_alarm_s": (alarm["time_s"]
                                                      if alarm else None),
                                "near_body_commands": len(commands),
                                "near_body_commanded_m": sum(
                                    row["commanded_step_m"] for row in commands)})
                print(path_name, case, run["outcome"],
                      run["minimum_proxy_gap_m"], run["minimum_prop_gap_m"],
                      results[-1]["near_body_commanded_m"], flush=True)
    finally:
        runner.geometry_at = original_geometry
        runner.observed_packet = original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-surface-track-escape.v1",
              "task": "T-9v3p3n", "source_sha256": sha(__file__),
              "runner_source_sha256": sha(runner.__file__),
              "v50_report_sha256": sha(v50.OUT / "report.json"),
              "v50_source_sha256": sha(v50.__file__),
              "v51_report_sha256": sha(v51.OUT / "report.json"),
              "scene_sha256": sha(runner.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(runner.SOURCE),
              "current_design_pointer_sha256": sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "sensor_heights_m": [.05, .30],
              "cluster_half_width_rays": CLUSTER_HALF_WIDTH_RAYS,
              "cluster_range_band_m": CLUSTER_RANGE_BAND_M,
              "maximum_association_step_m": MAX_ASSOCIATION_STEP_M,
              "maximum_track_age_s": MAX_TRACK_AGE_S,
              "prediction_horizon_s": PREDICTION_HORIZON_S,
              "held_center_clearance_m": HELD_CENTER_CLEARANCE_M,
              "minimum_track_speed_mps": MIN_TRACK_SPEED_MPS,
              "path_end_xy_m": {name: xy.tolist()
                                for name, xy in v50.PATHS.items()},
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
