"""V51: independent fused-range alarm and bounded escape during owner inference.

The monitor sees only each timed 72-ray packet and actor pose. It interrupts
the slow item worker after two near packets, chooses away from the nearest
measured surface, then checks every bounded step against the latest endpoints.
It does not classify the surface as a person or the scripted prop.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import owner_fixture_clear_prop as v50
import owner_full_angle_escape_runner as runner


v41 = v50.v41
v46 = v50.v46
v47 = v50.v47
ROOT = v50.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v51-full-angle-escape"
ALARM_RANGE_M = .55
ALARM_STREAK = 2
MAX_PACKET_AGE_S = .35
ESCAPE_SPEED_MPS = .25
MAX_STEP_M = .05
MAX_TRAVEL_M = .35
STEP_ENDPOINT_GAP_M = .27


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FullAngleEscape:
    def start(self, xy, time_s, packet):
        self.start_xy = np.asarray(xy).copy()
        self.start_s = float(time_s)
        self.last_packet = list(packet)
        self.last_packet_s = float(time_s)
        self.last_command_s = float(time_s)
        self.streak = 0
        self.alarm_s = None
        self.direction = np.zeros(2)
        self.distance_m = 0.

    def observe(self, time_s, xy, packet):
        valid = [(float(value), i) for i, value in enumerate(packet)
                 if value is not None]
        nearest, ray = min(valid) if valid else (float("inf"), None)
        healthy = len(valid) >= 36
        danger = healthy and nearest < ALARM_RANGE_M
        self.streak = self.streak+1 if danger else 0
        if self.alarm_s is None and self.streak >= ALARM_STREAK:
            bearing = math.radians(-180.+360.*ray/len(packet))
            self.direction = -np.array([math.cos(bearing), math.sin(bearing)])
            self.alarm_s = float(time_s)
            self.last_command_s = float(time_s)
        self.last_packet = list(packet)
        self.last_packet_s = float(time_s)
        return {"alarm": self.alarm_s is not None,
                "capture_time_s": round(float(time_s), 5),
                "valid_rays": len(valid), "healthy": healthy,
                "nearest_range_m": (round(nearest, 5) if valid else None),
                "nearest_bearing_deg": (-180.+360.*ray/len(packet)
                                        if ray is not None else None),
                "risk_streak": self.streak,
                "escape_direction": self.direction.tolist() if self.alarm_s
                is not None else None}

    def command(self, time_s, xy):
        if self.alarm_s is None:
            return np.zeros(2), 0., {"action": "monitor_full_angle",
                                       "commanded_step_m": 0.}
        age = float(time_s)-self.last_packet_s
        dt = max(0., float(time_s)-self.last_command_s)
        self.last_command_s = float(time_s)
        if age > MAX_PACKET_AGE_S:
            return self.direction, 0., {"action": "hold_stale_scan",
                                          "packet_age_s": age,
                                          "commanded_step_m": 0.}
        remaining = MAX_TRAVEL_M-self.distance_m
        if remaining <= 1e-9:
            return self.direction, 0., {"action": "escape_distance_reached",
                                          "packet_age_s": age,
                                          "commanded_step_m": 0.}
        distance = min(MAX_STEP_M, ESCAPE_SPEED_MPS*dt, remaining)
        endpoints = v41.JointHazardResponse.endpoints(
            self.last_packet, np.asarray(xy))
        if not len(endpoints):
            return self.direction, 0., {"action": "hold_no_range_endpoints",
                                          "packet_age_s": age,
                                          "commanded_step_m": 0.}
        nearest = float(np.min(np.linalg.norm(
            endpoints-(np.asarray(xy)+self.direction*distance), axis=1)))
        if nearest < STEP_ENDPOINT_GAP_M:
            return self.direction, 0., {"action": "hold_blocked_escape",
                                          "packet_age_s": age,
                                          "minimum_endpoint_gap_m": nearest,
                                          "commanded_step_m": 0.}
        self.distance_m += distance
        return self.direction, distance, {
            "action": "step_away_from_nearest_surface",
            "packet_age_s": age, "minimum_endpoint_gap_m": nearest,
            "commanded_step_m": distance,
            "cumulative_escape_m": self.distance_m,
            "packet_time_s": round(self.last_packet_s, 5)}


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
        runner.geometry_at = v46.low_prop_geometry(original_geometry)
        runner.observed_packet = v46.plane_packet(original_packet, (.05, .30))
        for path_name, end_xy in v50.PATHS.items():
            obstacle_fn = lambda t, end_xy=end_xy: v50.prop_at(
                t, checkpoint_s, end_xy)
            for case in v47.CASES:
                history = []

                def retain_actor_packet(packet, time_s, stage):
                    if stage == "post_alarm_range_watch":
                        history.append({"time_s": float(time_s),
                                        "stage": stage, "packet": list(packet)})
                    return packet

                run = runner.run_case(
                    case, source, policy, scenario, vision, exemplar,
                    flank_probe_factory=lambda video:
                        v41.LocalizedFlankCamera(video),
                    flank_response=runner.FLANK_RESPONSE.get(case),
                    require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                    response_controller=(v41.JointHazardResponse()
                                         if case in runner.FLANK_RESPONSE else None),
                    pre_show_restage=v47.HistoryMeetingPose(history),
                    owner_fovy_deg=50., range_transform=retain_actor_packet,
                    near_body_controller=FullAngleEscape(),
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
    report = {"schema": "robot-learning-lab.ape-owner-full-angle-escape.v1",
              "task": "T-9v3p3n", "source_sha256": sha(__file__),
              "runner_source_sha256": sha(runner.__file__),
              "v50_report_sha256": sha(v50.OUT / "report.json"),
              "v50_source_sha256": sha(v50.__file__),
              "scene_sha256": sha(runner.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(runner.SOURCE),
              "current_design_pointer_sha256": sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "sensor_heights_m": [.05, .30],
              "alarm_range_m": ALARM_RANGE_M,
              "alarm_streak": ALARM_STREAK,
              "maximum_packet_age_s": MAX_PACKET_AGE_S,
              "escape_speed_mps": ESCAPE_SPEED_MPS,
              "maximum_step_m": MAX_STEP_M,
              "maximum_travel_m": MAX_TRAVEL_M,
              "minimum_endpoint_gap_m": STEP_ENDPOINT_GAP_M,
              "path_end_xy_m": {name: xy.tolist()
                                for name, xy in v50.PATHS.items()},
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
