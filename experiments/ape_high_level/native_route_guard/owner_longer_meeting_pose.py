"""V55: development screen of an earlier, farther owner meeting pose.

The actor checks a 60 cm southwest route using its current fused range packet
and every executed step. This is one candidate pose, not an optimized planner;
future prop geometry and owner item pixels are evaluator outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import owner_fixed_clock_pair as v54


runner = v54.runner
v50, v51, v52 = v54.v50, v54.v51, v54.v52
ROOT = v54.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v55-longer-meeting-pose"
TARGET_M = .60
SPEED_MPS = .25
MIN_ENDPOINT_GAP_M = .27
DIRECTION = v51.v47.v44.CANDIDATES["southwest"]


class LongerMeetingPose:
    max_ticks = 30

    def start(self, xy, packet, time_s):
        self.start_xy = np.asarray(xy).copy()
        self.direction = DIRECTION.copy()
        endpoints = v51.v41.JointHazardResponse.endpoints(packet, xy)
        distances = [float(np.min(np.linalg.norm(
            endpoints-(self.start_xy+self.direction*distance), axis=1)))
            for distance in np.linspace(0., TARGET_M, 7)]
        self.initial_corridor_gap_m = min(distances)
        self.choice = ("southwest_60cm" if
                       self.initial_corridor_gap_m >= MIN_ENDPOINT_GAP_M
                       else None)

    def ready(self, xy):
        return self.choice is not None and np.linalg.norm(
            np.asarray(xy)-self.start_xy) >= TARGET_M-.001

    def step(self, time_s, xy, packet):
        if self.choice is None:
            return self.direction, 0., {"action": "hold_corridor_unknown",
                                         "choice": None,
                                         "initial_corridor_gap_m":
                                             self.initial_corridor_gap_m}
        remaining = TARGET_M-np.linalg.norm(np.asarray(xy)-self.start_xy)
        if remaining <= .001:
            return self.direction, 0., {"action": "meeting_pose_reached",
                                         "choice": self.choice}
        if not runner.side_healthy(packet):
            return self.direction, 0., {"action": "hold_scan_unhealthy",
                                         "choice": self.choice}
        distance = min(SPEED_MPS*runner.HOLD_PERIOD_S, remaining)
        endpoints = v51.v41.JointHazardResponse.endpoints(packet, xy)
        nearest = float(np.min(np.linalg.norm(
            endpoints-(np.asarray(xy)+self.direction*distance), axis=1)))
        if nearest < MIN_ENDPOINT_GAP_M:
            return self.direction, 0., {"action": "hold_step_blocked",
                                         "choice": self.choice,
                                         "minimum_endpoint_gap_m": nearest}
        return self.direction, distance, {
            "action": "step_from_fresh_scan", "choice": self.choice,
            "minimum_endpoint_gap_m": nearest,
            "initial_corridor_gap_m": self.initial_corridor_gap_m,
            "packet_time_s": round(time_s-runner.HOLD_PERIOD_S, 5)}


def main():
    source = runner.original()
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = runner.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(runner.owner.TEACH_CROP).convert(
        "RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    original_geometry, original_packet = runner.geometry_at, runner.observed_packet
    results = []
    try:
        runner.geometry_at = v51.v46.low_prop_geometry(original_geometry)
        runner.observed_packet = v51.v46.plane_packet(
            original_packet, (.05, .30))
        for mode, controller_type in (("side_only", None),
                                      ("surface_track", v52.SurfaceTrackEscape)):
            for path_name, end_xy in v50.PATHS.items():
                obstacle_fn = lambda t, end_xy=end_xy: v50.prop_at(
                    t, checkpoint_s, end_xy)
                run = runner.run_case(
                    v54.CASE, source, policy, scenario, vision, exemplar,
                    flank_probe_factory=lambda video:
                        v51.v41.LocalizedFlankCamera(video),
                    require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                    pre_show_restage=LongerMeetingPose(), owner_fovy_deg=50.,
                    near_body_controller=(controller_type()
                                          if controller_type else None),
                    output_root=OUT / mode / path_name)
                rows = json.loads(Path(run["trace"]["path"]).read_text())
                alarm = next((row for row in rows if row["stage"] ==
                              "near_body_escape_plan"), None)
                steps = [row for row in rows if row["stage"] ==
                         "pre_show_restage_step"]
                results.append({"policy": mode, "path": path_name,
                                "run": run, "restage_steps": len(steps),
                                "meeting_xy_m": (steps[-1]["ape_xy_m"]
                                                 if steps else None),
                                "near_body_alarm_s": (alarm["time_s"]
                                                      if alarm else None)})
                print(mode, path_name, run["outcome"],
                      run["minimum_prop_gap_m"],
                      results[-1]["meeting_xy_m"], flush=True)
    finally:
        runner.geometry_at, runner.observed_packet = original_geometry, original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-longer-meeting-pose.v1",
              "task": "T-9v3p3n", "source_sha256": v52.sha(__file__),
              "runner_source_sha256": v52.sha(runner.__file__),
              "v54_report_sha256": v52.sha(v54.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "v36_report_sha256": v52.sha(runner.SOURCE),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "target_meeting_distance_m": TARGET_M,
              "meeting_direction_xy": DIRECTION.tolist(),
              "minimum_endpoint_gap_m": MIN_ENDPOINT_GAP_M,
              "monitor_period_s": runner.SHOW_MONITOR_PERIOD_S,
              "monitor_ticks": runner.FIXED_MONITOR_TICKS,
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
