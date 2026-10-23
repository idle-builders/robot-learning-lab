"""V56: actor-side multi-candidate owner meeting pose, then fresh show.

The planner receives only its resume packet, pose, and the task's remembered
owner-show station. It ranks range-clear routes by held free-space reserve and
stand-off distance within a provisional owner-camera range. The simulated
prop's later finish is never supplied to the planner.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import owner_longer_meeting_pose as v55


v54, v50, v51, v52 = v55.v54, v55.v50, v55.v51, v55.v52
runner = v55.runner
ROOT = v55.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v56-candidate-meeting-search"
OWNER_SHOW_STATION_XY_M = np.array([1., -2.8])
MAX_OWNER_STATION_RANGE_M = 2.40
MIN_HELD_ENDPOINT_RESERVE_M = .65
MIN_PATH_ENDPOINT_GAP_M = .27
DISTANCES_M = (.35, .45, .60, .70)
DIRECTIONS = {"west": np.array([-1., 0.]),
              "southwest": np.array([-math.sqrt(.5), -math.sqrt(.5)]),
              "south": np.array([0., -1.]),
              "southeast": np.array([math.sqrt(.5), -math.sqrt(.5)]),
              "northeast": np.array([math.sqrt(.5), math.sqrt(.5)])}
PATHS = {**v50.PATHS,
         "new_west_south_turn": np.array([1.15, -5.25])}


class CandidateMeetingPose:
    max_ticks = 32

    def start(self, xy, packet, time_s):
        self.start_xy = np.asarray(xy).copy()
        self.scores = []
        endpoints = v51.v41.JointHazardResponse.endpoints(packet, xy)
        healthy = len([value for value in packet if value is not None]) >= 36
        for name, direction in DIRECTIONS.items():
            for distance in DISTANCES_M:
                candidate = self.start_xy+distance*direction
                gaps = [float(np.min(np.linalg.norm(
                    endpoints-(self.start_xy+u*direction), axis=1)))
                    for u in np.linspace(0., distance, 15)]
                owner_range = float(np.linalg.norm(
                    OWNER_SHOW_STATION_XY_M-candidate))
                eligible = (healthy and min(gaps) >= MIN_PATH_ENDPOINT_GAP_M
                            and gaps[-1] >= MIN_HELD_ENDPOINT_RESERVE_M
                            and owner_range <= MAX_OWNER_STATION_RANGE_M)
                self.scores.append({"name": name, "distance_m": distance,
                                    "target_xy_m": candidate.tolist(),
                                    "minimum_path_endpoint_gap_m": min(gaps),
                                    "held_endpoint_reserve_m": gaps[-1],
                                    "owner_station_range_m": owner_range,
                                    "eligible": bool(eligible)})
        eligible = [row for row in self.scores if row["eligible"]]
        chosen = max(eligible, key=lambda row: (
            row["distance_m"], row["held_endpoint_reserve_m"],
            -row["owner_station_range_m"])) if eligible else None
        self.choice = (f"{chosen['name']}_{int(chosen['distance_m']*100)}cm"
                       if chosen else None)
        self.target_m = chosen["distance_m"] if chosen else 0.
        self.direction = DIRECTIONS[chosen["name"]] if chosen else np.zeros(2)

    def ready(self, xy):
        return self.choice is not None and np.linalg.norm(
            np.asarray(xy)-self.start_xy) >= self.target_m-.001

    def step(self, time_s, xy, packet):
        if self.choice is None:
            return self.direction, 0., {"action": "ask_owner_for_safer_show_pose",
                                         "choice": None,
                                         "candidate_scores": self.scores}
        remaining = self.target_m-np.linalg.norm(np.asarray(xy)-self.start_xy)
        if remaining <= .001:
            return self.direction, 0., {"action": "meeting_pose_reached",
                                         "choice": self.choice}
        if not runner.side_healthy(packet):
            return self.direction, 0., {"action": "hold_scan_unhealthy",
                                         "choice": self.choice}
        distance = min(v55.SPEED_MPS*runner.HOLD_PERIOD_S, remaining)
        endpoints = v51.v41.JointHazardResponse.endpoints(packet, xy)
        nearest = float(np.min(np.linalg.norm(
            endpoints-(np.asarray(xy)+self.direction*distance), axis=1)))
        if nearest < MIN_PATH_ENDPOINT_GAP_M:
            return self.direction, 0., {"action": "hold_step_blocked",
                                         "choice": self.choice,
                                         "minimum_endpoint_gap_m": nearest}
        return self.direction, distance, {
            "action": "step_from_fresh_scan", "choice": self.choice,
            "candidate_scores": self.scores if time_s <= 21.5 else None,
            "minimum_endpoint_gap_m": nearest,
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
                                      ("nearest_always", v51.FullAngleEscape),
                                      ("surface_track", v52.SurfaceTrackEscape)):
            for path_name, end_xy in PATHS.items():
                obstacle_fn = lambda t, end_xy=end_xy: v50.prop_at(
                    t, checkpoint_s, end_xy)
                run = runner.run_case(
                    v54.CASE, source, policy, scenario, vision, exemplar,
                    flank_probe_factory=lambda video:
                        v51.v41.LocalizedFlankCamera(video),
                    require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                    pre_show_restage=CandidateMeetingPose(),
                    owner_fovy_deg=50.,
                    near_body_controller=(controller_type()
                                          if controller_type else None),
                    output_root=OUT / mode / path_name)
                rows = json.loads(Path(run["trace"]["path"]).read_text())
                steps = [row for row in rows if row["stage"] ==
                         "pre_show_restage_step"]
                alarm = next((row for row in rows if row["stage"] ==
                              "near_body_escape_plan"), None)
                candidate = next((row for row in rows if row["stage"] ==
                                  "fresh_item_candidate"), None)
                results.append({"policy": mode, "path": path_name, "run": run,
                                "choice": (steps[0]["actor_decision"]["choice"]
                                           if steps else None),
                                "restage_steps": len(steps),
                                "meeting_xy_m": (steps[-1]["ape_xy_m"]
                                                 if steps else None),
                                "near_body_alarm_s": (alarm["time_s"]
                                                      if alarm else None),
                                "item_valid_depth_pixels": (candidate[
                                    "depth_valid_pixels"] if candidate else None)})
                print(mode, path_name, results[-1]["choice"],
                      run["outcome"], run["minimum_prop_gap_m"], flush=True)
    finally:
        runner.geometry_at, runner.observed_packet = original_geometry, original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-candidate-meeting-search.v1",
              "task": "T-9v3p3n", "source_sha256": v52.sha(__file__),
              "runner_source_sha256": v52.sha(runner.__file__),
              "v55_report_sha256": v52.sha(v55.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "v36_report_sha256": v52.sha(runner.SOURCE),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "known_owner_show_station_xy_m":
                  OWNER_SHOW_STATION_XY_M.tolist(),
              "maximum_owner_station_range_m": MAX_OWNER_STATION_RANGE_M,
              "minimum_held_endpoint_reserve_m": MIN_HELD_ENDPOINT_RESERVE_M,
              "minimum_path_endpoint_gap_m": MIN_PATH_ENDPOINT_GAP_M,
              "candidate_distances_m": list(DISTANCES_M),
              "candidate_directions": {name: direction.tolist()
                                       for name, direction in DIRECTIONS.items()},
              "path_end_xy_m": {name: xy.tolist() for name, xy in PATHS.items()},
              "monitor_period_s": runner.SHOW_MONITOR_PERIOD_S,
              "monitor_ticks": runner.FIXED_MONITOR_TICKS,
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
