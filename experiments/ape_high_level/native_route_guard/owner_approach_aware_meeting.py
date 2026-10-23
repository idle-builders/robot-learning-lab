"""V59: score meeting poses against a timed anonymous approach corridor.

The actor gets only the same 10 Hz range history, current pose, and remembered
owner station as V56. Returns newly revealed in the north sector become an
uncertain southbound occupancy tube. The tube is a provisional risk ranking,
not a collision guarantee; the later prop endpoint remains evaluator-only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import owner_candidate_meeting_search as v56


runner, v54, v50, v51, v52 = (v56.runner, v56.v54, v56.v50,
                              v56.v51, v56.v52)
ROOT = v56.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v59-approach-aware-meeting"
MIN_ENDPOINT_GAP_M = .27
MAX_OWNER_STATION_RANGE_M = 2.60
BODY_PROXY_RADIUS_M = .17
APPROACH_LATERAL_UNCERTAINTY_M = .40
APPROACH_SPEED_UPPER_MPS = 1.20
SHOW_INTERVAL_S = 3.50
NOVEL_RANGE_M = .50
FAR_SURFACE_M = 1.50
NORTH_RAY_INDEXES = range(45, 62)


def dynamic_surface_points(history, pose):
    first = history[0]["packet"]
    last = history[-1]["packet"]
    points = []
    for index in NORTH_RAY_INDEXES:
        old, now = first[index], last[index]
        if old is None or now is None or now <= FAR_SURFACE_M or \
                old-now < NOVEL_RANGE_M:
            continue
        angle = math.radians(-180.+360.*index/len(last))
        points.append({"ray_index": index, "first_range_m": old,
                       "last_range_m": now,
                       "surface_xy_m": (np.asarray(pose)+now*np.array(
                           [math.cos(angle), math.sin(angle)])).tolist()})
    return points


def time_indexed_surface_margin(start, target, distance, surface):
    """Minimum center-to-uncertain-tube gap during move and 3.5 s hold."""
    stop_s = distance/v56.v55.SPEED_MPS+SHOW_INTERVAL_S
    margins = []
    for t in np.arange(0., stop_s+.05, .10):
        fraction = min(t*v56.v55.SPEED_MPS/distance, 1.)
        xy = start+fraction*(target-start)
        x, y = surface
        # This ray revealed an approaching surface to the north. Its possible
        # southward travel grows with time; the unknown turn is a lateral band.
        dx = max(abs(xy[0]-x)-APPROACH_LATERAL_UNCERTAINTY_M, 0.)
        dy = max(y-xy[1]-APPROACH_SPEED_UPPER_MPS*t,
                 xy[1]-y, 0.)
        margins.append(math.hypot(dx, dy)-BODY_PROXY_RADIUS_M)
    return min(margins)


class ApproachAwareMeetingPose(v56.CandidateMeetingPose):
    def __init__(self, history):
        self.history = history

    def start(self, xy, packet, time_s):
        super().start(xy, packet, time_s)
        self.dynamic_surfaces = dynamic_surface_points(self.history, xy)
        endpoints = v51.v41.JointHazardResponse.endpoints(packet, xy)
        ranked = []
        for row in self.scores:
            target = np.asarray(row["target_xy_m"])
            static_margin = row["held_endpoint_reserve_m"]-BODY_PROXY_RADIUS_M
            approach_margin = min((time_indexed_surface_margin(
                self.start_xy, target, row["distance_m"],
                np.asarray(point["surface_xy_m"]))
                for point in self.dynamic_surfaces), default=float("inf"))
            row["actor_static_margin_m"] = static_margin
            row["actor_approach_margin_m"] = approach_margin
            row["actor_combined_margin_m"] = min(static_margin,
                                                  approach_margin)
            row["eligible"] = bool(
                len([x for x in packet if x is not None]) >= 36 and
                row["minimum_path_endpoint_gap_m"] >= MIN_ENDPOINT_GAP_M and
                row["held_endpoint_reserve_m"] >= MIN_ENDPOINT_GAP_M and
                row["owner_station_range_m"] <= MAX_OWNER_STATION_RANGE_M)
            if row["eligible"]:
                ranked.append(row)
        chosen = max(ranked, key=lambda row: (
            row["actor_combined_margin_m"], -row["distance_m"])) \
            if ranked and self.dynamic_surfaces else None
        self.choice = (f"{chosen['name']}_{int(chosen['distance_m']*100)}cm"
                       if chosen else None)
        self.target_m = chosen["distance_m"] if chosen else 0.
        self.direction = v56.DIRECTIONS[chosen["name"]] if chosen else np.zeros(2)
        self.choice_score = chosen["actor_combined_margin_m"] if chosen else None

    def step(self, time_s, xy, packet):
        direction, distance, decision = super().step(time_s, xy, packet)
        decision["dynamic_surfaces_from_hold"] = self.dynamic_surfaces
        decision["time_indexed_choice_margin_m"] = self.choice_score
        return direction, distance, decision


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
        for path_name, end_xy in v56.PATHS.items():
            history = []

            def retain(packet, time_s, stage):
                if stage == "post_alarm_range_watch":
                    history.append({"time_s": float(time_s),
                                    "packet": list(packet)})
                return packet

            obstacle_fn = lambda t, end_xy=end_xy: v50.prop_at(
                t, checkpoint_s, end_xy)
            run = runner.run_case(
                v54.CASE, source, policy, scenario, vision, exemplar,
                flank_probe_factory=lambda video:
                    v51.v41.LocalizedFlankCamera(video),
                require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                pre_show_restage=ApproachAwareMeetingPose(history),
                owner_fovy_deg=50., range_transform=retain,
                near_body_controller=v52.SurfaceTrackEscape(),
                output_root=OUT / path_name)
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            first = next((row for row in rows if row["stage"] ==
                          "pre_show_restage_step"), None)
            candidate = next((row for row in rows if row["stage"] ==
                              "fresh_item_candidate"), None)
            results.append({"path": path_name, "run": run,
                            "choice": (first["actor_decision"]["choice"]
                                       if first else None),
                            "item_valid_depth_pixels": (
                                candidate["depth_valid_pixels"]
                                if candidate else None)})
            print(path_name, results[-1]["choice"], run["outcome"],
                  run["minimum_proxy_gap_m"],
                  run["minimum_prop_gap_m"], flush=True)
    finally:
        runner.geometry_at, runner.observed_packet = original_geometry, original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-approach-aware-meeting.v1",
              "source_sha256": v52.sha(__file__),
              "runner_source_sha256": v52.sha(runner.__file__),
              "v56_report_sha256": v52.sha(v56.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "actor_tube": {"lateral_uncertainty_m":
                             APPROACH_LATERAL_UNCERTAINTY_M,
                             "speed_upper_mps": APPROACH_SPEED_UPPER_MPS,
                             "show_interval_s": SHOW_INTERVAL_S},
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
