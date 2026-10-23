"""V57 screen: hold the current owner-view pose through the same three turns.

This is an action counterfactual, not a planner. It uses the same fixed-clock
runner and actor sensors as V56; future prop endpoints are evaluator only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import owner_candidate_meeting_search as v56


runner, v54, v50, v51, v52 = (v56.runner, v56.v54, v56.v50,
                              v56.v51, v56.v52)
ROOT = v56.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v57-stay-meeting-screen"


class StayMeetingPose:
    max_ticks = 1

    def start(self, xy, packet, time_s):
        self.start_xy = np.asarray(xy).copy()
        self.packet_time_s = time_s

    def step(self, time_s, xy, packet):
        return np.zeros(2), 0., {"action": "hold_current_pose_for_owner_view",
                                 "choice": "stay_0cm",
                                 "packet_time_s": self.packet_time_s}

    def ready(self, xy):
        return np.linalg.norm(np.asarray(xy)-self.start_xy) < .001


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
            obstacle_fn = lambda t, end_xy=end_xy: v50.prop_at(
                t, checkpoint_s, end_xy)
            run = runner.run_case(
                v54.CASE, source, policy, scenario, vision, exemplar,
                flank_probe_factory=lambda video:
                    v51.v41.LocalizedFlankCamera(video),
                require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                pre_show_restage=StayMeetingPose(), owner_fovy_deg=50.,
                near_body_controller=v52.SurfaceTrackEscape(),
                output_root=OUT / path_name)
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            candidate = next((row for row in rows if row["stage"] ==
                              "fresh_item_candidate"), None)
            results.append({"path": path_name, "run": run,
                            "item_valid_depth_pixels": (
                                candidate["depth_valid_pixels"]
                                if candidate else None)})
            print(path_name, run["outcome"], run["minimum_prop_gap_m"],
                  results[-1]["item_valid_depth_pixels"], flush=True)
    finally:
        runner.geometry_at, runner.observed_packet = original_geometry, original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-stay-meeting-screen.v1",
              "source_sha256": v52.sha(__file__),
              "runner_source_sha256": v52.sha(runner.__file__),
              "v56_report_sha256": v52.sha(v56.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s, "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
