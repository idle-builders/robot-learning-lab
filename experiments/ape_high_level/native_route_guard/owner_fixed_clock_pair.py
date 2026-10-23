"""V54 matched-clock owner episodes for three near-body policies."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import owner_fixture_clear_prop as v50
import owner_full_angle_escape as v51
import owner_surface_track_escape as v52
import owner_fixed_clock_runner as runner


ROOT = v50.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v54-fixed-clock-pair"
POLICIES = {"side_only": None,
            "nearest_always": v51.FullAngleEscape,
            "surface_track": v52.SurfaceTrackEscape}
CASE = "parked_after_clear"


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
        for name, controller_type in POLICIES.items():
            for path_name, end_xy in v50.PATHS.items():
                history = []

                def retain_actor_packet(packet, time_s, stage):
                    if stage == "post_alarm_range_watch":
                        history.append({"time_s": float(time_s),
                                        "stage": stage, "packet": list(packet)})
                    return packet

                obstacle_fn = lambda t, end_xy=end_xy: v50.prop_at(
                    t, checkpoint_s, end_xy)
                run = runner.run_case(
                    CASE, source, policy, scenario, vision, exemplar,
                    flank_probe_factory=lambda video:
                        v51.v41.LocalizedFlankCamera(video),
                    require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                    pre_show_restage=v51.v47.HistoryMeetingPose(history),
                    owner_fovy_deg=50., range_transform=retain_actor_packet,
                    near_body_controller=(controller_type()
                                          if controller_type else None),
                    output_root=OUT / name / path_name)
                rows = json.loads(Path(run["trace"]["path"]).read_text())
                alarm = next((row for row in rows if row["stage"] ==
                              "near_body_escape_plan"), None)
                commands = [row["near_body_command"] for row in rows
                            if row["stage"] == "second_inference_range_watch"
                            and row["near_body_command"] is not None and
                            row["near_body_command"]["commanded_step_m"] > 0]
                results.append({"policy": name, "path": path_name,
                                "case": CASE, "run": run,
                                "near_body_alarm_s": (alarm["time_s"]
                                                      if alarm else None),
                                "near_body_commands": len(commands),
                                "near_body_commanded_m": sum(
                                    row["commanded_step_m"] for row in commands)})
                print(name, path_name, run["outcome"],
                      run["minimum_prop_gap_m"],
                      results[-1]["near_body_alarm_s"], flush=True)
    finally:
        runner.geometry_at, runner.observed_packet = original_geometry, original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-fixed-clock-pair.v1",
              "task": "T-9v3p3n", "source_sha256": v52.sha(__file__),
              "runner_source_sha256": v52.sha(runner.__file__),
              "v50_report_sha256": v52.sha(v50.OUT / "report.json"),
              "v51_report_sha256": v52.sha(v51.OUT / "report.json"),
              "v52_report_sha256": v52.sha(v52.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "v36_report_sha256": v52.sha(runner.SOURCE),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "identical_path_through_s": round(
                  checkpoint_s+v50.WAYPOINTS[-1][0], 5),
              "monitor_period_s": runner.SHOW_MONITOR_PERIOD_S,
              "monitor_ticks": runner.FIXED_MONITOR_TICKS,
              "monitor_duration_s": ((runner.FIXED_MONITOR_TICKS-1)*
                                     runner.SHOW_MONITOR_PERIOD_S),
              "sensor_heights_m": [.05, .30],
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
