"""V53: frozen V52 actor under a faster fixture-clear final prop turn."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import owner_surface_track_escape as v52


v50 = v52.v50
v51 = v52.v51
runner = v52.runner
ROOT = v52.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v53-faster-turn"
END_S = 4.95


def prop_at(time_s, checkpoint_s, end_xy):
    elapsed = np.clip(time_s-checkpoint_s, 0., END_S)
    points = (*v50.WAYPOINTS, (END_S, end_xy))
    for (left_t, left), (right_t, right) in zip(points, points[1:]):
        if elapsed <= right_t:
            u = np.clip((elapsed-left_t)/(right_t-left_t), 0., 1.)
            eased = u*u*(3.-2.*u)
            return left+eased*(right-left)
    return end_xy.copy()


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
        for path_name, end_xy in v50.PATHS.items():
            obstacle_fn = lambda t, end_xy=end_xy: prop_at(
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
                    near_body_controller=v52.SurfaceTrackEscape(),
                    output_root=OUT / path_name / case)
                rows = json.loads(Path(run["trace"]["path"]).read_text())
                commands = [row["near_body_command"] for row in rows
                            if row["stage"] == "second_inference_range_watch"
                            and row["near_body_command"] is not None and
                            row["near_body_command"]["commanded_step_m"] > 0]
                alarm = next((row for row in rows if row["stage"] ==
                              "near_body_escape_plan"), None)
                results.append({"path": path_name, "case": case, "run": run,
                                "near_body_alarm_s": (alarm["time_s"]
                                                      if alarm else None),
                                "near_body_commands": len(commands),
                                "near_body_commanded_m": sum(
                                    row["commanded_step_m"] for row in commands)})
                print(path_name, case, run["outcome"],
                      run["minimum_proxy_gap_m"], run["minimum_prop_gap_m"],
                      results[-1]["near_body_commanded_m"], flush=True)
    finally:
        runner.geometry_at, runner.observed_packet = original_geometry, original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-surface-track-faster-turn.v1",
              "task": "T-9v3p3n", "source_sha256": v52.sha(__file__),
              "v52_source_sha256": v52.sha(v52.__file__),
              "v52_report_sha256": v52.sha(v52.OUT / "report.json"),
              "runner_source_sha256": v52.sha(runner.__file__),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "v36_report_sha256": v52.sha(runner.SOURCE),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "waypoints": [{"after_checkpoint_s": t, "xy_m": xy.tolist()}
                            for t, xy in v50.WAYPOINTS],
              "faster_final_time_after_checkpoint_s": END_S,
              "baseline_final_time_after_checkpoint_s": v50.END_S,
              "path_end_xy_m": {name: xy.tolist()
                                for name, xy in v50.PATHS.items()},
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
