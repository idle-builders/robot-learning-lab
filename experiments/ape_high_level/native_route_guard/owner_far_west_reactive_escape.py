"""V62 frozen V59 meeting choice plus live surface-track response on V61 turn."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import owner_far_west_turn_holdout as v61


v59, v60 = v61.v60.v59, v61.v60
runner, v51, v52, v54 = v59.runner, v59.v51, v59.v52, v59.v54
ROOT = v61.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v62-far-west-reactive-escape"


def main():
    source = runner.original()
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = runner.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(runner.owner.TEACH_CROP).convert(
        "RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    original_geometry, original_packet = runner.geometry_at, runner.observed_packet
    history = []

    def retain(packet, time_s, stage):
        if stage == "post_alarm_range_watch":
            history.append({"time_s": float(time_s),
                            "packet": list(packet)})
        return packet

    try:
        runner.geometry_at = v51.v46.low_prop_geometry(original_geometry)
        runner.observed_packet = v51.v46.plane_packet(
            original_packet, (.05, .30))
        obstacle_fn = lambda t: v59.v50.prop_at(
            t, checkpoint_s, v61.END_XY_M)
        run = runner.run_case(
            v54.CASE, source, policy, scenario, vision, exemplar,
            flank_probe_factory=lambda video:
                v51.v41.LocalizedFlankCamera(video),
            require_camera_pose=True, route_obstacle_fn=obstacle_fn,
            pre_show_restage=v59.ApproachAwareMeetingPose(history),
            owner_fovy_deg=50., range_transform=retain,
            near_body_controller=v52.SurfaceTrackEscape(),
            output_root=OUT / "far_west_turn")
    finally:
        runner.geometry_at, runner.observed_packet = original_geometry, original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    alarm = next((row for row in rows if row["stage"] ==
                  "near_body_escape_plan"), None)
    commanded = sum(row["near_body_command"]["commanded_step_m"]
                    for row in rows if row["stage"] ==
                    "second_inference_range_watch" and
                    row["near_body_command"] is not None)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-far-west-reactive-escape.v1",
              "source_sha256": v52.sha(__file__),
              "v59_source_sha256": v52.sha(v59.__file__),
              "v61_report_sha256": v52.sha(v61.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "path_end_xy_m": v61.END_XY_M.tolist(),
              "run": run, "near_body_alarm_s": (
                  alarm["time_s"] if alarm else None),
              "near_body_commanded_m": commanded}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(run["outcome"], run["minimum_proxy_gap_m"],
          run["minimum_prop_gap_m"], report["near_body_alarm_s"],
          commanded, path, flush=True)


if __name__ == "__main__":
    main()
