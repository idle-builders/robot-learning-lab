"""V64: keep the V63 independent multi-surface response live while deferred."""

from __future__ import annotations

import json

import owner_deferred_show as v60
import owner_multi_surface_escape as v63


runner, v51, v52, v54 = v60.runner, v60.v51, v60.v52, v60.v54
ROOT = v60.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v64-deferred-multi-surface"
PATHS = {**v60.v59.v56.PATHS,
         "far_west_turn": v63.v61.END_XY_M}


def main():
    source = runner.original()
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    original_geometry, original_packet, original_out = (
        runner.geometry_at, runner.observed_packet, v60.OUT)
    results = []
    try:
        runner.geometry_at = v51.v46.low_prop_geometry(original_geometry)
        runner.observed_packet = v51.v46.plane_packet(
            original_packet, (.05, .30))
        v60.OUT = OUT
        for path_name, end_xy in PATHS.items():
            run = v60.run_case(
                source, scenario, path_name, end_xy, checkpoint_s,
                watch_controller=v63.MultiSurfaceEscape())
            results.append({"path": path_name, "run": run})
            print(path_name, run["outcome"],
                  run["minimum_proxy_gap_m"],
                  run["minimum_prop_gap_m"],
                  run["near_body_alarm_s"],
                  run["watch_commanded_m"], flush=True)
    finally:
        runner.geometry_at = original_geometry
        runner.observed_packet = original_packet
        v60.OUT = original_out
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-deferred-multi-surface.v1",
              "source_sha256": v52.sha(__file__),
              "v60_source_sha256": v52.sha(v60.__file__),
              "v63_source_sha256": v52.sha(v63.__file__),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "path_end_xy_m": {name: end.tolist()
                                for name, end in PATHS.items()},
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
