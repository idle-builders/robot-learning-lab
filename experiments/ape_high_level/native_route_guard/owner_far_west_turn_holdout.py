"""V61 frozen V60 actor policy on a fixture-clear farther-west final turn."""

from __future__ import annotations

import json

import numpy as np

import owner_deferred_show as v60


ROOT = v60.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v61-far-west-turn-holdout"
END_XY_M = np.array([.70, -5.15])


def main():
    source = v60.runner.original()
    policy = json.loads(v60.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    original_geometry, original_packet, original_out = (
        v60.runner.geometry_at, v60.runner.observed_packet, v60.OUT)
    try:
        v60.runner.geometry_at = v60.v51.v46.low_prop_geometry(original_geometry)
        v60.runner.observed_packet = v60.v51.v46.plane_packet(
            original_packet, (.05, .30))
        v60.OUT = OUT
        run = v60.run_case(source, scenario, "far_west_turn",
                           END_XY_M, checkpoint_s)
    finally:
        v60.runner.geometry_at = original_geometry
        v60.runner.observed_packet = original_packet
        v60.OUT = original_out
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-far-west-turn-holdout.v1",
              "source_sha256": v60.v52.sha(__file__),
              "v60_source_sha256": v60.v52.sha(v60.__file__),
              "v60_report_sha256": v60.v52.sha(v60.OUT / "report.json"),
              "scene_sha256": v60.v52.sha(v60.runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v60.v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "path_end_xy_m": END_XY_M.tolist(),
              "run": run}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(run["outcome"], run["minimum_proxy_gap_m"],
          run["minimum_prop_gap_m"], path, flush=True)


if __name__ == "__main__":
    main()
