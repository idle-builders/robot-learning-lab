"""V65: frozen V64 deferred controller under faster/slower final prop turns."""

from __future__ import annotations

import json

import numpy as np

import owner_deferred_multi_surface as v64


runner, v60, v63, v52 = v64.runner, v64.v60, v64.v63, v64.v52
ROOT = v64.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v65-deferred-turn-speed"
TURN_START_S, TURN_START_XY_M = v60.v50.WAYPOINTS[-1]
VARIANTS = (
    ("outbound_fast", "outbound_control", 4.85),
    ("inbound_fast", "inbound_holdout", 4.85),
    ("new_west_south_fast", "new_west_south_turn", 4.85),
    ("far_west_fast", "far_west_turn", 4.85),
    ("far_west_slow", "far_west_turn", 5.55),
)


def prop_at(time_s, checkpoint_s, endpoint, finish_after_s):
    elapsed = time_s-checkpoint_s
    if elapsed <= TURN_START_S:
        return v60.v50.prop_at(time_s, checkpoint_s, endpoint)
    u = float(np.clip((elapsed-TURN_START_S)/
                      (finish_after_s-TURN_START_S), 0., 1.))
    eased = u*u*(3.-2.*u)
    return TURN_START_XY_M+eased*(endpoint-TURN_START_XY_M)


def main():
    source = runner.original()
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    original_geometry, original_packet, original_out = (
        runner.geometry_at, runner.observed_packet, v60.OUT)
    results = []
    try:
        runner.geometry_at = v64.v51.v46.low_prop_geometry(original_geometry)
        runner.observed_packet = v64.v51.v46.plane_packet(
            original_packet, (.05, .30))
        v60.OUT = OUT
        for name, base_path, finish_after_s in VARIANTS:
            endpoint = v64.PATHS[base_path]
            route_fn = lambda t, endpoint=endpoint, finish_after_s=finish_after_s: \
                prop_at(t, checkpoint_s, endpoint, finish_after_s)
            run = v60.run_case(
                source, scenario, name, endpoint, checkpoint_s,
                watch_controller=v63.MultiSurfaceEscape(),
                route_obstacle_fn=route_fn)
            results.append({"variant": name, "base_path": base_path,
                            "finish_after_checkpoint_s": finish_after_s,
                            "run": run})
            print(name, run["minimum_proxy_gap_m"],
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
    report = {"schema": "robot-learning-lab.ape-deferred-turn-speed.v1",
              "source_sha256": v52.sha(__file__),
              "v64_report_sha256": v52.sha(v64.OUT / "report.json"),
              "v60_source_sha256": v52.sha(v60.__file__),
              "v63_source_sha256": v52.sha(v63.__file__),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "turn_start_after_checkpoint_s": TURN_START_S,
              "turn_start_xy_m": TURN_START_XY_M.tolist(),
              "variants": [{"name": name, "base_path": base,
                            "finish_after_checkpoint_s": end}
                           for name, base, end in VARIANTS],
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
