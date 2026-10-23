"""V66: low-only versus mid-only monitor packets after identical dual-plane retreat."""

from __future__ import annotations

import json

import owner_deferred_turn_speed_stress as v65


v64, v60, v63 = v65.v64, v65.v60, v65.v63
runner, v51, v52, v54 = v64.runner, v64.v51, v64.v52, v64.v54
ROOT = v64.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v66-deferred-range-height"
CASES = (
    ("outbound_control", 5.20),
    ("far_west_turn", 5.20),
    ("far_west_turn", 4.85),
)
PROFILES = {"low_05cm": (.05,), "mid_30cm": (.30,)}


def main():
    source = runner.original()
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    original_geometry, original_packet, original_out = (
        runner.geometry_at, runner.observed_packet, v60.OUT)
    fused = v51.v46.plane_packet(original_packet, (.05, .30))
    scanners = {name: v51.v46.plane_packet(original_packet, heights)
                for name, heights in PROFILES.items()}
    results = []
    try:
        runner.geometry_at = v51.v46.low_prop_geometry(original_geometry)
        v60.OUT = OUT
        for profile_name, heights in PROFILES.items():
            def observed(model, data, actual, case, tick, *, person_xy=None,
                         hidden_return=False):
                scanner = fused if tick < 1000 else scanners[profile_name]
                return scanner(model, data, actual, case, tick,
                               person_xy=person_xy,
                               hidden_return=hidden_return)

            runner.observed_packet = observed
            for base_path, finish_after_s in CASES:
                name = f"{profile_name}-{base_path}-{finish_after_s:.2f}s"
                endpoint = v64.PATHS[base_path]
                route_fn = lambda t, endpoint=endpoint, finish_after_s=finish_after_s: \
                    v65.prop_at(t, checkpoint_s, endpoint, finish_after_s)
                run = v60.run_case(
                    source, scenario, name, endpoint, checkpoint_s,
                    watch_controller=v63.MultiSurfaceEscape(),
                    route_obstacle_fn=route_fn)
                results.append({"variant": name, "profile": profile_name,
                                "monitor_heights_m": list(heights),
                                "base_path": base_path,
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
    report = {"schema": "robot-learning-lab.ape-deferred-range-height.v1",
              "source_sha256": v52.sha(__file__),
              "v64_report_sha256": v52.sha(v64.OUT / "report.json"),
              "v65_report_sha256": v52.sha(v65.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "planning_sensor_heights_m": [.05, .30],
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
