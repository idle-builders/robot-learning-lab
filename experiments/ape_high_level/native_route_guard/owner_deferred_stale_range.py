"""V68: freeze a few full range packets during the far-west prop turn.

The actor is given no freshness flag. It timestamps the repeated values as
though they came from new 10 Hz scans. This probes the temporal assumption in
the unchanged V64 multi-surface/defer controller, not a new response policy.
"""

from __future__ import annotations

import json

import owner_deferred_multi_surface as v64


runner, v60, v63, v52 = v64.runner, v64.v60, v64.v63, v64.v52
ROOT = v64.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v68-deferred-stale-range"
STALE_TICKS = tuple(range(1007, 1013))  # 24.41–24.91 s, around first turn
PATHS = ("outbound_control", "far_west_turn")


def stale_scanner(scanner):
    last = None

    def observed(model, data, actual, case, tick, *, person_xy=None,
                 hidden_return=False):
        nonlocal last
        fresh = scanner(model, data, actual, case, tick,
                        person_xy=person_xy,
                        hidden_return=hidden_return)
        if tick in STALE_TICKS:
            assert last is not None
            return list(last)
        last = list(fresh)
        return fresh

    return observed


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
        fused = v64.v51.v46.plane_packet(original_packet, (.05, .30))
        v60.OUT = OUT
        for path in PATHS:
            runner.observed_packet = stale_scanner(fused)
            run = v60.run_case(source, scenario, path, v64.PATHS[path],
                               checkpoint_s,
                               watch_controller=v63.MultiSurfaceEscape())
            results.append({"path": path, "run": run})
            print(path, run["minimum_proxy_gap_m"],
                  run["minimum_prop_gap_m"], run["near_body_alarm_s"],
                  run["watch_commanded_m"], flush=True)
    finally:
        runner.geometry_at = original_geometry
        runner.observed_packet = original_packet
        v60.OUT = original_out
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-deferred-stale-range.v1",
              "source_sha256": v52.sha(__file__),
              "v64_report_sha256": v52.sha(v64.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "stale_watch_ticks": list(STALE_TICKS),
              "stale_packet_age_at_end_s": .6,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
