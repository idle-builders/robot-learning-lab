"""Inject low-scan loss into the dual-height moving-prop route.

Two actor responses use the same scan stream: silently fall back to the 30 cm
plane, or stop motion when the low-plane packet is missing. The health flag is
sensor metadata, never simulator geometry or a contact oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_dynamic_suite_v439 import motion


ROOT = Path(__file__).resolve().parents[3]
SUITE = Path(__file__).with_name("robocasa_dynamic_suite_v439.json")
OUT = ROOT / "artifacts/ape-robocasa-low-plane-loss-v441"
BASE_SCAN = route.planar_scan


def configured_scan(drop_from_step: int | None, health: dict):
    step = 0

    def scan(model, data, xy, max_range_m, rng):
        nonlocal step
        route.LIDAR_Z_M = .05
        low = BASE_SCAN(model, data, xy, max_range_m, rng)
        route.LIDAR_Z_M = .30
        mid = BASE_SCAN(model, data, xy, max_range_m, rng)
        healthy = drop_from_step is None or step < drop_from_step
        health["low_packet_valid"] = healthy
        health["step"] = step
        step += 1
        if not healthy:
            return mid
        return [min((x for x in (a, b) if x is not None), default=None)
                for a, b in zip(low, mid)]

    return scan


def health_grid(health: dict):
    class Grid(route.SensorGrid):
        def plan(self, origin_xy, goal_xy):
            if not health["low_packet_valid"]:
                return None, "low_range_unhealthy_stop", []
            return super().plan(origin_xy, goal_xy)

    return Grid


def main() -> None:
    scenario = next(case for case in json.loads(SUITE.read_text())["cases"]
                    if case["id"] == "east_pinch")
    trial.obstacle_xy = motion(scenario)
    trial.START_X, trial.OBSTACLE_Y = scenario["from_xy_m"]
    route.MARGIN_M = .15
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    configs = (("healthy", None, False),
               ("missing_from_start_naive", 0, False),
               ("missing_from_start_gated", 0, True),
               ("missing_after_20_naive", 20, False),
               ("missing_after_20_gated", 20, True),
               ("missing_after_40_naive", 40, False))
    for label, drop_step, gate in configs:
        health = {"low_packet_valid": True, "step": -1}
        trial.planar_scan = configured_scan(drop_step, health)
        trial.SensorGrid = health_grid(health) if gate else route.SensorGrid
        trial.OUT = OUT / label
        episode = trial.episode(True)
        cases.append({"scenario": scenario, "package": label,
                      "drop_low_from_scan_step": drop_step,
                      "health_gate": gate, "episode": episode})
        print(label, episode["outcome"], episode["steps"],
              round(episode["end_time_s"], 2), flush=True)
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    trial.SensorGrid = route.SensorGrid
    trial.planar_scan = BASE_SCAN
    report = {"source_sha256": route.sha256(Path(__file__)),
              "suite_sha256": route.sha256(SUITE),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "scope": "east-pinch moving prop, dual 5+30cm scan, 15cm map margin, low-plane packet loss with naive fallback or health-gated stop",
              "cases": cases}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
