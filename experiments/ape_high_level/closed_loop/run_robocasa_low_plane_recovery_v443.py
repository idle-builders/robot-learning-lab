"""Try bounded stationary reobservation after low-plane packet loss.

The planner issues a zero-translation hold for at most two missing low-plane
captures. Motion resumes only after a fresh low scan; a third missing capture
ends the skill with an unhealthy-sensor stop. No evaluator geometry is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_dynamic_suite_v439 import motion


ROOT = Path(__file__).resolve().parents[3]
SUITE = Path(__file__).with_name("robocasa_dynamic_suite_v439.json")
OUT = ROOT / "artifacts/ape-robocasa-low-plane-recovery-v443"
BASE_SCAN = route.planar_scan


def range_stream(missing_from: int, missing_through: int | None, health: dict):
    step = 0

    def scan(model, data, xy, max_range_m, rng):
        nonlocal step
        route.LIDAR_Z_M = .05
        low = BASE_SCAN(model, data, xy, max_range_m, rng)
        route.LIDAR_Z_M = .30
        mid = BASE_SCAN(model, data, xy, max_range_m, rng)
        good = not (step >= missing_from and
                    (missing_through is None or step < missing_through))
        health["low_packet_valid"] = good
        health["scan_step"] = step
        step += 1
        if not good:
            return mid
        return [min((x for x in (a, b) if x is not None), default=None)
                for a, b in zip(low, mid)]

    return scan


def recovery_grid(health: dict):
    class Grid(route.SensorGrid):
        def __init__(self):
            super().__init__()
            self.missing_streak = 0

        def plan(self, origin_xy, goal_xy):
            if health["low_packet_valid"]:
                self.missing_streak = 0
                result = super().plan(origin_xy, goal_xy)
            else:
                self.missing_streak += 1
                if self.missing_streak <= 2:
                    result = (origin_xy.copy(), "hold_for_fresh_low_scan",
                              [self.cell(origin_xy)])
                else:
                    result = (None, "low_range_unhealthy_stop", [])
            health["decisions"].append({"scan_step": health["scan_step"],
                                        "low_packet_valid": health["low_packet_valid"],
                                        "missing_streak": self.missing_streak,
                                        "plan_status": result[1]})
            return result

    return Grid


def main() -> None:
    scenario = next(case for case in json.loads(SUITE.read_text())["cases"]
                    if case["id"] == "east_pinch")
    trial.obstacle_xy = motion(scenario)
    trial.START_X, trial.OBSTACLE_Y = scenario["from_xy_m"]
    route.MARGIN_M = .15
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for label, first, end in (("one_missing_at_20", 20, 21),
                              ("two_missing_at_20", 20, 22),
                              ("persistent_from_20", 20, None)):
        health = {"low_packet_valid": True, "scan_step": -1, "decisions": []}
        trial.planar_scan = range_stream(first, end, health)
        trial.SensorGrid = recovery_grid(health)
        trial.OUT = OUT / label
        episode = trial.episode(True)
        holds = [row for row in episode["trace"]
                 if row["plan_status"] == "hold_for_fresh_low_scan"]
        cases.append({"scenario": scenario, "package": label,
                      "missing_from_scan_step": first,
                      "missing_through_scan_step_exclusive": end,
                      "hold_count": len(holds),
                      "sensor_health_trace": health["decisions"],
                      "episode": episode})
        print(label, episode["outcome"], episode["steps"],
              "holds", len(holds), round(episode["end_time_s"], 2), flush=True)
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    trial.SensorGrid = route.SensorGrid
    trial.planar_scan = BASE_SCAN
    report = {"source_sha256": route.sha256(Path(__file__)),
              "suite_sha256": route.sha256(SUITE),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "scope": "stationary low-scan retry after one/two/persistent missing packets; 15cm map margin and moving east-pinch prop",
              "cases": cases}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
