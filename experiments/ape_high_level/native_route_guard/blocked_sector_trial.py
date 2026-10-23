"""V14: explicit left-sector loss on the V13 dense low-scan kitchen holdout.

The fault removes the side from which the scripted person enters. The actor
gets a per-ray health mask; the evaluator alone uses person pose and geometry.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import changed_speed_trial as holdout
import command_route_trial as command
import dense_low_trial as dense
import kitchen_native as kitchen
import kitchen_trial as trial
import run as guard


OUT = kitchen.OUT / "v14-left-sector-loss"
RESPONSES = ("ignore_health", "hold_on_unavailable")
ORIGINAL_READ = guard.ORIGINAL_READ
ORIGINAL_POINTS = guard.low_ray_points
ORIGINAL_HAZARD = trial.hazard_xy


def blocked_read(model, data, name, pose, case, t) -> dict:
    packet = dense.dense_read(model, data, name, pose, case, t)
    healthy = [bool(bearing <= 20.) for bearing in dense.YAW_DEG]
    packet["ray_health"] = healthy
    packet["unavailable_sector_deg"] = [20., 70.]
    packet["fault"] = "left_mount_sector_occluded"
    packet["ranges_m"] = [value if okay else None
                          for value, okay in zip(packet["ranges_m"], healthy)]
    packet["valid_returns"] = sum(value is not None for value in packet["ranges_m"])
    forward = [distance for distance, bearing in zip(packet["ranges_m"], dense.YAW_DEG)
               if distance is not None and abs(bearing) <= 25.]
    packet["min_forward_range_m"] = min(forward) if forward else None
    return packet


class CoverageHoldGuard:
    def __init__(self, background: np.ndarray):
        self.inner = command.CommandRouteGuard(background)
        self.background = self.inner.background

    def update(self, packet: dict, pose: np.ndarray, t: float) -> dict:
        decision = self.inner.update(packet, pose, t)
        decision["range_track_risk"] = decision["risk"]
        decision["coverage_unavailable"] = not all(packet["ray_health"])
        if decision["coverage_unavailable"]:
            decision["risk"] = True
            decision["reason"] = "incoming_side_range_unavailable"
        return decision


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    background = np.load(dense.OUT / "background_xy.npy")
    map_path = OUT / "background_xy.npy"
    np.save(map_path, background)
    stop = json.loads(guard.replay.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop["rows"] if r["case"] == "forward-stop-late"}
    trial.hazard_xy = holdout.hazard_xy
    guard.ORIGINAL_READ = blocked_read
    guard.low_ray_points = dense.dense_points
    results = []
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            for response in RESPONSES:
                guard.FullScanGuard = (command.CommandRouteGuard if response == "ignore_health"
                                       else CoverageHoldGuard)
                trial.OUT = OUT / response
                for world in range(4):
                    for case in holdout.CASES:
                        video = ((world == 0 and case == "slow_cross_held") or
                                 (world == 1 and case == "fast_clear_held"))
                        result = trial.run_episode(model, saved, background,
                                                   world, case, "mapped_low_guard",
                                                   delays[world], video=video)
                        result["response"] = response
                        results.append(result)
                        print(response, world, case, result["stop_command_s"],
                              result["min_modeled_body_person_gap_m"], flush=True)
    finally:
        trial.hazard_xy = ORIGINAL_HAZARD
        guard.ORIGINAL_READ = ORIGINAL_READ
        guard.low_ray_points = ORIGINAL_POINTS
        guard.FullScanGuard = command.OriginalGuard
    report = {"schema": "robot-learning-lab.ape-robocasa-low-sector-loss.v1",
              "task": "T-9v3p3n",
              "body_status": "historical_rejected_direct_frame",
              "method": "V13 dense low scan, V11 crossing scripts, native stop-tail and initial-view map, with all >20deg left rays explicitly unavailable for the live actor. Compare unchanged command-route tracking to a two-packet stop on unhealthy incoming-side coverage. No physical sensor fault distribution is claimed.",
              "fault": {"kind": "blocked_left_sector", "yaw_deg": [20., 70.],
                        "starts_at_s": 0., "ends_at_s": 8.},
              "sensor": {"rays": 73, "period_s": 1/15, "max_range_m": 1.2,
                         "prior_map_sha256": trial.sha(map_path),
                         "static_map_points": len(background)},
              "scene_sha256": trial.sha(kitchen.OUT / "scene.xml"),
              "source_sha256": trial.sha(Path(__file__)),
              "episode_runner_sha256": trial.sha(Path(trial.__file__)),
              "dense_sensor_sha256": trial.sha(Path(dense.__file__)),
              "command_guard_sha256": trial.sha(Path(command.__file__)),
              "original_guard_sha256": trial.sha(Path(guard.__file__)),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "body_sha256": trial.sha(kitchen.BODY),
              "v13_report_sha256": trial.sha(dense.OUT / "report.json"),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
