"""Frozen changed-speed holdout for the V10 command-route kitchen guard."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import command_route_trial as command
import initial_view_trial as initial
import kitchen_native as kitchen
import kitchen_trial as trial
import run as guard


OUT = kitchen.OUT / "v11-changed-speed"
CASES = ("fast_clear_held", "slow_cross_held", "pause_path_held")
ORIGINAL_HAZARD_XY = trial.hazard_xy


def hazard_xy(case: str, t: float) -> tuple[float, float]:
    y = {
        "fast_clear_held": -1.05-.72*min(t, 4.0),
        "slow_cross_held": -1.05-.22*min(t, 4.0),
        "pause_path_held": -1.05-.50*min(t, 1.5),
    }[case]
    return 3.55, y


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    background = np.load(initial.OUT / "background_xy.npy")
    map_path = OUT / "background_xy.npy"
    np.save(map_path, background)
    stop = json.loads(guard.replay.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop["rows"] if r["case"] == "forward-stop-late"}
    trial.OUT = OUT
    trial.hazard_xy = hazard_xy
    guard.FullScanGuard = command.CommandRouteGuard
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            results = []
            for world in range(4):
                for case in CASES:
                    for mode in trial.MODES:
                        video = ((world == 0 and (mode == "mapped_low_guard" or
                                                 case == "slow_cross_held")) or
                                 (world in (1, 3) and case == "fast_clear_held" and
                                  mode == "mapped_low_guard") or
                                 (world == 1 and case == "fast_clear_held" and
                                  mode == "none"))
                        result = trial.run_episode(model, saved, background,
                                                   world, case, mode,
                                                   delays[world], video=video)
                        results.append(result)
                        print(world, case, mode, result["stop_command_s"],
                              result["min_modeled_body_person_gap_m"], flush=True)
    finally:
        trial.hazard_xy = ORIGINAL_HAZARD_XY
        guard.FullScanGuard = command.OriginalGuard
    report = {"schema": "robot-learning-lab.ape-robocasa-command-route-holdout.v1",
              "task": "T-9v3p3n",
              "body_status": "historical_rejected_direct_frame",
              "method": "Frozen V10 commanded-route guard, same initial-view map and native body. Changed person speeds and pause timing were set before this run; paired no-sensor controls use the same scripts.",
              "case_definitions": {
                  "fast_clear_held": {"start_xy_m": [3.55, -1.05], "vy_mps": -.72},
                  "slow_cross_held": {"start_xy_m": [3.55, -1.05], "vy_mps": -.22},
                  "pause_path_held": {"start_xy_m": [3.55, -1.05],
                                      "vy_mps": -.50, "pause_at_s": 1.5}},
              "room": {"scene_sha256": trial.sha(kitchen.OUT / "scene.xml")},
              "sensor": {"name": "low", "period_s": 1/15,
                         "static_map_points": len(background),
                         "prior_map_sha256": trial.sha(map_path),
                         "ideal_odom": True},
              "source_sha256": trial.sha(Path(__file__)),
              "episode_runner_sha256": trial.sha(Path(trial.__file__)),
              "command_guard_sha256": trial.sha(Path(command.__file__)),
              "original_guard_sha256": trial.sha(Path(guard.__file__)),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "body_sha256": trial.sha(kitchen.BODY),
              "v10_report_sha256": trial.sha(command.OUT / "report.json"),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
