"""V18: one healthy wide front low scanner on V17's right-entry cases."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import command_route_trial as command
import dense_low_trial as dense
import kitchen_native as kitchen
import kitchen_trial as trial
import right_entry_trial as right
import run as guard


OUT = kitchen.OUT / "v18-single-front"
MAP_MODES = ("own_initial_view", "shared_three_fan_initial_view")
ORIGINAL_READ = guard.ORIGINAL_READ
ORIGINAL_POINTS = guard.low_ray_points
ORIGINAL_HAZARD = trial.hazard_xy


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    stop = json.loads(guard.replay.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop["rows"] if r["case"] == "forward-stop-late"}
    trial.hazard_xy = right.hazard_xy
    guard.ORIGINAL_READ = dense.dense_read
    guard.low_ray_points = dense.dense_points
    guard.FullScanGuard = command.CommandRouteGuard
    results = []
    maps = {}
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            own = dense.initial_map(model, saved)
            shared = np.load(right.OUT / "background_xy.npy")
            for mode, background in ((MAP_MODES[0], own), (MAP_MODES[1], shared)):
                trial.OUT = OUT / mode
                trial.OUT.mkdir(parents=True, exist_ok=True)
                path = trial.OUT / "background_xy.npy"
                np.save(path, background)
                maps[mode] = {"points": len(background), "sha256": trial.sha(path)}
                for world in range(4):
                    for case in right.CASES:
                        result = trial.run_episode(model, saved, background,
                                                   world, case, "mapped_low_guard",
                                                   delays[world],
                                                   video=(world == 0 and case in
                                                          ("fast_right_held",
                                                           "slow_right_held")))
                        result["map_mode"] = mode
                        results.append(result)
                        print(mode, world, case, result["stop_command_s"],
                              result["min_modeled_body_person_gap_m"], flush=True)
    finally:
        trial.hazard_xy = ORIGINAL_HAZARD
        guard.ORIGINAL_READ = ORIGINAL_READ
        guard.low_ray_points = ORIGINAL_POINTS
        guard.FullScanGuard = command.OriginalGuard
    report = {
        "schema": "robot-learning-lab.ape-robocasa-single-front-right-entry.v1",
        "task": "T-9v3p3n",
        "body_status": "historical_rejected_direct_frame",
        "method": "Exact final V17 right-entry scripts, scene and native trajectories; healthy single 73-ray front fan at 15Hz/1.2m. One mode maps from its own hazard-free initial views; one receives V17's shared three-fan initial map as a map advantage. Same command-route guard and historical stop-tail.",
        "maps": maps,
        "sensor": {"period_s": 1/15, "rays": 73, "max_range_m": dense.MAX_M,
                   "origin_local_m": [.08, 0., -.15],
                   "yaw_range_deg": [-70., 70.], "ideal_odom": True},
        "scene_sha256": trial.sha(kitchen.OUT / "scene.xml"),
        "source_sha256": trial.sha(Path(__file__)),
        "dense_source_sha256": trial.sha(Path(dense.__file__)),
        "v17_source_sha256": trial.sha(Path(right.__file__)),
        "episode_runner_sha256": trial.sha(Path(trial.__file__)),
        "command_guard_sha256": trial.sha(Path(command.__file__)),
        "native_trace_sha256": trial.sha(kitchen.TRACE),
        "body_sha256": trial.sha(kitchen.BODY),
        "v17_report_sha256": trial.sha(right.OUT / "report.json"),
        "mujoco_version": mujoco.__version__,
        "results": results,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
