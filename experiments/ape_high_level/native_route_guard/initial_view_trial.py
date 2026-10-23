"""Challenge the kitchen guard with a prior map from the start view only.

The room was seen once without the person before this matched episode. Unlike
V8, no future-path scans enter the actor map. This is still an ideal prior,
not autonomous online mapping or pose-estimation qualification.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import kitchen_native as kitchen
import kitchen_trial as trial
import run as guard


OUT = kitchen.OUT / "v9-initial-view-map"


def initial_map(model, saved) -> np.ndarray:
    model.body_pos[model.body("approaching_person").id, :2] = (10., -1.)
    model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
    data = mujoco.MjData(model)
    points = []
    for world in range(4):
        pose = kitchen.put_native_pose(model, data, saved, world, .02, 3.0)
        packet = guard.ORIGINAL_READ(model, data, "low", pose, "static_offpath", .02)
        points.extend(guard.low_ray_points(packet, pose))
    bins = {tuple(np.round(p/.02).astype(int)) for p in points}
    return np.asarray(sorted(bins), dtype=float).reshape(-1, 2)*.02


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    with np.load(kitchen.TRACE, allow_pickle=False) as saved:
        background = initial_map(model, saved)
        map_path = OUT / "background_xy.npy"
        np.save(map_path, background)
        stop = json.loads(guard.replay.STOP_REPORT.read_text())
        delays = {r["world"]: r["command_to_observed_stop_s"]
                  for r in stop["rows"] if r["case"] == "forward-stop-late"}
        trial.OUT = OUT
        results = []
        for world in range(4):
            for case in trial.CASES:
                for mode in trial.MODES:
                    video = ((world == 0 and (mode == "mapped_low_guard" or
                                             case == "slow_cross")) or
                             (world == 3 and case == "fast_clear"))
                    result = trial.run_episode(model, saved, background, world,
                                               case, mode, delays[world],
                                               video=video)
                    results.append(result)
                    print(world, case, mode, result["stop_command_s"],
                          result["min_modeled_body_person_gap_m"], flush=True)
    report = {"schema": "robot-learning-lab.ape-robocasa-initial-view-guard.v1",
              "task": "T-9v3p3n",
              "body_status": "historical_rejected_direct_frame",
              "method": "Same V8 articulated native trace, scene, guard and scripted cases; map includes only one hazard-free initial-pose low scan from each of four worlds. Actor gets ideal pose/rays; hazard-free map is provided before live person enters.",
              "room": {"scene_sha256": trial.sha(kitchen.OUT / "scene.xml")},
              "sensor": {"name": "low", "period_s": 1/15,
                         "static_map_points": len(background),
                         "prior_map_sha256": trial.sha(map_path),
                         "ideal_odom": True},
              "source_sha256": trial.sha(Path(__file__)),
              "episode_runner_sha256": trial.sha(Path(trial.__file__)),
              "scene_builder_sha256": trial.sha(Path(kitchen.__file__)),
              "guard_source_sha256": trial.sha(Path(guard.__file__)),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "stop_report_sha256": trial.sha(guard.replay.STOP_REPORT),
              "body_sha256": trial.sha(kitchen.BODY),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
