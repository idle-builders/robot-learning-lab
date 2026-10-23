"""Matched 17-versus-73-ray low-scan challenge on V11 kitchen crossings."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import changed_speed_trial as holdout
import command_route_trial as command
import kitchen_native as kitchen
import kitchen_trial as trial
import run as guard


OUT = kitchen.OUT / "v13-dense-low"
YAW_DEG = np.linspace(-70., 70., 73)
MAX_M = 1.2
ORIGINAL_READ = guard.ORIGINAL_READ
ORIGINAL_POINTS = guard.low_ray_points
ORIGINAL_HAZARD = trial.hazard_xy


def dense_read(model, data, name, pose, case, t) -> dict:
    if name != "low":
        raise ValueError(name)
    yaw = guard.replay.yaw_from_quat(pose[3:7])
    origin = pose[:3] + np.array([.08*math.cos(yaw), .08*math.sin(yaw), -.15])
    ranges = []
    for bearing_deg in YAW_DEG:
        angle = yaw + math.radians(float(bearing_deg))
        direction = np.array([math.cos(angle), math.sin(angle), 0.])
        geomid = np.array([-1], dtype=np.int32)
        hit = mujoco.mj_ray(model, data, origin, direction,
                            np.array([1, 0, 0, 0, 0, 0], np.uint8),
                            1, -1, geomid)
        ranges.append(None if hit < 0 or hit > MAX_M else round(float(hit), 5))
    forward = [distance for distance, bearing in zip(ranges, YAW_DEG)
               if distance is not None and abs(bearing) <= 25]
    return {"sensor": "low_dense", "captured_at_s": round(float(t), 4),
            "min_forward_range_m": min(forward) if forward else None,
            "valid_returns": sum(value is not None for value in ranges),
            "ranges_m": ranges}


def dense_points(packet: dict, pose: np.ndarray) -> list[np.ndarray]:
    yaw = guard.replay.yaw_from_quat(pose[3:7])
    origin = pose[:2] + .08*np.array([math.cos(yaw), math.sin(yaw)])
    points = []
    for bearing_deg, distance in zip(YAW_DEG, packet["ranges_m"]):
        if distance is None:
            continue
        angle = yaw + math.radians(float(bearing_deg))
        point = origin + distance*np.array([math.cos(angle), math.sin(angle)])
        if point[0] > pose[0]-.05 and abs(point[1]-pose[1]) <= .85:
            points.append(point)
    return points


def initial_map(model, saved) -> np.ndarray:
    model.body_pos[model.body("approaching_person").id, :2] = (10., -1.)
    model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
    data = mujoco.MjData(model)
    points = []
    for world in range(4):
        pose = kitchen.put_native_pose(model, data, saved, world, .02, 3.)
        points.extend(dense_points(dense_read(model, data, "low", pose,
                                              "static_offpath", .02), pose))
    bins = {tuple(np.round(p/.02).astype(int)) for p in points}
    return np.asarray(sorted(bins), dtype=float).reshape(-1, 2)*.02


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    stop = json.loads(guard.replay.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop["rows"] if r["case"] == "forward-stop-late"}
    trial.OUT = OUT
    trial.hazard_xy = holdout.hazard_xy
    guard.ORIGINAL_READ = dense_read
    guard.low_ray_points = dense_points
    guard.FullScanGuard = command.CommandRouteGuard
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            background = initial_map(model, saved)
            map_path = OUT / "background_xy.npy"
            np.save(map_path, background)
            results = []
            for world in range(4):
                for case in holdout.CASES:
                    result = trial.run_episode(model, saved, background,
                                               world, case, "mapped_low_guard",
                                               delays[world],
                                               video=(case == "fast_clear_held" or world == 0))
                    results.append(result)
                    print(world, case, result["stop_command_s"],
                          result["min_modeled_body_person_gap_m"], flush=True)
    finally:
        trial.hazard_xy = ORIGINAL_HAZARD
        guard.ORIGINAL_READ = ORIGINAL_READ
        guard.low_ray_points = ORIGINAL_POINTS
        guard.FullScanGuard = command.OriginalGuard
    report = {"schema": "robot-learning-lab.ape-robocasa-dense-low-scan.v1",
              "task": "T-9v3p3n",
              "body_status": "historical_rejected_direct_frame",
              "method": "V11 changed-speed cases and V10 command-route guard, same scene, low mount, 15Hz and 1.2m max; replace 17 ideal rays over 140 degrees with 73 ideal rays and rebuild an initial-pose map from that sensor. No new controller rollout.",
              "sensor": {"name": "low_dense", "period_s": 1/15,
                         "origin_relative_m": [.08, 0, -.15],
                         "yaw_range_deg": [-70, 70], "rays": 73,
                         "max_range_m": MAX_M,
                         "static_map_points": len(background),
                         "prior_map_sha256": trial.sha(map_path),
                         "ideal_odom": True},
              "scene_sha256": trial.sha(kitchen.OUT / "scene.xml"),
              "source_sha256": trial.sha(Path(__file__)),
              "episode_runner_sha256": trial.sha(Path(trial.__file__)),
              "command_guard_sha256": trial.sha(Path(command.__file__)),
              "original_guard_sha256": trial.sha(Path(guard.__file__)),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "body_sha256": trial.sha(kitchen.BODY),
              "v11_report_sha256": trial.sha(holdout.OUT / "report.json"),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
