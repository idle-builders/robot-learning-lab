"""V15: independent low left-flank fan fills V14's blocked front sector.

Both fans are ideal, 15 Hz, and 1.2 m. The front sensor's +20..+70 degree
sector remains unavailable. A second declared mount provides only that sector.
No scene labels, scripted hazard pose or evaluator gap reach the actor guard.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import blocked_sector_trial as blocked
import changed_speed_trial as holdout
import command_route_trial as command
import dense_low_trial as dense
import kitchen_native as kitchen
import kitchen_trial as trial
import run as guard


OUT = kitchen.OUT / "v15-left-flank-fan"
FRONT_ORIGIN_LOCAL_M = np.array([.08, 0., -.15])
FLANK_ORIGIN_LOCAL_M = np.array([0., .10, -.15])
ORIGINAL_READ = guard.ORIGINAL_READ
ORIGINAL_POINTS = guard.low_ray_points
ORIGINAL_HAZARD = trial.hazard_xy


def world_origin(pose: np.ndarray, local: np.ndarray) -> np.ndarray:
    yaw = guard.replay.yaw_from_quat(pose[3:7])
    ch, sh = math.cos(yaw), math.sin(yaw)
    return pose[:3] + np.array([ch*local[0]-sh*local[1],
                                 sh*local[0]+ch*local[1], local[2]])


def fused_read(model, data, name, pose, case, t) -> dict:
    if name != "low":
        raise ValueError(name)
    primary = dense.dense_read(model, data, name, pose, case, t)
    yaw = guard.replay.yaw_from_quat(pose[3:7])
    flank_origin = world_origin(pose, FLANK_ORIGIN_LOCAL_M)
    ranges = []
    sources = []
    for bearing_deg, primary_value in zip(dense.YAW_DEG, primary["ranges_m"]):
        if bearing_deg <= 20.:
            ranges.append(primary_value)
            sources.append("front_low")
            continue
        angle = yaw+math.radians(float(bearing_deg))
        direction = np.array([math.cos(angle), math.sin(angle), 0.])
        geomid = np.array([-1], dtype=np.int32)
        hit = mujoco.mj_ray(model, data, flank_origin, direction,
                            np.array([1, 0, 0, 0, 0, 0], np.uint8),
                            1, -1, geomid)
        ranges.append(None if hit < 0 or hit > dense.MAX_M else round(float(hit), 5))
        sources.append("left_flank_low")
    forward = [value for bearing, value in zip(dense.YAW_DEG, ranges)
               if value is not None and abs(bearing) <= 25.]
    primary_health = [bool(bearing <= 20.) for bearing in dense.YAW_DEG]
    flank_health = [not okay for okay in primary_health]
    return {"sensor": "front_low_plus_left_flank", "captured_at_s": round(float(t), 4),
            "min_forward_range_m": min(forward) if forward else None,
            "valid_returns": sum(value is not None for value in ranges),
            "ranges_m": ranges, "ray_sources": sources,
            "ray_health": [True]*len(ranges),
            "primary_ray_health": primary_health,
            "flank_ray_health": flank_health,
            "primary_blocked_sector_deg": [20., 70.]}


def fused_points(packet: dict, pose: np.ndarray) -> list[np.ndarray]:
    yaw = guard.replay.yaw_from_quat(pose[3:7])
    front_origin = world_origin(pose, FRONT_ORIGIN_LOCAL_M)[:2]
    flank_origin = world_origin(pose, FLANK_ORIGIN_LOCAL_M)[:2]
    points = []
    for bearing_deg, distance, source in zip(dense.YAW_DEG,
                                              packet["ranges_m"],
                                              packet["ray_sources"]):
        if distance is None:
            continue
        angle = yaw+math.radians(float(bearing_deg))
        origin = front_origin if source == "front_low" else flank_origin
        point = origin+distance*np.array([math.cos(angle), math.sin(angle)])
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
        points.extend(fused_points(fused_read(model, data, "low", pose,
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
    guard.ORIGINAL_READ = fused_read
    guard.low_ray_points = fused_points
    guard.FullScanGuard = command.CommandRouteGuard
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            background = initial_map(model, saved)
            map_path = OUT / "background_xy.npy"
            np.save(map_path, background)
            results = []
            for world in range(4):
                for case in holdout.CASES:
                    video = (case == "fast_clear_held" or world == 0)
                    result = trial.run_episode(model, saved, background,
                                               world, case, "mapped_low_guard",
                                               delays[world], video=video)
                    results.append(result)
                    print(world, case, result["stop_command_s"],
                          result["min_modeled_body_person_gap_m"], flush=True)
    finally:
        trial.hazard_xy = ORIGINAL_HAZARD
        guard.ORIGINAL_READ = ORIGINAL_READ
        guard.low_ray_points = ORIGINAL_POINTS
        guard.FullScanGuard = command.OriginalGuard
    report = {"schema": "robot-learning-lab.ape-robocasa-low-flank-fusion.v1",
              "task": "T-9v3p3n",
              "body_status": "historical_rejected_direct_frame",
              "method": "V14 blocked front-left sector plus ideal independent low side fan at (0,+0.10,-0.15)m. Same V11 crossing scripts, 15Hz, 1.2m, 73 total bearings, command-route guard, historical native stop-tail. Initial static map built only from the fused starting view. No physical mount clearance or sensor noise qualification.",
              "sensor": {"period_s": 1/15, "rays": 73, "max_range_m": 1.2,
                         "front_origin_local_m": FRONT_ORIGIN_LOCAL_M.tolist(),
                         "flank_origin_local_m": FLANK_ORIGIN_LOCAL_M.tolist(),
                         "primary_unavailable_sector_deg": [20., 70.],
                         "flank_sector_deg": [20., 70.],
                         "static_map_points": len(background),
                         "prior_map_sha256": trial.sha(map_path),
                         "ideal_odom": True},
              "scene_sha256": trial.sha(kitchen.OUT / "scene.xml"),
              "source_sha256": trial.sha(Path(__file__)),
              "episode_runner_sha256": trial.sha(Path(trial.__file__)),
              "dense_sensor_sha256": trial.sha(Path(dense.__file__)),
              "command_guard_sha256": trial.sha(Path(command.__file__)),
              "original_guard_sha256": trial.sha(Path(guard.__file__)),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "body_sha256": trial.sha(kitchen.BODY),
              "v14_report_sha256": trial.sha(blocked.OUT / "report.json"),
              "v13_report_sha256": trial.sha(dense.OUT / "report.json"),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
