"""V17 right-entry development test for a left-only low-fan suite.

This deliberately changes the side of entry while retaining the historical
native trajectories, furnished room, planned-route guard and stop-tail law.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import command_route_trial as command
import dense_low_trial as dense
import flank_sensor_trial as left
import kitchen_native as kitchen
import kitchen_trial as trial
import run as guard


OUT = kitchen.OUT / "v17-right-entry"
CASES = ("fast_right_held", "slow_right_held", "pause_right_held")
MODES = ("none", "left_only_right_missing", "left_plus_right")
RIGHT_ORIGIN_LOCAL_M = np.array([0., -.10, -.15])
ORIGINAL_READ = guard.ORIGINAL_READ
ORIGINAL_POINTS = guard.low_ray_points
ORIGINAL_HAZARD = trial.hazard_xy
MODE = "left_plus_right"


def hazard_xy(case: str, t: float) -> tuple[float, float]:
    y = {
        "fast_right_held": min(-1.00, -2.55 + .72*t),
        "slow_right_held": -2.55 + .22*min(t, 4.),
        "pause_right_held": -2.55 + .50*min(t, 1.5),
    }[case]
    return 3.55, y


def three_source_read(model, data, name, pose, case, t) -> dict:
    packet = left.fused_read(model, data, name, pose, case, t)
    yaw = guard.replay.yaw_from_quat(pose[3:7])
    right_origin = left.world_origin(pose, RIGHT_ORIGIN_LOCAL_M)
    ranges = []
    sources = []
    for bearing, original, source in zip(dense.YAW_DEG, packet["ranges_m"],
                                         packet["ray_sources"]):
        if bearing >= -20.:
            ranges.append(original)
            sources.append(source)
            continue
        if MODE == "left_only_right_missing":
            ranges.append(None)
            sources.append("front_right_unavailable")
            continue
        angle = yaw + math.radians(float(bearing))
        direction = np.array([math.cos(angle), math.sin(angle), 0.])
        geomid = np.array([-1], dtype=np.int32)
        hit = mujoco.mj_ray(model, data, right_origin, direction,
                            np.array([1, 0, 0, 0, 0, 0], np.uint8),
                            1, -1, geomid)
        ranges.append(None if hit < 0 or hit > dense.MAX_M else round(float(hit), 5))
        sources.append("right_flank_low")
    healthy = [source != "front_right_unavailable" for source in sources]
    forward = [value for value, bearing in zip(ranges, dense.YAW_DEG)
               if value is not None and abs(bearing) <= 25.]
    packet.update(sensor="front_low_plus_left_flank" +
                  ("_plus_right_flank" if MODE == "left_plus_right" else "_right_unavailable"),
                  ranges_m=ranges, ray_sources=sources, ray_health=healthy,
                  primary_ray_health=[source == "front_low" for source in sources],
                  flank_ray_health=[source == "left_flank_low" for source in sources],
                  right_flank_ray_health=[source == "right_flank_low" for source in sources],
                  right_unavailable_sector_deg=[-70., -20.] if MODE ==
                  "left_only_right_missing" else None,
                  valid_returns=sum(value is not None for value in ranges),
                  min_forward_range_m=min(forward) if forward else None)
    return packet


def three_source_points(packet: dict, pose: np.ndarray) -> list[np.ndarray]:
    yaw = guard.replay.yaw_from_quat(pose[3:7])
    origins = {
        "front_low": left.world_origin(pose, left.FRONT_ORIGIN_LOCAL_M)[:2],
        "left_flank_low": left.world_origin(pose, left.FLANK_ORIGIN_LOCAL_M)[:2],
        "right_flank_low": left.world_origin(pose, RIGHT_ORIGIN_LOCAL_M)[:2],
    }
    points = []
    for bearing, distance, source in zip(dense.YAW_DEG, packet["ranges_m"],
                                         packet["ray_sources"]):
        if distance is None:
            continue
        angle = yaw + math.radians(float(bearing))
        point = origins[source] + distance*np.array([math.cos(angle), math.sin(angle)])
        if point[0] > pose[0]-.05 and abs(point[1]-pose[1]) <= .85:
            points.append(point)
    return points


def initial_map(model, saved) -> np.ndarray:
    global MODE
    old_mode = MODE
    MODE = "left_plus_right"
    model.body_pos[model.body("approaching_person").id, :2] = (10., -1.)
    model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
    data = mujoco.MjData(model)
    points = []
    try:
        for world in range(4):
            pose = kitchen.put_native_pose(model, data, saved, world, .02, 3.)
            points.extend(three_source_points(three_source_read(
                model, data, "low", pose, "static_offpath", .02), pose))
    finally:
        MODE = old_mode
    bins = {tuple(np.round(point/.02).astype(int)) for point in points}
    return np.asarray(sorted(bins), dtype=float).reshape(-1, 2)*.02


def main() -> None:
    global MODE
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    stop = json.loads(guard.replay.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop["rows"] if r["case"] == "forward-stop-late"}
    trial.hazard_xy = hazard_xy
    guard.ORIGINAL_READ = three_source_read
    guard.low_ray_points = three_source_points
    guard.FullScanGuard = command.CommandRouteGuard
    results = []
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            background = initial_map(model, saved)
            map_path = OUT / "background_xy.npy"
            np.save(map_path, background)
            for mode in MODES:
                MODE = mode
                trial.OUT = OUT / mode
                for world in range(4):
                    for case in CASES:
                        video = world == 0 and case in ("fast_right_held",
                                                      "slow_right_held")
                        result = trial.run_episode(model, saved, background,
                                                   world, case,
                                                   "none" if mode == "none"
                                                   else "mapped_low_guard",
                                                   delays[world], video=video)
                        result["configuration"] = mode
                        results.append(result)
                        print(mode, world, case, result["stop_command_s"],
                              result["min_modeled_body_person_gap_m"], flush=True)
    finally:
        trial.hazard_xy = ORIGINAL_HAZARD
        guard.ORIGINAL_READ = ORIGINAL_READ
        guard.low_ray_points = ORIGINAL_POINTS
        guard.FullScanGuard = command.OriginalGuard
    report = {
        "schema": "robot-learning-lab.ape-robocasa-right-entry-fan-development.v1",
        "task": "T-9v3p3n",
        "body_status": "historical_rejected_direct_frame",
        "method": "Right-to-center crossing in same RoboCasa scene and native traces. Fast script is capped at y=-1.00m after scene-fixture screen found an unbounded path intersected the counter. Shared ideal initial-view static map made from all three low fans, including right fan even for the no-right configuration; that gives the no-right control a map advantage. Front -70..-20deg sector is unavailable in both sensor modes. No new learned control or physical stop.",
        "cases": {"start_xy_m": [3.55, -2.55],
                  "fast_vy_mps": .72, "fast_cap_y_m": -1.00,
                  "slow_vy_mps": .22,
                  "pause_vy_mps": .50, "pause_at_s": 1.5},
        "sensor": {"period_s": 1/15, "rays": 73,
                   "right_sector_rays": int(sum(b < -20. for b in dense.YAW_DEG)),
                   "max_range_m": dense.MAX_M,
                   "front_origin_local_m": left.FRONT_ORIGIN_LOCAL_M.tolist(),
                   "left_origin_local_m": left.FLANK_ORIGIN_LOCAL_M.tolist(),
                   "right_origin_local_m": RIGHT_ORIGIN_LOCAL_M.tolist(),
                   "initial_map_points": len(background),
                   "initial_map_sha256": trial.sha(map_path),
                   "ideal_odom": True},
        "scene_sha256": trial.sha(kitchen.OUT / "scene.xml"),
        "source_sha256": trial.sha(Path(__file__)),
        "v15_source_sha256": trial.sha(Path(left.__file__)),
        "episode_runner_sha256": trial.sha(Path(trial.__file__)),
        "command_guard_sha256": trial.sha(Path(command.__file__)),
        "native_trace_sha256": trial.sha(kitchen.TRACE),
        "body_sha256": trial.sha(kitchen.BODY),
        "v15_report_sha256": trial.sha(left.OUT / "report.json"),
        "mujoco_version": mujoco.__version__,
        "results": results,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
