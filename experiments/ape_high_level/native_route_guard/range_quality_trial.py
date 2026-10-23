"""V16: seeded range-quality stress on V15's independent left-flank fan.

The noise/dropout levels are illustrative perturbations, not a measured device
model. Healthy no-echo and unavailable rays remain separate packet states.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import changed_speed_trial as holdout
import command_route_trial as command
import dense_low_trial as dense
import flank_sensor_trial as flank
import kitchen_native as kitchen
import kitchen_trial as trial
import run as guard


OUT = kitchen.OUT / "v16-range-quality"
MODES = {
    "mild": {"range_sigma_m": .01, "ray_unavailable_probability": .02,
             "flank_blackout_s": None},
    "strong": {"range_sigma_m": .03, "ray_unavailable_probability": .10,
               "flank_blackout_s": None},
    "flank_blackout": {"range_sigma_m": 0., "ray_unavailable_probability": 0.,
                       "flank_blackout_s": [.10, .90]},
}
SEEDS = (13, 29)
ORIGINAL_READ = guard.ORIGINAL_READ
ORIGINAL_POINTS = guard.low_ray_points
ORIGINAL_HAZARD = trial.hazard_xy
RNG = np.random.default_rng(0)
MODE = "mild"


def perturbed_read(model, data, name, pose, case, t) -> dict:
    packet = flank.fused_read(model, data, name, pose, case, t)
    config = MODES[MODE]
    values = []
    health = []
    for distance, source in zip(packet["ranges_m"], packet["ray_sources"]):
        unavailable = bool(RNG.random() < config["ray_unavailable_probability"])
        if (source == "left_flank_low" and config["flank_blackout_s"] is not None
                and config["flank_blackout_s"][0] <= t < config["flank_blackout_s"][1]):
            unavailable = True
        health.append(not unavailable)
        if unavailable or distance is None:
            values.append(None)
        else:
            reading = float(distance) + float(RNG.normal(0., config["range_sigma_m"]))
            values.append(round(float(np.clip(reading, 0., dense.MAX_M)), 5))
    packet["ranges_m"] = values
    packet["ray_health"] = health
    packet["primary_ray_health"] = [okay and source == "front_low"
                                    for okay, source in zip(health, packet["ray_sources"])]
    packet["flank_ray_health"] = [okay and source == "left_flank_low"
                                  for okay, source in zip(health, packet["ray_sources"])]
    packet["valid_returns"] = sum(value is not None for value in values)
    forward = [value for value, bearing in zip(values, dense.YAW_DEG)
               if value is not None and abs(bearing) <= 25.]
    packet["min_forward_range_m"] = min(forward) if forward else None
    packet["quality_condition"] = MODE
    packet["unavailable_rays"] = len(health)-sum(health)
    return packet


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    background = np.load(flank.OUT / "background_xy.npy")
    map_path = OUT / "background_xy.npy"
    np.save(map_path, background)
    stop = json.loads(guard.replay.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop["rows"] if r["case"] == "forward-stop-late"}
    trial.hazard_xy = holdout.hazard_xy
    guard.ORIGINAL_READ = perturbed_read
    guard.low_ray_points = flank.fused_points
    guard.FullScanGuard = command.CommandRouteGuard
    global MODE, RNG
    results = []
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            for mode_index, mode in enumerate(MODES):
                MODE = mode
                for seed in SEEDS:
                    trial.OUT = OUT / mode / f"seed-{seed}"
                    for world in range(4):
                        for case_index, case in enumerate(holdout.CASES):
                            packet_seed = (seed*10000 + mode_index*1000 +
                                           world*100 + case_index)
                            RNG = np.random.default_rng(packet_seed)
                            video = seed == SEEDS[0] and world == 0 and case in (
                                "fast_clear_held", "slow_cross_held")
                            result = trial.run_episode(model, saved, background,
                                                       world, case, "mapped_low_guard",
                                                       delays[world], video=video)
                            result.update(condition=mode, seed=seed,
                                          packet_seed=packet_seed)
                            results.append(result)
                            print(mode, seed, world, case,
                                  result["stop_command_s"],
                                  result["min_modeled_body_person_gap_m"], flush=True)
    finally:
        trial.hazard_xy = ORIGINAL_HAZARD
        guard.ORIGINAL_READ = ORIGINAL_READ
        guard.low_ray_points = ORIGINAL_POINTS
        guard.FullScanGuard = command.OriginalGuard
    report = {
        "schema": "robot-learning-lab.ape-robocasa-low-fan-range-quality.v1",
        "task": "T-9v3p3n",
        "body_status": "historical_rejected_direct_frame",
        "method": "Frozen V15 actor, fused initial-view map, V11 changed-speed cases and historical native stop tails. Seeded range noise, per-ray unavailability and timed complete flank loss are illustrative, not sensor specifications. Guard receives packet health but does not yet use it.",
        "conditions": MODES,
        "seeds": list(SEEDS),
        "sensor": {"period_s": 1/15, "rays": 73, "max_range_m": dense.MAX_M,
                   "front_origin_local_m": flank.FRONT_ORIGIN_LOCAL_M.tolist(),
                   "flank_origin_local_m": flank.FLANK_ORIGIN_LOCAL_M.tolist(),
                   "initial_map_sha256": trial.sha(map_path),
                   "ideal_odom": True},
        "source_sha256": trial.sha(Path(__file__)),
        "v15_source_sha256": trial.sha(Path(flank.__file__)),
        "episode_runner_sha256": trial.sha(Path(trial.__file__)),
        "command_guard_sha256": trial.sha(Path(command.__file__)),
        "scene_sha256": trial.sha(kitchen.OUT / "scene.xml"),
        "native_trace_sha256": trial.sha(kitchen.TRACE),
        "body_sha256": trial.sha(kitchen.BODY),
        "v15_report_sha256": trial.sha(flank.OUT / "report.json"),
        "mount_screen_sha256": trial.sha(flank.OUT / "mount-visibility.json"),
        "mujoco_version": mujoco.__version__,
        "results": results,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
