"""V27: person enters the held native-body corridor after skill deadline.

This exposes the missing moving/waiting/stopped monitor in the V24 runner.
The scripted person is withheld from the route until after the 3 s command.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import dense_low_trial as dense
import heldout_plan_scene as previous
import kitchen_native as kitchen
import kitchen_trial as trial
import online_cluster_guard as online
import plan_bounded_trial as plan
import run as guard


OUT = kitchen.OUT / "v27-held-body-intrusion"
CASES = ("late_left_pause", "late_right_pause")
ENTRY_START_S = 3.10
ENTRY_SPEED_MPS = .62
ENTRY_CENTER_Y_M = -1.80
ORIGINAL_READ = guard.ORIGINAL_READ
ORIGINAL_POINTS = guard.low_ray_points
ORIGINAL_HAZARD = trial.hazard_xy
ORIGINAL_CLASS = guard.FullScanGuard
ORIGINAL_UNKNOWN = guard.UNKNOWN_LATERAL_M


def hazard_xy(case: str, t: float) -> tuple[float, float]:
    dt = max(0., t-ENTRY_START_S)
    if case == "late_left_pause":
        y = max(ENTRY_CENTER_Y_M, -1.05-ENTRY_SPEED_MPS*dt)
    elif case == "late_right_pause":
        y = min(ENTRY_CENTER_Y_M, -2.55+ENTRY_SPEED_MPS*dt)
    else:
        raise ValueError(case)
    return 3.55, y


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = previous.OUT / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    stop = json.loads(guard.replay.STOP_REPORT.read_text())
    delays = {row["world"]: row["command_to_observed_stop_s"]
              for row in stop["rows"] if row["case"] == "forward-stop-late"}
    trial.hazard_xy = hazard_xy
    guard.ORIGINAL_READ = dense.dense_read
    guard.low_ray_points = dense.dense_points
    guard.UNKNOWN_LATERAL_M = guard.RISK_RADIUS_M
    results = []
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            for mode in ("none", "mapped_low_guard"):
                trial.OUT = OUT / ("no_sensor" if mode == "none" else
                                   "bounded_guard")
                guard.FullScanGuard = (ORIGINAL_CLASS if mode == "none" else
                                      plan.PlanBoundedGuard)
                for world in range(4):
                    for case in CASES:
                        row = trial.run_episode(model, saved,
                                                online.NO_MAP_SENTINEL_XY_M,
                                                world, case, mode,
                                                delays[world], video=(world == 0))
                        row["configuration"] = ("no_sensor" if mode == "none" else
                                                 "bounded_guard")
                        results.append(row)
                        print(row["configuration"], world, case,
                              row["stop_command_s"],
                              row["min_modeled_body_person_gap_m"], flush=True)
    finally:
        trial.hazard_xy = ORIGINAL_HAZARD
        guard.ORIGINAL_READ = ORIGINAL_READ
        guard.low_ray_points = ORIGINAL_POINTS
        guard.FullScanGuard = ORIGINAL_CLASS
        guard.UNKNOWN_LATERAL_M = ORIGINAL_UNKNOWN
    report = {
        "schema": "robot-learning-lab.ape-robocasa-held-body-intrusion.v1",
        "task": "T-9v3p3n",
        "body_status": "historical_rejected_direct_frame",
        "method": "V24 bounded actor on the V26 box-stool scene with person stationary at either side until3.10s, after the 3.00s forward skill deadline. The person then moves .62m/s toward y=-1.80m and pauses. Identical no-sensor controls; ideal low rays and pose, but the shared episode runner does not take new guard packets after its stop command. Sampled full-body-person gap includes the native stop tail and held interval. This is an architectural failure challenge, not a policy or physical collision rollout.",
        "scene_sha256": trial.sha(scene),
        "source_sha256": trial.sha(Path(__file__)),
        "actor_source_sha256": trial.sha(Path(plan.__file__)),
        "runner_source_sha256": trial.sha(Path(trial.__file__)),
        "native_trace_sha256": trial.sha(kitchen.TRACE),
        "body_sha256": trial.sha(kitchen.BODY),
        "v26_report_sha256": trial.sha(previous.OUT / "report.json"),
        "entry_start_s": ENTRY_START_S,
        "entry_speed_mps": ENTRY_SPEED_MPS,
        "entry_pause_y_m": ENTRY_CENTER_Y_M,
        "results": results,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
