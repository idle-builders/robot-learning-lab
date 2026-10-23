"""Use the commanded forward route in the kitchen near-body forecast.

This V10 diagnostic keeps V9's four-point initial-view map and person cases.
The only changed decision input is planned +X velocity (the saved actor's
0.35 m/s command), replacing short-window ego odometry velocity in closest
approach. It is a nominal route proxy, not a swept body clearance guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import initial_view_trial as initial
import kitchen_native as kitchen
import kitchen_trial as trial
import run as guard


OUT = kitchen.OUT / "v10-command-route"
COMMAND_VELOCITY_XY_MPS = np.array([.35, 0.])
OriginalGuard = guard.FullScanGuard


class CommandRouteGuard:
    def __init__(self, background: np.ndarray):
        self.inner = OriginalGuard(background)
        self.background = self.inner.background

    def update(self, packet: dict, pose: np.ndarray, t: float) -> dict:
        decision = self.inner.update(packet, pose, t)
        risk = False
        for track in decision["tracks"]:
            track["odometry_forecast_risk"] = track["risk"]
            track["odometry_forecast_closest_approach_m"] = track["closest_approach_m"]
            if track["velocity_xy_mps"] is not None:
                point = np.asarray(track["point_xy_m"])
                velocity = np.asarray(track["velocity_xy_mps"])
                relative = point - pose[:2]
                relative_velocity = velocity - COMMAND_VELOCITY_XY_MPS
                vv = float(relative_velocity @ relative_velocity)
                cpa_s = (float(np.clip(-(relative @ relative_velocity)/vv,
                                       0, guard.HORIZON_S)) if vv > 1e-8 else 0.)
                cpa_m = float(np.linalg.norm(relative + cpa_s*relative_velocity))
                track["closest_approach_s"] = round(cpa_s, 4)
                track["closest_approach_m"] = round(cpa_m, 4)
                track["risk"] = bool(relative[0] > -.10 and cpa_m < guard.RISK_RADIUS_M)
            risk |= track["risk"]
        decision["odometry_forecast_risk"] = decision["risk"]
        decision["risk"] = bool(risk)
        decision["planned_velocity_xy_mps"] = COMMAND_VELOCITY_XY_MPS.tolist()
        return decision


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
    guard.FullScanGuard = CommandRouteGuard
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            results = []
            for world in range(4):
                for case in trial.CASES:
                    result = trial.run_episode(model, saved, background, world,
                                               case, "mapped_low_guard",
                                               delays[world],
                                               video=(world == 0 or
                                                      world == 3 and case == "fast_clear"))
                    results.append(result)
                    print(world, case, result["stop_command_s"],
                          result["min_modeled_body_person_gap_m"], flush=True)
    finally:
        guard.FullScanGuard = OriginalGuard
    report = {"schema": "robot-learning-lab.ape-robocasa-command-route-guard.v1",
              "task": "T-9v3p3n",
              "body_status": "historical_rejected_direct_frame",
              "method": "V9 initial-view map, native root/joint response and person crossings unchanged. Mature-track closest approach now uses the requested +0.35 m/s X route rather than noisy instantaneous root velocity. Immature-track unknown-region rule unchanged.",
              "command_velocity_xy_mps": COMMAND_VELOCITY_XY_MPS.tolist(),
              "room": {"scene_sha256": trial.sha(kitchen.OUT / "scene.xml")},
              "sensor": {"name": "low", "period_s": 1/15,
                         "static_map_points": len(background),
                         "prior_map_sha256": trial.sha(map_path),
                         "ideal_odom": True},
              "source_sha256": trial.sha(Path(__file__)),
              "episode_runner_sha256": trial.sha(Path(trial.__file__)),
              "original_guard_sha256": trial.sha(Path(guard.__file__)),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "stop_report_sha256": trial.sha(guard.replay.STOP_REPORT),
              "body_sha256": trial.sha(kitchen.BODY),
              "v9_control_report_sha256": trial.sha(initial.OUT / "report.json"),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
