"""Illustrative odometry drift sensitivity of the mapped low-ray guard."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import run as base


OUT = base.ROOT / "artifacts/ape-native-route-guard/v6-odom-sensitivity"
DRIFT_X_M_AT_3S = .05
DRIFT_Y_M_AT_3S = .03
DRIFT_YAW_DEG_AT_3S = 3.0
DRIFT = False


def odom_pose(true_pose: np.ndarray, t: float) -> np.ndarray:
    if not DRIFT:
        return true_pose
    fraction = min(t/3.0, 1.0)
    pose = true_pose.copy()
    pose[0] += fraction*DRIFT_X_M_AT_3S
    pose[1] += fraction*DRIFT_Y_M_AT_3S
    yaw = base.replay.yaw_from_quat(true_pose[3:7]) + math.radians(
        fraction*DRIFT_YAW_DEG_AT_3S)
    pose[3:7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
    return pose


def read_with_drift(model, data, name, true_pose, case, t):
    packet = base.ORIGINAL_READ(model, data, name, true_pose, case, t)
    if name == "low":
        if base.GUARD is None:
            raise RuntimeError("Missing guard")
        decision = base.GUARD.update(packet, odom_pose(true_pose, t), t)
        packet["route_guard"] = decision
        packet["raw_min_forward_range_m"] = packet["min_forward_range_m"]
        packet["min_forward_range_m"] = base.STOP_SIGNAL_M if decision["risk"] else None
        packet["odom_drift_condition"] = "drift" if DRIFT else "ideal"
    return packet


def main() -> None:
    global DRIFT
    OUT.mkdir(parents=True, exist_ok=True)
    scene_path = OUT / "scene.xml"
    scene_path.write_text(base.replay.make_scene())
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    saved = np.load(base.replay.TRACE, allow_pickle=False)
    background = base.map_static_room(model, saved)
    background_path = OUT / "background_xy.npy"
    np.save(background_path, background)
    stop = json.loads(base.replay.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop["rows"] if r["case"] == "forward-stop-late"}
    base.replay.hazard_xy = base.close_hazard
    base.replay.read_rays = read_with_drift
    base.replay.overlay = base.overlay_guard
    base.replay.STOP_THRESHOLD_M = .75
    results = []
    for condition in ("ideal", "drift"):
        DRIFT = condition == "drift"
        base.replay.OUT = OUT / condition
        for world in range(4):
            for case in base.CASES:
                base.GUARD = base.FullScanGuard(background)
                result = base.replay.run_case(model, saved, world, case, "low15",
                                              delays[world], video=(world == 0))
                results.append({"condition": condition, **result})
                print(condition, world, case, result["stop_command_s"],
                      result["min_modeled_body_gap_m"], flush=True)
    report = {"schema": "robot-learning-lab.native-route-guard.odom-sensitivity.v1",
              "task": "T-9v3p3n",
              "purpose": "Matched ideal vs illustrative linear drift while actor registers low scan against the same prior room map",
              "drift_at_3s": {"x_m": DRIFT_X_M_AT_3S,
                                "y_m": DRIFT_Y_M_AT_3S,
                                "yaw_deg": DRIFT_YAW_DEG_AT_3S},
              "scene_sha256": base.replay.sha(scene_path),
              "background_map_sha256": base.replay.sha(background_path),
              "source_sha256": base.replay.sha(Path(__file__)),
              "guard_source_sha256": base.replay.sha(Path(base.__file__)),
              "native_trace_sha256": base.replay.sha(base.replay.TRACE),
              "body_sha256": base.replay.sha(base.replay.BODY),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    saved.close()


if __name__ == "__main__":
    main()
