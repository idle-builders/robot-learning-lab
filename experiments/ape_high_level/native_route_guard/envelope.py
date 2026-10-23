"""Expose continuation and immediate-stop corridors alongside the guard action.

The 1.68 s / 0.11314 m envelope is the maximum observed native late-stop
window/path over four historical worlds. It bounds a modeled root path only;
the evaluator still scores all budget solids after the actual shifted tail.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import run as base


OUT = base.ROOT / "artifacts/ape-native-route-guard/v7-stop-envelope"


class EnvelopeGuard:
    def __init__(self, background: np.ndarray, stop_delay_s: float,
                 stop_path_m: float):
        self.tracker = base.FullScanGuard(background)
        self.stop_delay_s = stop_delay_s
        self.stop_path_m = stop_path_m
        self.previous_pose: tuple[float, np.ndarray] | None = None

    def update(self, packet: dict, pose: np.ndarray, t: float) -> dict:
        result = self.tracker.update(packet, pose, t)
        ego = np.asarray(pose[:2])
        yaw = base.replay.yaw_from_quat(pose[3:7])
        velocity = np.array([.35*math.cos(yaw), .35*math.sin(yaw)])
        if self.previous_pose:
            prev_t, prev_xy = self.previous_pose
            if t > prev_t:
                velocity = (ego-prev_xy)/(t-prev_t)
        self.previous_pose = (t, ego)
        for track in result["tracks"]:
            if track["velocity_xy_mps"] is None:
                track["continued_min_margin_m"] = None
                track["immediate_stop_min_margin_m"] = None
                track["stop_prediction_status"] = "unknown_velocity"
                continue
            point = np.asarray(track["point_xy_m"])
            hazard_v = np.asarray(track["velocity_xy_mps"])
            times = np.linspace(0, 2.5, 51)
            continued = point[None, :] + times[:, None]*hazard_v[None, :]
            continued -= ego[None, :] + times[:, None]*velocity[None, :]
            continued_margin = float(np.linalg.norm(continued, axis=1).min()-base.RISK_RADIUS_M)
            stop_times = np.linspace(0, self.stop_delay_s, 35)
            hazard_stop = point[None, :] + stop_times[:, None]*hazard_v[None, :]
            # The saved native path length can bend; subtract the whole
            # envelope around the current root rather than assuming straight
            # braking. This is conservative in XY, not full-body geometry.
            stop_margin = float(np.linalg.norm(hazard_stop-ego[None, :], axis=1).min()-
                                base.RISK_RADIUS_M-self.stop_path_m)
            track["continued_min_margin_m"] = round(continued_margin, 4)
            track["immediate_stop_min_margin_m"] = round(stop_margin, 4)
            track["stop_prediction_status"] = (
                "modeled_stop_cannot_clear" if stop_margin <= 0 else
                "modeled_stop_root_envelope_clear")
        result["stop_tail_envelope"] = {"max_observed_delay_s": self.stop_delay_s,
                                        "max_observed_path_m": self.stop_path_m,
                                        "body_proxy_radius_m": base.RISK_RADIUS_M}
        return result


GUARD: EnvelopeGuard | None = None


def read_envelope(model, data, name, pose, case, t):
    packet = base.ORIGINAL_READ(model, data, name, pose, case, t)
    if name == "low":
        if GUARD is None:
            raise RuntimeError("Missing guard")
        result = GUARD.update(packet, pose, t)
        packet["route_guard"] = result
        packet["raw_min_forward_range_m"] = packet["min_forward_range_m"]
        packet["min_forward_range_m"] = base.STOP_SIGNAL_M if result["risk"] else None
    return packet


def main() -> None:
    global GUARD
    OUT.mkdir(parents=True, exist_ok=True)
    scene_path = OUT / "scene.xml"
    scene_path.write_text(base.replay.make_scene())
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    saved = np.load(base.replay.TRACE, allow_pickle=False)
    background = base.map_static_room(model, saved)
    background_path = OUT / "background_xy.npy"
    np.save(background_path, background)
    stop = json.loads(base.replay.STOP_REPORT.read_text())
    rows = [r for r in stop["rows"] if r["case"] == "forward-stop-late"]
    delays = {r["world"]: r["command_to_observed_stop_s"] for r in rows}
    max_delay = max(delays.values())
    max_path = max(r["planar_path_to_observed_stop_m"] for r in rows)
    base.replay.hazard_xy = base.close_hazard
    base.replay.read_rays = read_envelope
    base.replay.overlay = base.overlay_guard
    base.replay.STOP_THRESHOLD_M = .75
    base.replay.OUT = OUT
    results = []
    for world in range(4):
        for case in base.CASES:
            GUARD = EnvelopeGuard(background, max_delay, max_path)
            result = base.replay.run_case(model, saved, world, case, "low15",
                                          delays[world], video=(world == 0))
            results.append(result)
            print(world, case, result["stop_command_s"],
                  result["min_modeled_body_gap_m"], flush=True)
    report = {"schema": "robot-learning-lab.native-route-guard.stop-envelope.v1",
              "task": "T-9v3p3n",
              "purpose": "Expose sampled continued-route and conservative immediate-stop root corridor next to each full-scan decision",
              "stop_envelope": {"max_delay_s": max_delay,
                                "max_path_m": max_path,
                                "body_proxy_radius_m": base.RISK_RADIUS_M},
              "scene_sha256": base.replay.sha(scene_path),
              "background_map_sha256": base.replay.sha(background_path),
              "source_sha256": base.replay.sha(Path(__file__)),
              "guard_source_sha256": base.replay.sha(Path(base.__file__)),
              "stop_report_sha256": base.replay.sha(base.replay.STOP_REPORT),
              "native_trace_sha256": base.replay.sha(base.replay.TRACE),
              "body_sha256": base.replay.sha(base.replay.BODY),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    saved.close()


if __name__ == "__main__":
    main()
