"""Development test of a scan-tracked near-body collision forecast.

The gate receives only ray returns plus an explicitly idealized root odometry
estimate. It does not receive the hazard script, type, or evaluator gap.
Point tracks are nearest-neighbor scan associations; the short-horizon
constant-velocity forecast is intentionally small and inspectable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import run
from sweep_threshold import ORIGINAL_HAZARD, crossing_hazard


OUT = run.ROOT / "artifacts/ape-native-stop-sensing/v6-motion-gate"
THRESHOLD_M = .75
TRACK_ASSOCIATION_M = .16
COLLISION_RADIUS_M = .22
CLEARANCE_WAIVER_M = .27
LATERAL_WAIVER_MPS = .25
HORIZON_S = 2.5
CASES = ("approaching_person", "low_pet", "static_offpath",
         "false_return", "crossing_clear")


class ScanTracker:
    def __init__(self) -> None:
        self.track: list[tuple[float, np.ndarray]] = []
        self.previous_pose: tuple[float, np.ndarray] | None = None

    def risk(self, packet: dict, pose: np.ndarray, t: float) -> tuple[bool, dict]:
        heading = run.yaw_from_quat(pose[3:7])
        ch, sh = math.cos(heading), math.sin(heading)
        origin = pose[:2] + np.array([.08*ch, .08*sh])
        points = []
        for i, distance in enumerate(packet["ranges_m"]):
            if distance is None or not 5 <= i <= 12:
                continue
            angle = heading + math.radians(-70 + 8.75*i)
            point = origin + distance*np.array([math.cos(angle), math.sin(angle)])
            if point[0] > pose[0] and abs(point[1]-pose[1]) <= .36:
                points.append(point)
        # Track the leading visible surface in the planned travel corridor.
        # This avoids confusing the crossing person with furniture at y=-.54.
        observation = min(points, key=lambda p: p[0]) if points else None
        if observation is None:
            self.track = []
        elif self.track and np.linalg.norm(observation-self.track[-1][1]) <= TRACK_ASSOCIATION_M:
            self.track = (self.track + [(t, observation)])[-7:]
        else:
            self.track = [(t, observation)]

        ego_xy = np.asarray(pose[:2])
        ego_v = np.array([.35*ch, .35*sh])
        if self.previous_pose is not None:
            prev_t, prev_xy = self.previous_pose
            if t > prev_t:
                ego_v = (ego_xy-prev_xy)/(t-prev_t)
        self.previous_pose = (t, ego_xy)

        predictions = []
        if len(self.track) >= 3 and self.track[-1][0]-self.track[0][0] >= .12:
            times = np.array([r[0] for r in self.track])
            positions = np.array([r[1] for r in self.track])
            dt = times-times.mean()
            velocity = (dt[:, None]*(positions-positions.mean(axis=0))).sum(axis=0)/(dt*dt).sum()
            point = positions[-1]
            relative = point-ego_xy
            relative_velocity = velocity-ego_v
            vv = float(relative_velocity@relative_velocity)
            closest_t = float(np.clip(-relative@relative_velocity/vv, 0, HORIZON_S)) if vv > 1e-8 else 0.
            closest = relative+closest_t*relative_velocity
            closest_m = float(np.linalg.norm(closest))
            lateral_mps = float(abs(relative_velocity[1]))
            predictions.append({"point_xy_m": point.round(4).tolist(),
                                "estimated_velocity_xy_mps": velocity.round(4).tolist(),
                                "estimated_relative_lateral_speed_mps": round(lateral_mps, 4),
                                "closest_approach_s": round(closest_t, 4),
                                "closest_approach_m": round(closest_m, 4),
                                "risk": (closest_m < CLEARANCE_WAIVER_M or
                                         lateral_mps < LATERAL_WAIVER_MPS)})
        # Untracked, near forward returns remain conservative: a sensor must
        # accumulate a motion history before it may waive a stop.
        tracked_near = [p for p in predictions
                        if np.linalg.norm(np.asarray(p["point_xy_m"])-ego_xy) < THRESHOLD_M+.15]
        risk = any(p["risk"] for p in tracked_near) if tracked_near else True
        return risk, {"track_samples": len(self.track), "near_predictions": tracked_near}


TRACKER = ScanTracker()
ORIGINAL_READ_RAYS = run.read_rays


def forecast_read_rays(model, data, name, pose, case, t):
    packet = ORIGINAL_READ_RAYS(model, data, name, pose, case, t)
    if name != "low":
        return packet
    raw = packet["min_forward_range_m"]
    risk, forecast = TRACKER.risk(packet, pose, t)
    packet["raw_min_forward_range_m"] = raw
    packet["forecast"] = forecast
    packet["forecast_risk"] = risk
    if not risk:
        packet["min_forward_range_m"] = None
    return packet


def main() -> None:
    global TRACKER
    OUT.mkdir(parents=True, exist_ok=True)
    xml_path = OUT / "scene.xml"
    xml_path.write_text(run.make_scene())
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    saved = np.load(run.TRACE, allow_pickle=False)
    stop_report = json.loads(run.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop_report["rows"] if r["case"] == "forward-stop-late"}
    if set(delays) != set(range(4)):
        raise ValueError("Missing native worlds")
    run.hazard_xy = crossing_hazard
    run.read_rays = forecast_read_rays
    run.STOP_THRESHOLD_M = THRESHOLD_M
    run.OUT = OUT
    results = []
    for world in range(4):
        for case in CASES:
            TRACKER = ScanTracker()
            video = world == 0 and case in ("crossing_clear", "low_pet")
            result = run.run_case(model, saved, world, case, "low15",
                                  delays[world], video=video)
            results.append(result)
            print(world, case, result["stop_command_s"],
                  result["min_modeled_body_gap_m"], flush=True)
    report = {
        "schema": "robot-learning-lab.native-stop-sensing.motion-gate.v1",
        "task": "T-7f9ef6",
        "purpose": "Development test of scan-tracked closest-approach gate atop 75 cm low stream",
        "actor_inputs": "low ray ranges, ideal root yaw and XY odometry; no hazard type, script, or evaluator geometry",
        "threshold_m": THRESHOLD_M,
        "track_association_m": TRACK_ASSOCIATION_M,
        "collision_radius_m": COLLISION_RADIUS_M,
        "clearance_waiver_m": CLEARANCE_WAIVER_M,
        "lateral_waiver_mps": LATERAL_WAIVER_MPS,
        "forecast_horizon_s": HORIZON_S,
        "scene_sha256": run.sha(xml_path),
        "v3_runner_sha256": run.sha(Path(run.__file__)),
        "source_sha256": run.sha(Path(__file__)),
        "native_trace_sha256": run.sha(run.TRACE),
        "body_sha256": run.sha(run.BODY),
        "mujoco_version": mujoco.__version__,
        "results": results,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    saved.close()


if __name__ == "__main__":
    main()
