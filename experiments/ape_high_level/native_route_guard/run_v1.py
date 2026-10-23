"""Full-scan route guard on historical 50 cm native stop tails.

Actor information is ideal range packets plus explicitly ideal root odometry.
The evaluator, hazard scripts and exact full-body geometry remain inside the
shared replay. This is a development guard, not a physical safety controller.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/ape_high_level/native_stop_sensing"
sys.path.insert(0, str(SOURCE))
import run as replay  # noqa: E402
import holdout_close  # noqa: E402


OUT = ROOT / "artifacts/ape-native-route-guard/v1-close"
RISK_RADIUS_M = .27
UNKNOWN_FORWARD_M = .55
UNKNOWN_LATERAL_M = .40
TRACK_ASSOCIATION_M = .22
HORIZON_S = 2.5
STOP_SIGNAL_M = .01


class FullScanGuard:
    def __init__(self):
        self.tracks: list[list[tuple[float, np.ndarray]]] = []
        self.previous_pose: tuple[float, np.ndarray] | None = None

    def update(self, packet: dict, pose: np.ndarray, t: float) -> dict:
        yaw = replay.yaw_from_quat(pose[3:7])
        ch, sh = math.cos(yaw), math.sin(yaw)
        origin = pose[:2] + np.array([.08*ch, .08*sh])
        points: list[np.ndarray] = []
        for i, distance in enumerate(packet["ranges_m"]):
            if distance is None:
                continue
            angle = yaw + math.radians(-70 + i*8.75)
            point = origin + distance*np.array([math.cos(angle), math.sin(angle)])
            if point[0] > pose[0]-.05 and abs(point[1]-pose[1]) <= .85:
                points.append(point)
        # Consecutive scan rays with similar endpoints form a visible surface.
        clusters: list[list[np.ndarray]] = []
        for point in points:
            if clusters and np.linalg.norm(point-clusters[-1][-1]) <= .16:
                clusters[-1].append(point)
            else:
                clusters.append([point])
        observations = [min(cluster, key=lambda p: p[1]) for cluster in clusters]

        available = set(range(len(observations)))
        updated = []
        for track in self.tracks:
            if not available:
                continue
            choice = min(available, key=lambda i: np.linalg.norm(observations[i]-track[-1][1]))
            if np.linalg.norm(observations[choice]-track[-1][1]) <= TRACK_ASSOCIATION_M:
                updated.append((track+[(t, observations[choice])])[-7:])
                available.remove(choice)
        for i in available:
            updated.append([(t, observations[i])])
        self.tracks = updated

        ego_xy = np.asarray(pose[:2])
        ego_v = np.array([.35*ch, .35*sh])
        if self.previous_pose:
            pt, pxy = self.previous_pose
            if t > pt:
                ego_v = (ego_xy-pxy)/(t-pt)
        self.previous_pose = (t, ego_xy)

        details = []
        risk = False
        for track in self.tracks:
            point = track[-1][1]
            relative = point-ego_xy
            dx, dy = map(float, relative)
            mature = len(track) >= 3 and track[-1][0]-track[0][0] >= .12
            if mature:
                times = np.array([row[0] for row in track])
                xy = np.array([row[1] for row in track])
                centered = times-times.mean()
                velocity = (centered[:, None]*(xy-xy.mean(axis=0))).sum(axis=0)/(centered@centered)
                rv = velocity-ego_v
                vv = float(rv@rv)
                cpa_s = float(np.clip(-(relative@rv)/vv, 0, HORIZON_S)) if vv > 1e-8 else 0.
                cpa_m = float(np.linalg.norm(relative+cpa_s*rv))
                threat = dx > -.10 and cpa_m < RISK_RADIUS_M
            else:
                velocity = np.array([np.nan, np.nan])
                cpa_s = None
                cpa_m = None
                threat = (0 <= dx <= UNKNOWN_FORWARD_M and abs(dy) <= UNKNOWN_LATERAL_M)
            risk |= threat
            details.append({"point_xy_m": point.round(4).tolist(),
                            "samples": len(track),
                            "velocity_xy_mps": velocity.round(4).tolist() if mature else None,
                            "closest_approach_s": round(cpa_s, 4) if cpa_s is not None else None,
                            "closest_approach_m": round(cpa_m, 4) if cpa_m is not None else None,
                            "risk": bool(threat)})
        return {"risk": bool(risk), "tracks": details}


GUARD = FullScanGuard()
ORIGINAL_READ = replay.read_rays


def read_with_guard(model, data, name, pose, case, t):
    packet = ORIGINAL_READ(model, data, name, pose, case, t)
    if name == "low":
        packet["raw_min_forward_range_m"] = packet["min_forward_range_m"]
        decision = GUARD.update(packet, pose, t)
        packet["route_guard"] = decision
        packet["min_forward_range_m"] = STOP_SIGNAL_M if decision["risk"] else None
    return packet


def overlay_guard(frame: np.ndarray, row: dict, case: str, mode: str) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    panel = np.zeros((82, image.shape[1], 3), np.uint8)
    panel[:] = (26, 28, 31)
    lines = [f"{case} | {mode} | t={row['time_s']:.2f}s | {row['action']}",
             f"guard signal={row['min_sensor_range_m']}  model gap={row['evaluator_gap_m']:.3f} m",
             f"stop command={row['stop_command_s']} s  (historical native-tail transfer)"]
    for n, line in enumerate(lines):
        cv2.putText(panel, line, (12, 25+n*24), cv2.FONT_HERSHEY_SIMPLEX,
                    .54, (220, 230, 232), 1, cv2.LINE_AA)
    return np.concatenate((image, panel), axis=0)


def main() -> None:
    global GUARD
    OUT.mkdir(parents=True, exist_ok=True)
    xml_path = OUT / "scene.xml"
    xml_path.write_text(replay.make_scene())
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    saved = np.load(replay.TRACE, allow_pickle=False)
    stop = json.loads(replay.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop["rows"] if r["case"] == "forward-stop-late"}
    replay.hazard_xy = holdout_close.hazard
    replay.read_rays = read_with_guard
    replay.overlay = overlay_guard
    replay.STOP_THRESHOLD_M = .75
    replay.OUT = OUT
    results = []
    for world in range(4):
        for case in holdout_close.CASES:
            GUARD = FullScanGuard()
            result = replay.run_case(model, saved, world, case, "low15", delays[world],
                                     video=(world == 0))
            results.append(result)
            print(world, case, result["stop_command_s"],
                  result["min_modeled_body_gap_m"], flush=True)
    report = {"schema": "robot-learning-lab.native-route-guard.v1",
              "task": "T-9v3p3n",
              "purpose": "Full low scan point tracking against a route corridor on frozen V9 close crossings",
              "actor_inputs": "ideal low ray packets and root XY/yaw odometry only",
              "risk_radius_m": RISK_RADIUS_M,
              "unknown_forward_m": UNKNOWN_FORWARD_M,
              "unknown_lateral_m": UNKNOWN_LATERAL_M,
              "stop_signal_m": STOP_SIGNAL_M,
              "scene_sha256": replay.sha(xml_path),
              "source_sha256": replay.sha(Path(__file__)),
              "runner_sha256": replay.sha(Path(replay.__file__)),
              "case_source_sha256": replay.sha(Path(holdout_close.__file__)),
              "native_trace_sha256": replay.sha(replay.TRACE),
              "body_sha256": replay.sha(replay.BODY),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    saved.close()


if __name__ == "__main__":
    main()
