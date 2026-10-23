"""V22: actor-side cluster memory with no hazard-free prior room map.

Visible low-scan surfaces are clustered and associated in world coordinates
using explicitly ideal root pose. Unknown occupancy is never declared free.
This remains a nominal route forecast, not a swept physical safety layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

import changed_layout_trial as cases
import command_route_trial as command
import dense_low_trial as dense
import kitchen_native as kitchen
import kitchen_trial as trial
import near_clutter_trial as clutter
import run as guard


OUT = kitchen.OUT / "v22-online-clusters"
NO_MAP_SENTINEL_XY_M = np.array([[999., 999.]])
CLUSTER_LINK_M = .16
TRACK_ASSOCIATION_M = .28
TRACK_STALE_S = .30
MAX_TRACK_SAMPLES = 7
ORIGINAL_READ = guard.ORIGINAL_READ
ORIGINAL_POINTS = guard.low_ray_points
ORIGINAL_HAZARD = trial.hazard_xy


class OnlineClusterGuard:
    def __init__(self, ignored_background: np.ndarray):
        if not np.array_equal(ignored_background, NO_MAP_SENTINEL_XY_M):
            raise ValueError("Online guard must not receive a room survey")
        # The common episode runner queries this logging-only tree. It contains
        # no scene point and is not read by the decision below.
        self.background = cKDTree(NO_MAP_SENTINEL_XY_M)
        self.tracks: list[dict] = []
        self.next_id = 1

    def update(self, packet: dict, pose: np.ndarray, t: float) -> dict:
        points = dense.dense_points(packet, pose)
        clusters: list[list[np.ndarray]] = []
        for point in points:
            if clusters and np.linalg.norm(point-clusters[-1][-1]) <= CLUSTER_LINK_M:
                clusters[-1].append(point)
            else:
                clusters.append([point])
        observed = []
        for cluster in clusters:
            xy = np.asarray(cluster)
            center = np.median(xy, axis=0)
            extent = float(np.max(np.linalg.norm(xy-center, axis=1)))
            observed.append({"center": center, "extent": extent,
                             "ray_points": len(cluster)})

        available = set(range(len(observed)))
        alive = []
        for track in self.tracks:
            if not available or t-track["history"][-1][0] > TRACK_STALE_S:
                continue
            latest = track["history"][-1][1]
            nearest = min(available, key=lambda j:
                          float(np.linalg.norm(observed[j]["center"]-latest)))
            if np.linalg.norm(observed[nearest]["center"]-latest) <= TRACK_ASSOCIATION_M:
                item = observed[nearest]
                track["history"] = (track["history"] +
                                    [(t, item["center"], item["extent"],
                                      item["ray_points"])])[-MAX_TRACK_SAMPLES:]
                alive.append(track)
                available.remove(nearest)
        for j in sorted(available):
            item = observed[j]
            alive.append({"id": self.next_id,
                          "history": [(t, item["center"], item["extent"],
                                       item["ray_points"])]})
            self.next_id += 1
        self.tracks = alive

        ego = np.asarray(pose[:2])
        details = []
        risk = False
        for track in self.tracks:
            history = track["history"]
            center = history[-1][1]
            extent = min(.16, history[-1][2])
            relative = center-ego
            mature = len(history) >= 3 and history[-1][0]-history[0][0] >= .12
            if mature:
                times = np.asarray([sample[0] for sample in history])
                positions = np.asarray([sample[1] for sample in history])
                ct = times-times.mean()
                velocity = (ct[:, None]*(positions-positions.mean(axis=0))).sum(
                    axis=0)/(ct@ct)
                relative_velocity = velocity-command.COMMAND_VELOCITY_XY_MPS
                vv = float(relative_velocity@relative_velocity)
                closest_s = (float(np.clip(-float(relative@relative_velocity)/vv,
                                           0., guard.HORIZON_S)) if vv > 1e-8 else 0.)
                closest_m = float(np.linalg.norm(relative+closest_s*relative_velocity))
                # The local surface footprint is a proxy, not a calibrated body hull.
                threshold = guard.RISK_RADIUS_M+extent
                threatened = bool(relative[0] > -.10 and closest_m < threshold)
            else:
                velocity = None
                closest_s = None
                closest_m = None
                threshold = None
                threatened = bool(0 <= relative[0] <= guard.UNKNOWN_FORWARD_M and
                                  abs(relative[1]) <= guard.UNKNOWN_LATERAL_M+extent)
            risk |= threatened
            details.append({"id": track["id"],
                            "center_xy_m": center.round(5).tolist(),
                            "surface_extent_m": round(extent, 5),
                            "samples": len(history),
                            "ray_points": history[-1][3],
                            "velocity_xy_mps": (velocity.round(5).tolist()
                                                if velocity is not None else None),
                            "closest_approach_s": (round(closest_s, 5)
                                                   if closest_s is not None else None),
                            "closest_approach_m": (round(closest_m, 5)
                                                   if closest_m is not None else None),
                            "risk_threshold_m": (round(threshold, 5)
                                                 if threshold is not None else None),
                            "risk": threatened})
        return {"risk": bool(risk), "tracks": details,
                "scan_clusters": len(observed),
                "online_track_count": len(self.tracks),
                "map_source": "none_online_clusters",
                "planned_velocity_xy_mps": command.COMMAND_VELOCITY_XY_MPS.tolist()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(clutter.OUT / "scene.xml"))
    stop = json.loads(guard.replay.STOP_REPORT.read_text())
    delays = {row["world"]: row["command_to_observed_stop_s"]
              for row in stop["rows"] if row["case"] == "forward-stop-late"}
    trial.OUT = OUT
    trial.hazard_xy = cases.hazard_xy
    guard.ORIGINAL_READ = dense.dense_read
    guard.low_ray_points = dense.dense_points
    guard.FullScanGuard = OnlineClusterGuard
    results = []
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            for world in range(4):
                for case in cases.CASES:
                    video = world == 0 and case in ("fast_clear_held",
                                                      "slow_cross_held",
                                                      "pause_path_held",
                                                      "fast_right_held",
                                                      "slow_right_held")
                    row = trial.run_episode(model, saved, NO_MAP_SENTINEL_XY_M,
                                            world, case, "mapped_low_guard",
                                            delays[world], video=video)
                    results.append(row)
                    print(world, case, row["stop_command_s"],
                          row["min_modeled_body_person_gap_m"], flush=True)
    finally:
        trial.hazard_xy = ORIGINAL_HAZARD
        guard.ORIGINAL_READ = ORIGINAL_READ
        guard.low_ray_points = ORIGINAL_POINTS
        guard.FullScanGuard = command.OriginalGuard
    report = {"schema": "robot-learning-lab.ape-robocasa-online-cluster-guard.v1",
              "task": "T-9v3p3n", "body_status": "historical_rejected_direct_frame",
              "method": "Same V20 layout3 near clutter, six scripts and native stop tails, but no hazard-free room survey. Online cluster centers/extent and short histories come only from 15Hz 73-ray 1.2m low packets plus ideal root pose; nominal +0.35m/s route forecast commands the historical stop-tail. The dummy (999,999)m point exists only for the old runner's logging tree and is rejected as a real map by the actor constructor.",
              "algorithm": {"cluster_link_m": CLUSTER_LINK_M,
                            "track_association_m": TRACK_ASSOCIATION_M,
                            "track_stale_s": TRACK_STALE_S,
                            "max_track_samples": MAX_TRACK_SAMPLES,
                            "unknown_forward_m": guard.UNKNOWN_FORWARD_M,
                            "unknown_lateral_m": guard.UNKNOWN_LATERAL_M,
                            "risk_radius_m": guard.RISK_RADIUS_M,
                            "no_map_sentinel_xy_m": NO_MAP_SENTINEL_XY_M.tolist()},
              "scene_sha256": trial.sha(clutter.OUT / "scene.xml"),
              "source_sha256": trial.sha(Path(__file__)),
              "dense_source_sha256": trial.sha(Path(dense.__file__)),
              "episode_runner_sha256": trial.sha(Path(trial.__file__)),
              "command_guard_sha256": trial.sha(Path(command.__file__)),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "body_sha256": trial.sha(kitchen.BODY),
              "v20_report_sha256": trial.sha(clutter.OUT / "report.json"),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
