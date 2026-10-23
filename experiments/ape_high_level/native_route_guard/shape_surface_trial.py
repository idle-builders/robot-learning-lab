"""V25: fit circular visible surfaces before tracking, with V24 skill horizon.

This tests whether silhouette-point sliding explains V24's remaining false
crossing stop. It is a geometry-model ablation, not a person detector.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import command_route_trial as command
import dense_low_trial as dense
import online_cluster_guard as online
import online_corridor_trial as corridor
import plan_bounded_trial as plan
import run as guard
from kitchen_trial import sha


OUT = online.kitchen.OUT / "v25-surface-shape"
CIRCLE_RADIUS_M = (.04, .24)
CIRCLE_MAX_RESIDUAL_M = .008
OriginalOnlineGuard = online.OnlineClusterGuard


def surface_estimate(xy: np.ndarray) -> tuple[np.ndarray, float, str, float | None]:
    median = np.median(xy, axis=0)
    fallback_extent = float(np.max(np.linalg.norm(xy-median, axis=1)))
    if len(xy) < 4:
        return median, fallback_extent, "median", None
    local = xy-median
    coefficients, *_ = np.linalg.lstsq(
        np.column_stack((2*local, np.ones(len(local)))),
        np.sum(local*local, axis=1), rcond=None)
    center = median+coefficients[:2]
    radius2 = coefficients[2]+float(coefficients[:2]@coefficients[:2])
    if radius2 <= 0:
        return median, fallback_extent, "median", None
    radius = float(np.sqrt(radius2))
    residual = float(np.max(np.abs(np.linalg.norm(xy-center, axis=1)-radius)))
    if CIRCLE_RADIUS_M[0] <= radius <= CIRCLE_RADIUS_M[1] and \
            residual <= CIRCLE_MAX_RESIDUAL_M:
        return center, radius, "circle", residual
    return median, fallback_extent, "median", residual


class ShapeSurfaceGuard(OriginalOnlineGuard):
    def update(self, packet: dict, pose: np.ndarray, t: float) -> dict:
        points = dense.dense_points(packet, pose)
        clusters: list[list[np.ndarray]] = []
        for point in points:
            if clusters and np.linalg.norm(point-clusters[-1][-1]) <= online.CLUSTER_LINK_M:
                clusters[-1].append(point)
            else:
                clusters.append([point])
        observed = []
        for cluster in clusters:
            xy = np.asarray(cluster)
            center, extent, model, residual = surface_estimate(xy)
            observed.append({"center": center, "extent": extent,
                             "ray_points": len(cluster),
                             "surface_model": model,
                             "circle_residual_m": residual})

        available = set(range(len(observed)))
        alive = []
        for track in self.tracks:
            if not available or t-track["history"][-1][0] > online.TRACK_STALE_S:
                continue
            latest = track["history"][-1][1]
            nearest = min(available, key=lambda j: float(np.linalg.norm(
                observed[j]["center"]-latest)))
            if np.linalg.norm(observed[nearest]["center"]-latest) <= \
                    online.TRACK_ASSOCIATION_M:
                item = observed[nearest]
                track["history"] = (track["history"] +
                                    [(t, item["center"], item["extent"],
                                      item["ray_points"], item["surface_model"],
                                      item["circle_residual_m"])])[-online.MAX_TRACK_SAMPLES:]
                alive.append(track)
                available.remove(nearest)
        for j in sorted(available):
            item = observed[j]
            alive.append({"id": self.next_id,
                          "history": [(t, item["center"], item["extent"],
                                       item["ray_points"], item["surface_model"],
                                       item["circle_residual_m"])]})
            self.next_id += 1
        self.tracks = alive

        ego = np.asarray(pose[:2])
        skill_remaining_s = max(0., plan.SKILL_END_S-t)
        horizon_s = min(guard.HORIZON_S,
                        skill_remaining_s+plan.STOP_EQUIVALENT_S)
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
                                           0., horizon_s)) if vv > 1e-8 else 0.)
                closest_m = float(np.linalg.norm(relative+closest_s*relative_velocity))
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
                            "surface_model": history[-1][4],
                            "circle_residual_m": (round(history[-1][5], 5)
                                                  if history[-1][5] is not None else None),
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
                "planned_velocity_xy_mps": command.COMMAND_VELOCITY_XY_MPS.tolist(),
                "skill_remaining_s": round(skill_remaining_s, 5),
                "forecast_horizon_s": round(horizon_s, 5),
                "stop_tail_advance_budget_m": plan.HISTORICAL_STOP_ADVANCE_M}


def main() -> None:
    old_class = online.OnlineClusterGuard
    old_out = corridor.OUT
    online.OnlineClusterGuard = ShapeSurfaceGuard
    corridor.OUT = OUT
    try:
        corridor.main()
    finally:
        online.OnlineClusterGuard = old_class
        corridor.OUT = old_out
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.ape-robocasa-shape-surface-guard.v1"
    report["method"] += (" V25 development-only change relative to V24:"
                         " fit a circular arc when at least four range"
                         " endpoints support a 4-24cm radius with <=8mm"
                         " maximum radial residual; otherwise use the"
                         " visible-surface median. The same requested-skill"
                         " bounded horizon and historical stop travel apply.")
    report["algorithm"]["circle_radius_m"] = list(CIRCLE_RADIUS_M)
    report["algorithm"]["circle_max_residual_m"] = CIRCLE_MAX_RESIDUAL_M
    report["shape_surface_source_sha256"] = sha(Path(__file__))
    report["v24_report_sha256"] = sha(plan.OUT / "report.json")
    report["development_on_v24_failure"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
