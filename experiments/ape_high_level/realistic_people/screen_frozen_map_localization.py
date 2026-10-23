"""Reject or retain a frozen range-map pose correction on saved Isaac actions.

This is an actor-input replay, not a closed-loop route: only commanded eye
motions, calibrated initial pose and unlabeled range beams enter the estimate.
Saved proxy positions are read afterward for evaluation. The static-return
filter and matcher gates are diagnostic settings, not a deployable estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


HEIGHT_M = .80
STATIC_RETURN_MIN_M = 2.3
STATIC_RETURN_MAX_M = 6.0
MAX_CORRECTION_M = .15
MIN_INLIER_FRACTION = .45
INLIER_DISTANCE_M = .15
PRIOR_WEIGHT = .08


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoints(frame: dict, xy: np.ndarray, heading_deg: float) -> np.ndarray:
    plane = next(row for row in frame["planes"]
                 if abs(row["height_above_floor_m"] - HEIGHT_M) < 1e-3)
    points = []
    for beam in plane["beams"]:
        distance = beam["distance_m"]
        if distance is None or not STATIC_RETURN_MIN_M <= distance < STATIC_RETURN_MAX_M:
            continue
        angle = math.radians(heading_deg + beam["angle_deg"])
        points.append(xy + distance * np.array([math.sin(angle), math.cos(angle)]))
    return np.asarray(points, dtype=float).reshape(-1, 2)


def replay(run: Path, odometry_scale: float) -> dict:
    evidence = run / "evidence"
    range_path = evidence / "range-fan-trace.json"
    action_path = evidence / "online-action-trace.json"
    scene_path = evidence / "scene-info.json"
    ranges = json.loads(range_path.read_text())["frames"]
    actions = json.loads(action_path.read_text())["actions"]
    scene = json.loads(scene_path.read_text())
    if len(ranges) != len(actions) or len(actions) < 2:
        raise ValueError("Range and action streams must be aligned and nonempty")
    initial = np.asarray(scene["camera_eye_xyz_m"][:2], dtype=float)
    odometry = initial.copy()
    corrected = initial.copy()
    heading_deg = 0.0
    first = endpoints(ranges[0], initial, heading_deg)
    if len(first) < 12:
        raise ValueError("Initial upper-plane static-return map is too sparse")
    map_tree = cKDTree(first)
    rows = []
    for frame in range(1, len(actions)):
        command = actions[frame - 1]["command_for_next_frame"]
        heading_deg += command["yaw_delta_deg"]
        distance = odometry_scale * command["forward_step_m"]
        delta = distance * np.array([math.sin(math.radians(heading_deg)),
                                     math.cos(math.radians(heading_deg))])
        odometry += delta
        corrected += delta
        current = endpoints(ranges[frame], np.zeros(2), heading_deg)
        offset = np.zeros(2)
        accepted = False
        inlier_fraction = None
        if len(current) >= 12:
            def residual(candidate: np.ndarray) -> float:
                distances, _ = map_tree.query(current + corrected + candidate)
                trimmed = np.sort(np.minimum(distances, .25))[:
                    max(10, int(.7 * len(distances)))]
                return float(np.mean(trimmed ** 2) + PRIOR_WEIGHT * (candidate @ candidate))

            optimized = minimize(residual, np.zeros(2), method="Nelder-Mead",
                                 options={"maxiter": 150})
            offset = np.asarray(optimized.x)
            distances, _ = map_tree.query(current + corrected + offset)
            inlier_fraction = float(np.mean(distances < INLIER_DISTANCE_M))
            accepted = (np.linalg.norm(offset) < MAX_CORRECTION_M and
                        inlier_fraction > MIN_INLIER_FRACTION)
            if accepted:
                corrected += offset
        truth = np.asarray(actions[frame - 1]["proxy_eye_xyz_m_evaluator_only"][:2])
        rows.append({"frame": frame,
                     "source_time_s": ranges[frame]["timeline_time_s"],
                     "static_return_count": len(current),
                     "suggested_offset_xy_m": offset.tolist(),
                     "inlier_fraction": inlier_fraction,
                     "correction_accepted": accepted,
                     "odometry_error_m_evaluator_only": float(np.linalg.norm(odometry - truth)),
                     "corrected_error_m_evaluator_only": float(np.linalg.norm(corrected - truth))})
    return {"run": str(run), "odometry_scale": odometry_scale,
            "initial_static_returns": len(first),
            "accepted_corrections": sum(row["correction_accepted"] for row in rows),
            "final_odometry_error_m": rows[-1]["odometry_error_m_evaluator_only"],
            "final_corrected_error_m": rows[-1]["corrected_error_m_evaluator_only"],
            "inputs_sha256": {path.name: sha(path)
                              for path in (range_path, action_path, scene_path)},
            "frames": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {"scope": "Offline preflight of frozen-map registration on saved Isaac sensor-driven camera paths; no planner feedback or physical body",
              "method": {"upper_plane_height_m": HEIGHT_M,
                         "candidate_static_return_m": [STATIC_RETURN_MIN_M,
                                                        STATIC_RETURN_MAX_M],
                         "max_correction_m": MAX_CORRECTION_M,
                         "minimum_inlier_fraction": MIN_INLIER_FRACTION,
                         "inlier_distance_m": INLIER_DISTANCE_M,
                         "prior_weight": PRIOR_WEIGHT,
                         "map_update": "none"},
              "results": [replay(run, scale) for run in args.runs
                          for scale in (1.0, .8)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for result in report["results"]:
        print(Path(result["run"]).name, result["odometry_scale"],
              "odom_cm", round(100 * result["final_odometry_error_m"], 1),
              "frozen_map_cm", round(100 * result["final_corrected_error_m"], 1))


if __name__ == "__main__":
    main()
