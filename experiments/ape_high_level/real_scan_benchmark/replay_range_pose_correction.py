"""Replay frozen-map range correction on native-shaped real-room owner traces.

This is a sensor replay. The saved actor commands and photographs do not change
when localization changes, so a live owner-route rerun is still required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/cozy_apartment_owner_screen"
CROSS = ROOT / "artifacts/ape-real-room-next-captures-20261014/native_pose_crossroom_owner_screen"
PRECOMMIT = ROOT / "artifacts/ape-real-room-next-captures-20261014/range_pose_correction_precommit.json"
OUT = ROOT / "artifacts/ape-real-room-next-captures-20261014/range_pose_correction_report.json"
SOURCE = "varied150-forward-seed144"


def points(frame: dict) -> np.ndarray:
    heading = float(frame["actor_heading_deg"])
    bearing = frame["owner_actor"].get("owner_bearing_deg")
    xy = []
    for distances in (frame["range_low_m"], frame["range_mid_m"]):
        for angle, distance in zip(frame["range_angles_deg"], distances):
            if (distance is None or not .2 <= distance <= 3. or
                    (bearing is not None and
                     abs((angle + float(bearing) + 180.) % 360. - 180.) <= 25.)):
                continue
            direction = math.radians(heading + angle)
            xy.append([distance * math.sin(direction),
                       distance * math.cos(direction)])
    return np.asarray(xy, dtype=float).reshape(-1, 2)


def replay(entry: dict) -> dict:
    trace = json.loads((ROOT / entry["trace"]).read_text())
    frames = trace["trace"]
    source_route = json.loads((ROOT / trace["source_capture"]).read_text())
    start = source_route.get("start_xz")
    if start is None:
        eye = source_route["frames"][0]["eye_m"]
        start = [eye[0], eye[2]]
    start = np.asarray(start, dtype=float)
    first_map = points(frames[0])
    tree = cKDTree(first_map) if len(first_map) >= 20 else None
    prior_native = np.asarray(frames[0]["actor_pose_xy_m"], dtype=float)
    prior_actual = start.copy()
    corrected = prior_native.copy()
    rows = []
    for frame in frames:
        native = np.asarray(frame["actor_pose_xy_m"], dtype=float)
        if frame is not frames[0]:
            corrected += native - prior_native
        prior_native = native
        scan = points(frame)
        map_distance = (tree.query(scan + corrected)[0]
                        if tree is not None and len(scan) else np.array([]))
        supported = map_distance <= .08
        support_count = int(supported.sum())
        support_fraction = float(supported.mean()) if len(supported) else 0.
        status = "weak_initial_map" if tree is None else "weak_static_scan"
        offset = np.zeros(2)
        inlier_fraction = 0.
        if tree is not None and support_count >= 20 and support_fraction >= .48:
            stable = scan[supported]

            def residual(candidate: np.ndarray) -> float:
                distances, _ = tree.query(stable + corrected + candidate)
                nearest = np.sort(distances)[:max(16, int(.7 * len(distances)))]
                return float(np.mean(np.minimum(nearest, .12) ** 2) +
                             .08 * np.sum(candidate ** 2))

            answer = minimize(residual, np.zeros(2), method="Nelder-Mead",
                              options={"xatol": 1e-4, "fatol": 1e-7,
                                       "maxiter": 100})
            offset = np.asarray(answer.x)
            distances, _ = tree.query(stable + corrected + offset)
            inlier_fraction = float(np.mean(distances < .10))
            if np.linalg.norm(offset) <= .13 and inlier_fraction >= .48:
                corrected += offset
                status = "registered"
            else:
                status = "registration_rejected"
        actual_capture = np.asarray(frame.get(
            "evaluator_actual_xz_at_capture_m", prior_actual), dtype=float)
        truth = actual_capture - start
        prior_actual = np.asarray(frame["evaluator_actual_xz_m"], dtype=float)
        rows.append({"frame": frame["frame"],
                     "native_pose_xy_m": native.tolist(),
                     "corrected_pose_xy_m": corrected.tolist(),
                     "range_points": len(scan),
                     "static_support_count": support_count,
                     "static_support_fraction": support_fraction,
                     "status": status,
                     "suggested_offset_m": offset.tolist(),
                     "inlier_fraction": inlier_fraction,
                     "native_error_m_evaluator_only": float(np.linalg.norm(native - truth)),
                     "corrected_error_m_evaluator_only": float(np.linalg.norm(corrected - truth))})
    return {"name": entry["name"], "scene_id": entry["scene_id"],
            "source_trace": entry["trace"],
            "initial_map_points": len(first_map),
            "frames": len(rows),
            "registered_frames": sum(r["status"] == "registered" for r in rows),
            "rejected_frames": sum(r["status"] != "registered" for r in rows),
            "maximum_native_error_m": max(r["native_error_m_evaluator_only"] for r in rows),
            "maximum_corrected_error_m": max(r["corrected_error_m_evaluator_only"] for r in rows),
            "final_native_error_m": rows[-1]["native_error_m_evaluator_only"],
            "final_corrected_error_m": rows[-1]["corrected_error_m_evaluator_only"],
            "wrong_accepted_frames": [r["frame"] for r in rows if
                r["status"] == "registered" and
                r["corrected_error_m_evaluator_only"] >
                r["native_error_m_evaluator_only"] + .02],
            "observations": rows}


def main() -> None:
    json.loads(PRECOMMIT.read_text())
    paths = [BASE / f"owner_batch_native_shaped_arbitrated_rangeprojected_farconverse_{SOURCE}_v1.json",
             CROSS / f"owner_batch_native_shaped_arbitrated_rangeprojected_farconverse_{SOURCE}_v1.json"]
    results = []
    for path in paths:
        batch = json.loads(path.read_text())
        results.extend(replay(entry) for entry in batch["scenes"])
    output = {"schema": "ape.real-capture-frozen-range-pose-replay.v1",
              "precommit": str(PRECOMMIT.relative_to(ROOT)),
              "scope": "Four existing native-shaped owner routes; frozen first-frame two-height range map, no simulator truth in corrections, ideal yaw, offline actor-input replay only.",
              "source_batches": [str(p.relative_to(ROOT)) for p in paths],
              "results": results}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    for row in results:
        print(row["name"], row["registered_frames"], "registered",
              "native", round(row["maximum_native_error_m"], 3),
              "range", round(row["maximum_corrected_error_m"], 3),
              "wrong accepted", row["wrong_accepted_frames"])


if __name__ == "__main__":
    main()
