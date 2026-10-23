"""Compare head RGB-D scene motion with body-only post-stop view gates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-onboard-motion-v1"
BODY_RESULT = ROOT / "artifacts/ape-stop-observability-v1/result.json"
CASES = ("forward-stop-early", "forward-stop-late")
FOV_DEG = 67.0
MIN_MATCHED_DEPTH = 80
MIN_INLIERS = 60
MIN_INLIER_FRACTION = .5
QUIET_TRANSLATION_M_PER_PACKET = .002  # 2 mm / 100 ms, exploratory.
WINDOW = 4  # 400 ms at 10 Hz.


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_motion(rgb: np.ndarray, depth: np.ndarray) -> list[dict]:
    height, width = rgb.shape[1:3]
    # MJCF camera fovy is vertical. MuJoCo Renderer returns optical-axis Z
    # distance after undoing the OpenGL projection, not radial range.
    focal = height / (2 * math.tan(math.radians(FOV_DEG / 2)))
    camera_matrix = np.asarray([[focal, 0, width / 2],
                                [0, focal, height / 2], [0, 0, 1]], dtype=float)
    detector = cv2.ORB_create(2500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    keypoints, descriptors = [], []
    for frame in rgb:
        key, desc = detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), None)
        keypoints.append(key)
        descriptors.append(desc)
    records = [{"status": "first_frame", "feature_count": len(keypoints[0]),
                "matched_depth_features": 0, "pnp_inliers": 0,
                "pnp_inlier_fraction": 0.0, "camera_translation_m": None}]
    for i in range(1, len(rgb)):
        record = {"status": "insufficient_features", "feature_count": len(keypoints[i]),
                  "matched_depth_features": 0, "pnp_inliers": 0,
                  "pnp_inlier_fraction": 0.0, "camera_translation_m": None}
        old_desc, new_desc = descriptors[i - 1], descriptors[i]
        if old_desc is None or new_desc is None:
            records.append(record)
            continue
        objects, pixels = [], []
        for pair in matcher.knnMatch(old_desc, new_desc, k=2):
            if len(pair) != 2:
                continue
            candidate, runner_up = pair
            if candidate.distance >= .72 * runner_up.distance:
                continue
            u, v = keypoints[i - 1][candidate.queryIdx].pt
            z = float(depth[i - 1, min(int(v), height - 1), min(int(u), width - 1)])
            if not .3 < z < 5.0:
                continue
            ray_x, ray_y = (u - width / 2) / focal, (v - height / 2) / focal
            objects.append((ray_x * z, ray_y * z, z))
            pixels.append(keypoints[i][candidate.trainIdx].pt)
        record["matched_depth_features"] = len(objects)
        if len(objects) < MIN_MATCHED_DEPTH:
            records.append(record)
            continue
        success, rotation, translation, inliers = cv2.solvePnPRansac(
            np.asarray(objects, dtype=np.float32), np.asarray(pixels, dtype=np.float32),
            camera_matrix, None, iterationsCount=400, reprojectionError=2.0,
            confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if not success or inliers is None:
            record["status"] = "pnp_failed"
            records.append(record)
            continue
        record["pnp_inliers"] = len(inliers)
        record["pnp_inlier_fraction"] = round(len(inliers) / len(objects), 4)
        if len(inliers) < MIN_INLIERS or len(inliers) / len(objects) < MIN_INLIER_FRACTION:
            record["status"] = "low_registration_quality"
            records.append(record)
            continue
        displacement = (-cv2.Rodrigues(rotation)[0].T @ translation).reshape(3)
        record["status"] = "registered"
        record["camera_translation_m"] = round(float(np.linalg.norm(displacement)), 6)
        records.append(record)
    return records


def first_camera_gate(times: np.ndarray, records: list[dict], stop_s: float) -> int | None:
    for end in range(len(times)):
        if times[end] < stop_s or end < WINDOW - 1:
            continue
        window = range(end - WINDOW + 1, end + 1)
        if all(times[j] >= stop_s and
               records[j]["status"] == "registered" and
               records[j]["camera_translation_m"] <= QUIET_TRANSLATION_M_PER_PACKET
               for j in window):
            return end
    return None


def evaluate() -> None:
    body = json.loads(BODY_RESULT.read_text())
    capture = json.loads((SOURCE / "capture-manifest.json").read_text())
    if body["current_design_pointer"]["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical body selection label changed")
    body_rows = {(r["case"], r["world"]): r for r in body["rows"]}
    captures = {(c["case"], c["world"]): c for c in capture["cases"]}
    rows, packets = [], []
    for case in CASES:
        for world in range(4):
            source = captures[(case, world)]
            path = ROOT / source["rgbd_path"]
            if sha(path) != source["rgbd_sha256"]:
                raise ValueError(f"Camera source changed: {case}, world {world}")
            data = np.load(path)
            times = data["time_s"].astype(float)
            records = match_motion(data["rgb"], data["depth_m"])
            camera_world = data["evaluator_camera_world_pos_m"].astype(float)
            evaluator_steps = np.r_[np.nan, np.linalg.norm(np.diff(camera_world, axis=0), axis=1)]
            if len(records) != 80 or not np.isclose(float(data["dt_s"]), .1):
                raise ValueError("Wrong camera sample timing")
            baseline = body_rows[(case, world)]
            stop_s = baseline["stop_command_s"]
            truth_s = baseline["evaluator_stationary_at_s"]
            body_s = baseline["gates"]["fused"]["first_accept_at_s"]
            vision_index = first_camera_gate(times, records, stop_s)
            # Simulate an actual featureless camera stream, rather than merely
            # telling the gate that the camera is unhealthy.
            masked_rgb = data["rgb"].copy()
            masked_rgb[times >= stop_s + .4] = 36
            loss_records = match_motion(masked_rgb, data["depth_m"])
            loss_index = first_camera_gate(times, loss_records, stop_s)
            fused_index = next((i for i in range(len(times)) if vision_index is not None and
                                i >= vision_index and times[i] >= body_s), None)
            if loss_index is not None:
                raise AssertionError("Feature loss must not authorize a post-stop camera view")
            def score(index: int | None) -> dict:
                timestamp = None if index is None else round(float(times[index]), 3)
                return {"first_accept_s": timestamp,
                        "lead_or_lag_vs_evaluator_s": None if timestamp is None else
                        round(timestamp - truth_s, 3),
                        "premature": bool(timestamp is not None and timestamp < truth_s),
                        "nonresponse": timestamp is None}
            rows.append({"case": case, "world": world, "split": "development" if world == 0 else "same_room_heldout_trajectory",
                         "stop_command_s": stop_s, "evaluator_stationary_s": truth_s,
                         "body_only": {"first_accept_s": body_s,
                                       "lead_or_lag_vs_evaluator_s": baseline["gates"]["fused"]["lead_or_lag_vs_truth_s"],
                                       "premature": baseline["gates"]["fused"]["premature_before_evaluator_stationary"],
                                       "nonresponse": False},
                         "rgbd_only": score(vision_index), "body_plus_rgbd": score(fused_index),
                         "rgbd_texture_lost": score(loss_index)})
            for i, record in enumerate(records):
                packets.append({"case": case, "world": world, "sample_at_s": round(float(times[i]), 3),
                                **record, "feature_count_after_texture_loss": loss_records[i]["feature_count"],
                                "evaluator_camera_translation_m": None if i == 0 else round(float(evaluator_steps[i]), 6),
                                "evaluator_absolute_error_m": None if i == 0 or record["camera_translation_m"] is None
                                else round(abs(record["camera_translation_m"] - float(evaluator_steps[i])), 6)})
    with (SOURCE / "camera-motion-packets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=packets[0].keys())
        writer.writeheader()
        writer.writerows(packets)
    summary = {}
    for condition in ("body_only", "rgbd_only", "body_plus_rgbd", "rgbd_texture_lost"):
        decisions = [r[condition] for r in rows]
        offsets = [d["lead_or_lag_vs_evaluator_s"] for d in decisions if not d["nonresponse"]]
        summary[condition] = {"accepted": len(offsets), "premature": sum(d["premature"] for d in decisions),
                              "nonresponse": sum(d["nonresponse"] for d in decisions),
                              "lead_or_lag_range_s": None if not offsets else [min(offsets), max(offsets)]}
    errors = [p["evaluator_absolute_error_m"] for p in packets if p["evaluator_absolute_error_m"] is not None]
    false_quiet = sum(p["camera_translation_m"] is not None and
                      p["camera_translation_m"] <= QUIET_TRANSLATION_M_PER_PACKET and
                      p["evaluator_camera_translation_m"] > QUIET_TRANSLATION_M_PER_PACKET
                      for p in packets)
    summary["rgbd_step_accuracy"] = {"registered_steps": len(errors),
                                     "median_absolute_error_m": round(float(np.median(errors)), 5),
                                     "p90_absolute_error_m": round(float(np.percentile(errors, 90)), 5),
                                     "false_quiet_steps_at_2mm": false_quiet}
    result = {"schema": "robot-learning-lab.ape-onboard-scene-motion.v1",
              "classification": "on-board RGB-D observation replay in a visual room; native physics not rerun",
              "source_capture_manifest_sha256": sha(SOURCE / "capture-manifest.json"),
              "source_body_result_sha256": sha(BODY_RESULT), "analysis_sha256": sha(Path(__file__)),
              "camera": {"name": "upper_rgb", "vertical_fov_deg": FOV_DEG,
                         "sample_period_s": .1, "depth_treatment": "MuJoCo Renderer optical-axis Z depth; local GL depth accuracy limited"},
              "feature_gate": {"minimum_matched_depth_features": MIN_MATCHED_DEPTH,
                               "minimum_pnp_inliers": MIN_INLIERS,
                               "minimum_inlier_fraction": MIN_INLIER_FRACTION,
                               "maximum_camera_translation_m_per_100ms": QUIET_TRANSLATION_M_PER_PACKET,
                               "consecutive_packets": WINDOW,
                               "texture_loss_at": "RGB pixels replaced by a uniform value from stop command + 0.4 s, permanent; PnP rerun on those actual blanked images"},
              "summary": summary, "rows": rows,
              "limits": ["World zero informed the exploratory 2 mm per 100 ms threshold; worlds one through three are different saved gait trajectories in the same visual room, not independent rooms or seeds.",
                         "Camera estimates include head bob and rotation, and a repeated tile wall can make PnP overconfident; this is not a calibrated visual-inertial SLAM or a physical depth sensor.",
                         "The visual room is contact-disabled and was absent from native physics. This does not test owner motion, collision avoidance, localization after occlusion or selected-body transfer."]}
    (SOURCE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    evaluate()
