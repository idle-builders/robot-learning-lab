"""Pure-Python inference for the frozen observable contact selector."""

from __future__ import annotations

import json
from pathlib import Path


NUMERIC_FEATURES = (
    "entry_error_x_mm", "entry_error_y_mm", "camera_x_mm", "camera_y_mm",
    "orange_pixels", "rail_v", "delay_ms", "gaze_head_yaw_rad",
    "pitch_offset_rad", "roll_clearance_rad",
)


def observed_features(report: dict) -> dict:
    entry = next((event for event in report["image_goal_capture_events"]
                  if event["time_s"] == report["reach_start_s"]), None)
    if not entry or not entry["accepted"] or not entry["used_for_control"]:
        raise ValueError("No controlled observed-RGB entry")
    selected = report["selection_features"]
    return {
        "entry_error_x_mm": round(selected["entry_error_xy_m"][0] * 1000, 6),
        "entry_error_y_mm": round(selected["entry_error_xy_m"][1] * 1000, 6),
        "camera_x_mm": round(selected["entry_camera_world_xyz_m"][0] * 1000, 6),
        "camera_y_mm": round(selected["entry_camera_world_xyz_m"][1] * 1000, 6),
        "orange_pixels": entry["orange_pixels"],
        "selection_mode": report["selection_mode"],
        "route_goal_source": selected["route_goal_source"],
        "rail_v": report["rail_v"],
        "delay_ms": report["delay_ms"],
        "gaze_head_yaw_rad": report["gaze_head_yaw_rad"],
        "pitch_offset_rad": report["pitch_offset_rad"],
        "roll_clearance_rad": report["roll_clearance_rad"],
    }


def predict(model: dict, features: dict) -> dict:
    if model["schema"] != "robot-learning-lab.cove50-v118-observable-selector.v1":
        raise ValueError("Unsupported contact selector")
    bounds = model["training_bounds"]
    reasons = []
    if features["selection_mode"] not in model["selection_modes"]:
        reasons.append("selection_mode_outside_training")
    if features["route_goal_source"] not in model["route_goal_sources"]:
        reasons.append("RGB_goal_source_outside_training")
    for key in NUMERIC_FEATURES:
        if not bounds[key][0] <= features[key] <= bounds[key][1]:
            reasons.append(f"{key}_outside_training")
    vector = ([float(features[key]) for key in NUMERIC_FEATURES] +
              [float(features["selection_mode"] == mode)
               for mode in model["selection_modes"]])
    outputs = []
    for tree in model["trees"]:
        node = 0
        while tree["feature"][node] >= 0:
            feature = tree["feature"][node]
            node = (tree["left"][node] if vector[feature] <= tree["threshold"][node]
                    else tree["right"][node])
        outputs.append(tree["value"][node])
    predicted_delta = sum(outputs) / len(outputs)
    shift_m = (-.002 if not reasons and
               predicted_delta > model["selection_threshold_s"] else 0.0)
    return {"shift_body_y_m": shift_m,
            "predicted_clipped_contact_delta_s": round(predicted_delta, 4),
            "abstention_reasons": reasons}


def load(path: Path) -> dict:
    return json.loads(path.read_text())
