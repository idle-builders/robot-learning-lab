"""Bounded observed-RGB reach target correction for the V135 candidate."""

from __future__ import annotations

import json
from pathlib import Path


SCHEMA = "robot-learning-lab.cove50-v135-planar-selector.v1"


def predict(model: dict, features: dict) -> dict:
    if model["schema"] != SCHEMA:
        raise ValueError("Unsupported V135 selector")
    reasons = []
    if features["route_goal_source"] != "selected_image_goal":
        reasons.append("route_goal_source_outside_rule")
    rule = model["rules"].get(features["selection_mode"])
    if rule is None or rule["entry_error_x_mm_gte"] is None:
        reasons.append("selection_mode_outside_rule")
    for key, (low, high) in model["training_bounds"].items():
        if not low <= features[key] <= high:
            reasons.append(f"{key}_outside_training")
    if not reasons and features["entry_error_x_mm"] < rule["entry_error_x_mm_gte"]:
        reasons.append("entry_error_x_below_rule")
    if (not reasons and rule["entry_error_y_mm_gte"] is not None and
            features["entry_error_y_mm"] < rule["entry_error_y_mm_gte"]):
        reasons.append("entry_error_y_below_rule")
    return {"shift_body_y_m": -.002 if not reasons else 0.0,
            "abstention_reasons": reasons}


def load(path: Path) -> dict:
    return json.loads(path.read_text())
