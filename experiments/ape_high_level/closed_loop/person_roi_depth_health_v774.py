"""Allow a fresh detected-person depth ROI when unrelated background is invalid."""

from __future__ import annotations

import time

import numpy as np

import owner_depth_change_component_v516 as health
import probe_layout2_person_roi_depth_v773 as probe
import robocasa_owner_reacquire_rgbd_v560 as detector
from screen_robocasa_owner_safe_stop_v494 import depth_point


ORIGINAL_OBSERVE = detector.observe
MIN_PERSON_ROI_VALID = .60


def observe(rgb, depth, data, model, camera, estimate_xy, actual_xy):
    if depth is None:
        return ORIGINAL_OBSERVE(rgb, depth, data, model, camera,
                                estimate_xy, actual_xy)
    global_healthy, fractions = health.packet_health(depth, depth)
    if global_healthy:
        result = ORIGINAL_OBSERVE(rgb, depth, data, model, camera,
                                  estimate_xy, actual_xy)
        result["depth_health_basis"] = "global_central_patch"
        return result
    begun = time.perf_counter()
    found = probe.candidates(rgb)
    wall = time.perf_counter() - begun
    if not found or found[0]["score"] < detector.PERSON_SCORE:
        return {"status": "person_unconfirmed",
                "depth_health_basis": "person_roi_after_global_failure",
                "central_valid_fraction": fractions[0],
                "detections": found, "detector_wall_s": wall}
    top = found[0]
    local_fraction = probe.roi_fraction(depth, top["box_xyxy"])
    if local_fraction < MIN_PERSON_ROI_VALID:
        return {"status": "person_depth_unhealthy",
                "depth_health_basis": "person_roi_after_global_failure",
                "central_valid_fraction": fractions[0],
                "person_roi_valid_fraction": local_fraction,
                "detections": found, "detector_wall_s": wall}
    point = depth_point(top["box_xyxy"], depth, data, model, camera)
    if point is None:
        return {"status": "person_depth_unlocalized",
                "depth_health_basis": "person_roi_after_global_failure",
                "central_valid_fraction": fractions[0],
                "person_roi_valid_fraction": local_fraction,
                "detections": found, "detector_wall_s": wall}
    point = np.asarray(point)
    point[:2] += np.asarray(estimate_xy) - np.asarray(actual_xy)
    return {"status": "person_surface_localized",
            "depth_health_basis": "person_roi_after_global_failure",
            "central_valid_fraction": fractions[0],
            "person_roi_valid_fraction": local_fraction,
            "detections": found,
            "point_xyz_est_m": point.tolist(), "detector_wall_s": wall}
