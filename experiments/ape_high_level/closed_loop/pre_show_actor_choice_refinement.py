"""Keep point-level obstacle veto separate from incomplete ray coverage."""

from __future__ import annotations


MARGIN_M = .05


def classify_route(voxel_pose_samples: dict, point_pose_samples: dict) -> dict:
    """Classify a route from actor evidence without treating a voxel as a solid.

    A point gap is only a veto: sparse points do not prove unmeasured space
    clear. A voxel occupied mark with no corroborating close endpoint is an
    ambiguous map conflict, not a guaranteed collision.
    """
    samples = {}
    for label, voxel in voxel_pose_samples.items():
        point = point_pose_samples[label]
        gap = point["minimum_gap_m"]
        if gap is not None and gap < MARGIN_M:
            status = "measured_point_veto"
        elif voxel["status"] == "occupied":
            status = "voxel_point_disagreement_unknown"
        elif voxel["status"] == "unknown":
            status = "unknown_swept_space"
        else:
            status = "map_covered_only"
        samples[label] = {"status": status,
                          "voxel_status": voxel["status"],
                          "right_upper_point_gap_m": gap}
    ordered = ("measured_point_veto", "voxel_point_disagreement_unknown",
               "unknown_swept_space", "map_covered_only")
    outcome = next(name for name in ordered
                   if name in {row["status"] for row in samples.values()})
    return {"status": outcome, "pose_samples": samples,
            "motion_authorized": False,
            "missing_gates": ["calibrated_pose_error_bound",
                              "dynamic_future_occupancy",
                              "selected_body_motion_skill"]}
