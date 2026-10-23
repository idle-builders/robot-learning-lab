"""Audit low obstacle visibility in saved head RGB-D against withheld PhysX hits.

The depth projection is actor-available.  PhysX hit prims and ranges are used
only here to score whether a depth surface agrees with the low obstacle; they
must never be supplied to the planner as object labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rgbd_low_surface import low_surface_fan


def audit(run: Path) -> dict:
    evidence = run / "evidence"
    actions = json.loads((evidence / "online-action-trace.json").read_text())["actions"]
    ranges = json.loads((evidence / "range-fan-trace.json").read_text())["frames"]
    with np.load(evidence / "ape_social_90deg-depth-sequence-stride3.npz") as sequence:
        depths = sequence["depth_m"]
    if not (len(actions) == len(ranges) == len(depths)):
        raise ValueError("RGB-D, actions and range frames are not synchronized")
    camera_yaw_deg = 0.0
    rows = []
    for index, (action, scan, depth) in enumerate(zip(actions, ranges, depths)):
        beams = scan["planes"][0]["beams"]
        angles = np.asarray([beam["angle_deg"] for beam in beams], dtype=np.float32)
        projected = low_surface_fan(depth, angles, horizontal_fov_deg=90.0,
            camera_yaw_deg=camera_yaw_deg,
            body_yaw_deg=scan["body_heading_deg_commanded"])
        hits = projected["hit_ranges_m"]
        cube = np.asarray(["ApeRouteBlock" in str(beam["hit_prim_evaluator_only"])
                           for beam in beams])
        actual = np.asarray([beam["distance_m"] if beam["distance_m"] is not None
                             else np.nan for beam in beams], dtype=np.float32)
        matched = cube & np.isfinite(hits) & (np.abs(hits - actual) <= .10)
        wrong_depth = cube & np.isfinite(hits) & ~matched
        rows.append({"frame": index,
            "low_range_masked_for_actor": bool(action.get("low_range_blind_sector_deg")),
            "depth_low_surface_hit_bins": projected["hit_bins"],
            "depth_low_surface_candidate_pixels": projected["candidate_pixels"],
            "low_range_cube_hit_bins_evaluator_only": int(cube.sum()),
            "depth_cube_corresponding_bins_evaluator_only": int(matched.sum()),
            "depth_other_surface_at_cube_bearing_bins_evaluator_only": int(wrong_depth.sum()),
            "depth_min_horizontal_low_hit_m": projected["min_horizontal_hit_m"],
            "matched_cube_range_median_abs_error_m_evaluator_only": (
                float(np.median(np.abs(hits[matched] - actual[matched])))
                if matched.any() else None)})
        camera_yaw_deg = action["proxy_heading_deg_evaluator_only"]
    return {"scope": "saved-packet low-surface visibility audit; calibrated fixed eye camera, "
            "not a free-space map or live depth-fusion route; cube truth only scores after capture",
            "run": str(run), "frames": len(rows),
            "masked_frames_with_depth_cube_correspondence": sum(
                row["low_range_masked_for_actor"] and
                row["depth_cube_corresponding_bins_evaluator_only"] > 0 for row in rows),
            "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = audit(args.run)
    (args.run / "rgbd-low-surface-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"run": report["run"], "frames": report["frames"],
        "masked_frames_with_depth_cube_correspondence": report[
            "masked_frames_with_depth_cube_correspondence"],
        "last_five": report["rows"][-5:]}, indent=2))
