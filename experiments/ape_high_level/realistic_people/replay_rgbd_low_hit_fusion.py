"""Replay an unchanged routed episode with and without RGB-D low hit evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fan_view_route import FanViewRoute
from rgbd_low_surface import low_surface_fan


def replay(run: Path) -> dict:
    evidence = run / "evidence"
    actions_path = evidence / "online-action-trace.json"
    ranges_path = evidence / "range-fan-trace.json"
    depth_path = evidence / "ape_social_90deg-depth-sequence-stride3.npz"
    actions = json.loads(actions_path.read_text())["actions"]
    ranges = json.loads(ranges_path.read_text())["frames"]
    with np.load(depth_path) as sequence:
        depths = sequence["depth_m"]
    if len(actions) != len(ranges) or len(actions) != len(depths):
        raise ValueError("Saved sensor and action sequences disagree")
    variants = {}
    for fusion in (False, True):
        planner = FanViewRoute(proxy_radius_m=.20, planning_margin_m=.04)
        rows = []
        for frame, (action, scan, depth) in enumerate(zip(actions, ranges, depths)):
            plan = action["route_plan"]
            if not plan or "target_xy_m" not in plan:
                continue
            angles = np.asarray([beam["angle_deg"] for beam in
                                 scan["planes"][0]["beams"]], dtype=np.float32)
            distances = np.asarray([[beam["distance_m"] if
                beam["distance_m"] is not None else np.inf for beam in plane["beams"]]
                for plane in scan["planes"]], dtype=np.float32)
            hit_only = None
            if action.get("low_range_blind_sector_deg"):
                distances[0] = np.nan
                if fusion:
                    hit_only = low_surface_fan(depth, angles,
                        horizontal_fov_deg=90.0,
                        camera_yaw_deg=scan["camera_heading_deg_commanded"],
                        body_yaw_deg=scan["body_heading_deg_commanded"])[
                            "hit_ranges_m"]
            replayed = planner.plan(distances, angles,
                np.asarray(action["localization"]["pose_xy_m"]),
                scan["body_heading_deg_commanded"],
                np.asarray(plan["target_xy_m"]), action["observation_time_s"],
                hit_only_ranges_m=hit_only)
            rows.append({"frame": frame, "saved_intent": plan["intent"],
                "replayed_intent": replayed["intent"],
                "reason": replayed.get("reason"),
                "rgbd_low_hit_beams": replayed["rgbd_low_hit_beams"]})
        variants["rgbd_hits" if fusion else "range_only"] = rows
    baseline = variants["range_only"]
    if not all(row["saved_intent"] == row["replayed_intent"] for row in baseline):
        raise AssertionError("Range-only replay does not reproduce saved route")
    return {"scope": "same saved RGB-D, range, pose and target packets; route "
            "counterfactual only, with no live trajectory after the first changed command",
            "source_run": str(run),
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (actions_path, ranges_path, depth_path)},
            "baseline_matches_saved": True,
            "changed_frames": [a["frame"] for a, b in zip(*variants.values())
                if a["replayed_intent"] != b["replayed_intent"]],
            "variants": variants}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = replay(args.run)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"baseline_matches_saved": result["baseline_matches_saved"],
        "changed_frames": result["changed_frames"],
        "changed_rows": [row for row in result["variants"]["rgbd_hits"]
            if row["frame"] in result["changed_frames"]]}, indent=2))
