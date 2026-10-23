"""Score passive head RTX returns against the authored held low-cube challenge.

Cube coordinates are evaluator-only. This script does not feed the actor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from audit_viewpoint_route import box_clearance


ROOT = Path("artifacts/ape-realistic-people")
RUN = ROOT / "isaac61-midroom-held-probe-passive-os1-v216"


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    evidence = RUN / "evidence"
    lidar_file = evidence / "rtx-lidar-trace.json"
    action_file = evidence / "online-action-trace.json"
    range_file = evidence / "range-fan-trace.json"
    scene_file = evidence / "scene-info.json"
    monitor_file = evidence / "continuous-vision-monitor.json"
    lidar = load(lidar_file)
    actions = load(action_file)["actions"]
    ranges = load(range_file)["frames"]
    scene = load(scene_file)
    monitor = load(monitor_file)["frames"]
    assert len(lidar["frames"]) == len(actions) == len(ranges) == len(monitor) == 36
    size = scene["route_obstacle_evaluator_only"]["size_xy_m"]
    rows = []
    for frame, (laser, action, fan, watched) in enumerate(zip(
            lidar["frames"], actions, ranges, monitor)):
        assert laser["frame"] == action["frame"] == frame
        eye = action["proxy_eye_xyz_m_evaluator_only"][:2]
        cube = action["route_cube_center_xy_m_evaluator_only"]
        gap = box_clearance(eye, cube, size)
        rows.append({"frame": frame, "sim_time_s": fan["timeline_time_s"],
            "cube_gap_m_evaluator_only": gap,
            "rtx_low_cube_returns_evaluator_only": laser.get("evaluator_cube_low_returns", 0),
            "rtx_low_cube_nearest_horizontal_m_evaluator_only": laser.get(
                "evaluator_cube_nearest_horizontal_m"),
            "rtx_all_low_band_returns": laser.get("low_band_returns", 0),
            "rtx_valid_ranges": laser.get("valid_ranges", 0),
            "rtx_minimum_valid_range_m": laser.get("range_min_m"),
            "range_fan_low_finite_beams": sum(beam["distance_m"] is not None
                for beam in fan["planes"][0]["beams"]),
            "held_alarm": bool(watched.get("held_raw_alarm"))})
    hits = [row for row in rows if row["rtx_low_cube_returns_evaluator_only"]]
    near = [row for row in rows if row["cube_gap_m_evaluator_only"] <= .3]
    report = {"scope": "passive OS1 head RTX; actor uses ideal PhysX fan and fixed-eye probe",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (lidar_file, action_file, range_file, scene_file, monitor_file)},
        "rtx_cube_classification": "evaluator-only world-space points inside 32 cm cube XY box and 3–32 cm above floor; not an actor semantic label",
        "first_cube_low_return": hits[0] if hits else None,
        "last_cube_low_return": hits[-1] if hits else None,
        "first_held_alarm": next((row for row in rows if row["held_alarm"]), None),
        "frames_with_cube_low_return": len(hits),
        "frames_with_true_gap_at_most_30cm": len(near),
        "near_frames_with_cube_low_return": sum(
            row["rtx_low_cube_returns_evaluator_only"] > 0 for row in near),
        "minimum_valid_rtx_range_m": min(row["rtx_minimum_valid_range_m"]
            for row in rows if row["rtx_minimum_valid_range_m"] is not None),
        "cube_nearest_return_span_m": [min(row[
            "rtx_low_cube_nearest_horizontal_m_evaluator_only"] for row in hits),
            max(row["rtx_low_cube_nearest_horizontal_m_evaluator_only"]
                for row in hits)] if hits else None,
        "minimum_cube_gap_m_evaluator_only": min(row["cube_gap_m_evaluator_only"] for row in rows),
        "rows": rows,
        "limits": ["OS1 head mount at 45 cm is a provisional geometry, not a body integration",
            "Isaac warned that multi-tick lidar lacked motion BVH; dynamic-point motion distortion is not validated",
            "Empirical minimum returned range around 40.7 cm in this configuration; cube visibility cannot resolve its 15–30 cm box clearance",
            "A return detects occupied surface but unknown or absent returns do not certify free space",
            "Fixed kinematic eye omits body occlusion, inertia, IMU, joints and contact"]}
    out = RUN / "passive-rtx-low-intrusion-audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "first_cube_low_return", "last_cube_low_return", "first_held_alarm",
        "frames_with_cube_low_return", "near_frames_with_cube_low_return",
        "frames_with_true_gap_at_most_30cm", "minimum_valid_rtx_range_m",
        "cube_nearest_return_span_m", "minimum_cube_gap_m_evaluator_only")},
        indent=2))


if __name__ == "__main__":
    main()
