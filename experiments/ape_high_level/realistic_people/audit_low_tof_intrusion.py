"""Audit passive low multizone rays during an incoming-cube Isaac run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_viewpoint_route import box_clearance
from near_body_tof import zone_rays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    evidence = args.run / "evidence"
    scene_file = evidence / "scene-info.json"
    tof_file = evidence / "low-tof-trace.json"
    scene = json.loads(scene_file.read_text())
    tof = json.loads(tof_file.read_text())
    size = scene["route_obstacle_evaluator_only"]["size_xy_m"]
    floor_z = scene["reference_floor_z_m"]
    rows = []
    for sample in tof["samples"]:
        eye = sample["eye_xy_m_evaluator_only"]
        cube = sample["cube_center_xy_m_evaluator_only"]
        row = {"sim_time_s": sample["sim_time_s"],
               "cube_gap_m_evaluator_only": box_clearance(eye, cube, size),
               "cube_center_xy_m_evaluator_only": cube,
               "sensors": {}}
        for name, origin, directions in zone_rays(eye,
                sample["body_heading_deg_commanded"], floor_z):
            sensor = sample["sensors"][name]
            assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(
                origin, sensor["origin_xyz_m"]))
            assert len(sensor["range_m_by_zone"]) == 64
            points = []
            cube_points = []
            for distance, height, cube_hit, direction in zip(
                    sensor["range_m_by_zone"],
                    sensor["hit_height_above_floor_m_by_zone"],
                    sensor["cube_hit_by_zone_evaluator_only"], directions):
                if distance is None or height is None or height < .03:
                    continue
                xy = (origin[0] + distance * direction[0],
                      origin[1] + distance * direction[1])
                radial = math.hypot(xy[0] - eye[0], xy[1] - eye[1])
                points.append(radial)
                if cube_hit:
                    cube_points.append(radial)
            row["sensors"][name] = {
                "mount_cube_box_gap_m_evaluator_only": box_clearance(
                    origin[:2], cube, size),
                "nonfloor_hit_zones": len(points),
                "cube_hit_zones_evaluator_only": len(cube_points),
                "nearest_nonfloor_eye_radial_m": min(points) if points else None,
                "nearest_cube_eye_radial_m_evaluator_only": (
                    min(cube_points) if cube_points else None)}
        front = ("front_left", "front_center", "front_right")
        front_values = [row["sensors"][name]["nearest_nonfloor_eye_radial_m"]
                        for name in front]
        finite = [value for value in front_values if value is not None]
        row["nearest_front_nonfloor_eye_radial_m"] = min(finite) if finite else None
        rows.append(row)
    assert rows and all(b["sim_time_s"] > a["sim_time_s"]
                        for a, b in zip(rows, rows[1:]))
    gaps = [b["sim_time_s"] - a["sim_time_s"] for a, b in zip(rows, rows[1:])]
    sensors = list(rows[0]["sensors"])
    report = {"scope": "evaluator-only clearance rescore of single-center-ray-per-zone PhysX samples; actor ToF use, if any, is recorded in its own action trace",
        "run": args.run.name,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (scene_file, tof_file)},
        "sample_count": len(rows),
        "maximum_sim_sample_gap_s": max(gaps),
        "minimum_cube_box_gap_m_evaluator_only": min(
            row["cube_gap_m_evaluator_only"] for row in rows),
        "per_mount": {name: {"samples_with_cube_return": sum(
            row["sensors"][name]["cube_hit_zones_evaluator_only"] > 0
            for row in rows),
            "minimum_mount_cube_box_gap_m_evaluator_only": min(
                row["sensors"][name]["mount_cube_box_gap_m_evaluator_only"]
                for row in rows),
            "first_cube_return": next(({
                "sim_time_s": row["sim_time_s"],
                "cube_gap_m_evaluator_only": row["cube_gap_m_evaluator_only"],
                "cube_hit_zones_evaluator_only": row["sensors"][name][
                    "cube_hit_zones_evaluator_only"],
                "nearest_cube_eye_radial_m_evaluator_only": row["sensors"][name][
                    "nearest_cube_eye_radial_m_evaluator_only"]}
                for row in rows if row["sensors"][name]["cube_hit_zones_evaluator_only"]),
                None)} for name in sensors},
        "front_nonfloor_thresholds": {str(threshold): {
            "first": next(({"sim_time_s": row["sim_time_s"],
                "cube_gap_m_evaluator_only": row["cube_gap_m_evaluator_only"],
                "nearest_front_nonfloor_eye_radial_m": row[
                    "nearest_front_nonfloor_eye_radial_m"]}
                for row in rows if row["nearest_front_nonfloor_eye_radial_m"]
                    is not None and row["nearest_front_nonfloor_eye_radial_m"]
                    < threshold), None),
            "sample_count": sum(row["nearest_front_nonfloor_eye_radial_m"]
                is not None and row["nearest_front_nonfloor_eye_radial_m"]
                < threshold for row in rows)} for threshold in (.24, .31)},
        "limits": ["One central collision ray is an optimistic substitute for each optical zone",
                   "No reflectance, ambient-light, crosstalk, dropout, noise or body self-occlusion",
                   "A finite hit proves occupied surface, not full-body free swept volume",
                   "Evaluator cube labels are not visible to the actor"],
        "rows": rows}
    output = args.run / "low-tof-intrusion-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "sample_count", "maximum_sim_sample_gap_s",
        "minimum_cube_box_gap_m_evaluator_only", "per_mount",
        "front_nonfloor_thresholds")}, indent=2))


if __name__ == "__main__":
    main()
