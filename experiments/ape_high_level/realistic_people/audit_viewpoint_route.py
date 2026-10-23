"""Evaluate a saved viewpoint route against scene truth withheld from its planner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def box_clearance(xy: list[float], center: list[float], size: list[float]) -> float:
    outside = [max(abs(xy[i] - center[i]) - size[i] / 2, 0.)
               for i in range(2)]
    return math.hypot(*outside)


def evaluate(evidence: Path) -> dict:
    scene = json.loads((evidence / "scene-info.json").read_text())
    actions = json.loads((evidence / "online-action-trace.json").read_text())["actions"]
    motion = json.loads((evidence / "character-motion-trace.json").read_text())
    ranges = json.loads((evidence / "range-fan-trace.json").read_text())["frames"]
    obstacle = scene.get("route_obstacle_evaluator_only")
    if not actions or not obstacle:
        raise ValueError("Expected a route run with an authored obstacle")
    proxy_radius_m = actions[0].get("route_proxy_radius_m") or .12
    planning_clearance_m = actions[0].get("route_planning_clearance_m") or proxy_radius_m
    start = scene["camera_eye_xyz_m"][:2]
    positions = [start] + [row["proxy_eye_xyz_m_evaluator_only"][:2]
                           for row in actions]
    first_cube_center = actions[0].get("route_cube_center_xy_m_evaluator_only")
    cube_moved = any(row.get("route_cube_center_xy_m_evaluator_only") !=
                     first_cube_center for row in actions[1:])
    static_hits = []
    for frame in ranges:
        heading = frame["body_heading_deg_commanded"]
        for plane in frame["planes"]:
            origin = plane["origin_xyz_m"]
            for beam in plane["beams"]:
                source = beam["hit_prim_evaluator_only"] or ""
                distance = beam["distance_m"]
                if distance is None or not (source.startswith("/Root/") or
                    (source == "/World/ApeRouteBlock" and not cube_moved)):
                    continue
                bearing = math.radians(heading + beam["angle_deg"])
                static_hits.append((
                    [origin[0] + distance * math.sin(bearing),
                     origin[1] + distance * math.cos(bearing)], source))
    clearances = []
    closest_static_hit = (math.inf, None)
    for index, (first, second) in enumerate(zip(positions, positions[1:])):
        cube_center = (actions[index].get("route_cube_center_xy_m_evaluator_only")
                       or obstacle["center_xy_m"])
        segment_m = math.dist(first, second)
        for sample in range(max(1, math.ceil(segment_m / .01)) + 1):
            fraction = sample / max(1, math.ceil(segment_m / .01))
            xy = [first[i] + fraction * (second[i] - first[i]) for i in range(2)]
            clearances.append(box_clearance(
                xy, cube_center, obstacle["size_xy_m"]))
            for hit_xy, source in static_hits:
                distance = math.dist(xy, hit_xy)
                if distance < closest_static_hit[0]:
                    closest_static_hit = (distance, source)
    final_owner = motion[-1]["characters"][0]["xyz_m"][:2]
    plans = [row["route_plan"] for row in actions]
    commands = [row["command_for_next_frame"] for row in actions]
    return {
        "scope": "independent evaluator using saved cube and character geometry; "
                 "the reported radius is only a kinematic eye-proxy clearance",
        "frames": len(actions),
        "route_range_mode": actions[0].get("route_range_mode", "both"),
        "route_cube_moved": cube_moved,
        "range_heights_above_floor_m": scene.get("range_heights_above_floor_m",
                                                 [0.25, 0.80]),
        "cube_hit_beams_by_plane": [sum(
            "ApeRouteBlock" in str(beam["hit_prim_evaluator_only"])
            for frame in ranges for beam in frame["planes"][plane_index]["beams"])
            for plane_index in (0, 1)],
        "initial_owner_distance_m": math.dist(start, motion[0]["characters"][0]["xyz_m"][:2]),
        "final_owner_distance_m": math.dist(positions[-1], final_owner),
        "minimum_cube_box_clearance_m": min(clearances),
        "minimum_sampled_static_hit_clearance_m": closest_static_hit[0],
        "nearest_sampled_static_hit_prim_evaluator_only": closest_static_hit[1],
        "sampled_static_hit_overlap_found": closest_static_hit[0] < proxy_radius_m,
        "sampled_static_hit_limit": "Sparse PhysX hit points can prove a proxy overlap, but cannot certify all unobserved static surfaces clear.",
        "temporary_proxy_clearance_m": proxy_radius_m,
        "temporary_proxy_clearance_pass": min(clearances) >= proxy_radius_m,
        "route_planning_clearance_m": planning_clearance_m,
        "route_planning_clearance_pass": min(clearances) >= planning_clearance_m,
        "plan_intents": {name: sum(plan["intent"] == name for plan in plans)
                         for name in sorted({plan["intent"] for plan in plans})},
        "commanded_translation_m": sum(command["forward_step_m"] for command in commands),
        "commanded_forward_m": sum(max(0., command["forward_step_m"])
                                   for command in commands),
        "commanded_reverse_m": sum(max(0., -command["forward_step_m"])
                                   for command in commands),
        "commanded_path_length_m": sum(abs(command["forward_step_m"])
                                       for command in commands),
        "commanded_camera_yaw_deg": sum(command["yaw_delta_deg"] for command in commands),
        "commanded_body_yaw_deg": sum(command.get("body_yaw_delta_deg",
                                      command["yaw_delta_deg"]) for command in commands),
        "max_abs_head_yaw_offset_deg": max(abs(row.get(
            "route_head_yaw_offset_deg_commanded") or 0.) for row in actions),
        "visually_registered_frames": sum(row["localization"] is not None and
                                           row["localization"]["status"] == "visual_registered"
                                           for row in actions),
        "motion_interlock_frames": sum(row["actor"].get("command_interlock") ==
                                       "pose_motion_unconfirmed" for row in actions),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    report = evaluate(args.evidence)
    (args.evidence.parent / "viewpoint-route-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
