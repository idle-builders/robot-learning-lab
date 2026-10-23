"""Provisional low-mounted multizone ToF ray geometry for Isaac screening.

Each zone is represented by one central PhysX ray. This captures mounting,
discrete coverage, range and cadence, but not the device's optical response,
histogram processing, reflectance or crosstalk. It cannot authorize motion.
"""

from __future__ import annotations

import math


FOV_DEG = 45.0
ZONES_PER_AXIS = 8
MIN_RANGE_M = .02
MAX_RANGE_M = 4.0
SAMPLE_PERIOD_S = 1.0 / 15.0

# Body-relative forward/lateral offsets; all four centers are 12 cm high.
# They are screening hypotheses, not package- or self-occlusion-checked mounts.
MOUNTS = {
    "front_left": {"forward_m": .08, "lateral_m": -.12,
                   "height_m": .12, "yaw_deg": -45., "pitch_deg": -10.},
    "front_center": {"forward_m": .14, "lateral_m": 0.,
                     "height_m": .12, "yaw_deg": 0., "pitch_deg": -10.},
    "front_right": {"forward_m": .08, "lateral_m": .12,
                    "height_m": .12, "yaw_deg": 45., "pitch_deg": -10.},
    "rear_center": {"forward_m": -.12, "lateral_m": 0.,
                    "height_m": .12, "yaw_deg": 180., "pitch_deg": -10.},
}


def zone_rays(eye_xy: tuple[float, float], heading_deg: float,
              floor_z: float):
    """Yield each mount's world origin and 64 central-ray unit directions."""
    body = math.radians(heading_deg)
    for name, mount in MOUNTS.items():
        forward = mount["forward_m"]
        lateral = mount["lateral_m"]
        origin = (eye_xy[0] + forward * math.sin(body) + lateral * math.cos(body),
                  eye_xy[1] + forward * math.cos(body) - lateral * math.sin(body),
                  floor_z + mount["height_m"])
        rays = []
        for row in range(ZONES_PER_AXIS):
            vertical = math.radians(mount["pitch_deg"] +
                (row - (ZONES_PER_AXIS - 1) / 2) * FOV_DEG / ZONES_PER_AXIS)
            for column in range(ZONES_PER_AXIS):
                bearing = math.radians(heading_deg + mount["yaw_deg"] +
                    (column - (ZONES_PER_AXIS - 1) / 2) * FOV_DEG / ZONES_PER_AXIS)
                rays.append((math.cos(vertical) * math.sin(bearing),
                             math.cos(vertical) * math.cos(bearing),
                             math.sin(vertical)))
        yield name, origin, rays


def sample(range_query, carb, eye_xy: tuple[float, float],
           heading_deg: float, floor_z: float, time_s: float,
           cube_center_xy: tuple[float, float] | None):
    """Save actor-range values separately from evaluator-only hit identity."""
    sensors = {}
    for name, origin, rays in zone_rays(eye_xy, heading_deg, floor_z):
        distances = []
        heights = []
        cube_hits = []
        source = carb.Float3(*origin)
        for direction in rays:
            hit = range_query.raycast_closest(source, carb.Float3(*direction), MAX_RANGE_M)
            distance = float(hit["distance"]) if hit["hit"] else None
            if distance is not None and not MIN_RANGE_M <= distance <= MAX_RANGE_M:
                distance = None
            distances.append(distance)
            heights.append(origin[2] + distance * direction[2] - floor_z
                           if distance is not None else None)
            prim = str(hit.get("rigidBody") or hit.get("collision") or "") if hit["hit"] else ""
            cube_hits.append("/World/ApeRouteBlock" in prim)
        sensors[name] = {"origin_xyz_m": origin,
            "range_m_by_zone": distances,
            "hit_height_above_floor_m_by_zone": heights,
            "cube_hit_by_zone_evaluator_only": cube_hits}
    return {"sim_time_s": time_s,
            "eye_xy_m_evaluator_only": eye_xy,
            "body_heading_deg_commanded": heading_deg,
            "cube_center_xy_m_evaluator_only": cube_center_xy,
            "sensors": sensors}
