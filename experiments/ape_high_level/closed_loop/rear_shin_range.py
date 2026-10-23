"""Candidate rear-outward low range view attached to each Cove-50 shin.

It uses the V190 first-room/self ray and noise model. Optical package,
field-of-view hardware and joint cable path are not yet qualified.
"""

from __future__ import annotations

import hashlib
import math

import mujoco
import numpy as np

from audit_shin_range_v190 import (BEAMS, FOV_RAD, MAX_RANGE_M,
                                   PITCH_RAD, ROOM_GROUP, SELF_GROUP)


def scan(room, side: int, case: str) -> tuple[np.ndarray, dict]:
    """World returns from one rear-outward 8x8 first-hit shin fan."""
    if side not in (-1, 1):
        raise ValueError(side)
    seed = int.from_bytes(hashlib.sha256(case.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    body_name = "left_shin" if side == 1 else "right_shin"
    bid = room.model.body(body_name).id
    rotation = room.data.xmat[bid].reshape(3, 3)
    origin = (room.data.xpos[bid] +
              rotation @ np.array((.02, side * .04, -.05)))
    counts = {"rays": BEAMS * BEAMS, "self_first": 0, "floor_first": 0,
              "dropout": 0, "environment_points": 0,
              "origin_world_m": np.round(origin, 5).tolist()}
    points = []
    for row in range(BEAMS):
        elevation = PITCH_RAD + (row + .5 - BEAMS / 2) * FOV_RAD / BEAMS
        for column in range(BEAMS):
            azimuth = (side * math.radians(112.5) +
                       (column + .5 - BEAMS / 2) * FOV_RAD / BEAMS)
            local_ray = np.array((math.cos(elevation) * math.cos(azimuth),
                                  math.cos(elevation) * math.sin(azimuth),
                                  math.sin(elevation)))
            direction = rotation @ local_ray
            room_hit = np.array([-1], dtype=np.int32)
            self_hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(room.model, room.data, origin, direction,
                                     ROOM_GROUP, True, -1, room_hit)
            self_distance = mujoco.mj_ray(room.model, room.data, origin, direction,
                                          SELF_GROUP, True, -1, self_hit)
            if self_distance >= 0 and (distance < 0 or self_distance <= distance):
                counts["self_first"] += 1
                continue
            if distance < 0 or distance > MAX_RANGE_M:
                continue
            point = origin + distance * direction
            if point[2] < .015:
                counts["floor_first"] += 1
                continue
            if rng.random() < .025:
                counts["dropout"] += 1
                continue
            noisy = origin + max(.02, distance + rng.normal(0., .004)) * direction
            points.append(noisy)
            counts["environment_points"] += 1
    return np.asarray(points, dtype=float).reshape(-1, 3), counts
