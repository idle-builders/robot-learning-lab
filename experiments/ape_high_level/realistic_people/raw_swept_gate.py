"""Independent occupied-hit veto for one bounded eye-proxy command.

This is a local collision-monitor experiment, not a motion permission. It
requires only a current low scan and the proposed relative translation.
Unknown rays cannot prove a path free; the route's separate known-free gate
still owns positive authorization.
"""

from __future__ import annotations

import math
from statistics import median


def assess_swept_low_hits(ranges_m: list[float | None], angles_deg: list[float],
                          *, step_m: float, step_heading_rel_deg: float,
                          clearance_m: float = .24) -> dict:
    if len(ranges_m) != len(angles_deg):
        raise ValueError("ranges and angles must align")
    heading = math.radians(step_heading_rel_deg)
    end_x = step_m * math.sin(heading)
    end_y = step_m * math.cos(heading)
    step_squared = end_x * end_x + end_y * end_y
    swept = []
    measured_beams = 0
    for distance, angle_deg in zip(ranges_m, angles_deg):
        if distance is not None and distance > 0 and not math.isnan(distance):
            measured_beams += 1
        if distance is None or not math.isfinite(distance) or distance <= 0:
            swept.append(None)
            continue
        angle = math.radians(angle_deg)
        point_x = distance * math.sin(angle)
        point_y = distance * math.cos(angle)
        fraction = (max(0., min(1., (point_x * end_x + point_y * end_y) /
                                  step_squared)) if step_squared > 1e-8 else 0.)
        swept.append(math.hypot(point_x - fraction * end_x,
                                point_y - fraction * end_y))
    # A spatial patch makes the veto less sensitive to one noisy return.
    patches = [median(present) for start in range(len(swept) - 4)
               if len(present := [value for value in swept[start:start + 5]
                                  if value is not None]) >= 3]
    nearest = min(patches) if patches else None
    return {"veto": nearest is not None and nearest < clearance_m,
            "status": ("observed_hit_patch" if nearest is not None else
                       "measured_without_coherent_hit" if measured_beams else
                       "unknown"),
            "measured_beams": measured_beams,
            "nearest_coherent_swept_hit_m": (round(nearest, 5)
                                              if nearest is not None else None),
            "clearance_m": clearance_m, "step_m": step_m}
