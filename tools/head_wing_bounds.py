"""Construction bounds for a translated, clipped lower-wing seam strip."""

import math
import numpy as np
from lab.head_service import SPLIT_GAP_MM


def strip_bounds(robot, key, spec, seam_mm, raw_bounds):
    """Intersect containing CAD bounds with known clip planes and shell branches."""
    low = np.array(list(raw_bounds.min), float)
    high = np.array(list(raw_bounds.max), float)
    d = robot.config.dimensions
    g = next(link for link in robot.links if link.name == "head").geometry[0]
    lift = spec["seam_bridge_translation_mm"]
    overlap = spec["seam_overlap_mm"]
    top = d.neck_lower_port_top_m * 1000 - seam_mm
    lower_y = 29 + seam_mm
    upper_y = d.neck_lower_port_width_m * 500 - seam_mm
    side = 1 if key.endswith("left") else -1
    low[1] = max(low[1], lower_y if side == 1 else -upper_y)
    high[1] = min(high[1], upper_y if side == 1 else -lower_y)
    low[2] = max(low[2], top - overlap)
    high[2] = min(high[2], top + lift)
    split = robot.config.head_service.split_x_m * 1000
    if "_front_" in key:
        low[0] = max(low[0], split + SPLIT_GAP_MM / 2 + seam_mm)
    else:
        high[0] = min(high[0], split - SPLIT_GAP_MM / 2 - seam_mm)
    if g.shape == "ellipsoid":
        rx, ry, rz = np.array(g.size) * 1000
        ix, iy, iz = np.array(g.size) * 1000 - d.shell_wall_m * 1000
        source_z = (low[2] - lift, high[2] - lift)
        closest_z = (
            0 if source_z[0] <= 0 <= source_z[1] else min(abs(v) for v in source_z)
        )
        furthest_z = max(abs(v) for v in source_z)
        xmax = rx * math.sqrt(max(0, 1 - (lower_y / ry) ** 2 - (closest_z / rz) ** 2))
        inner_term = 1 - (upper_y / iy) ** 2 - (furthest_z / iz) ** 2
        xmin = ix * math.sqrt(max(0, inner_term))
        low[0] = max(low[0], -xmax)
        high[0] = min(high[0], xmax)
        if inner_term > 0:
            if low[0] > -xmin:
                low[0] = max(low[0], xmin)
            elif high[0] < xmin:
                high[0] = min(high[0], -xmin)
    if np.any(high <= low):
        raise ValueError("Empty construction bounds")
    # Numerical allowance only; actual source containment is checked by consumer.
    return low - 0.01, high + 0.01
