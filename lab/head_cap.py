"""Recover head-local shell wings for a removable-cap feasibility study.

These are unmounted study solids, not released robot parts. Shared head shape,
wall, neck port and rear-service split remain authoritative.
"""

from build123d import Box, Pos, scale

from .mechanical import hollow_shell, primitive
from .head_service import inner_envelope, SPLIT_GAP_MM


def shell_wings(robot, seam_mm=0.3, depth_mm=None, margin_mm=0.0):
    """Four head-owned wings; optional conservative moving-clearance envelope.

    Expanding each clipping half-space and a containing shell envelope contains
    the wing's Euclidean dilation. Ellipsoid scaling uses its smallest radius;
    it is deliberately looser than a true offset. Camera cutouts are filled in
    the clearance envelope, making it conservative there as well.
    """
    import math

    if not math.isfinite(seam_mm) or seam_mm <= 0:
        raise ValueError("seam must be positive and finite")
    if not math.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("margin must be nonnegative and finite")
    if depth_mm is not None and (not math.isfinite(depth_mm) or depth_mm <= 0):
        raise ValueError("depth must be positive and finite")
    d, settings = robot.config.dimensions, robot.config.head_service
    if not settings.enabled or not d.neck_mount_thickness_m:
        raise ValueError("cap requires mounted neck and split head")
    g = next(link for link in robot.links if link.name == "head").geometry[0]
    wall = d.shell_wall_m * 1000
    top = d.neck_lower_port_top_m * 1000 - seam_mm
    bottom = -d.head_height_m * 500 - 1 if depth_mm is None else top - depth_mm
    inner_y = 29 + seam_mm
    outer_y = d.neck_lower_port_width_m * 500 - seam_mm
    if outer_y <= inner_y or seam_mm >= 2 or margin_mm >= 5:
        raise ValueError("cap seam/band or margin outside study range")
    shell = hollow_shell(g, d.shell_wall_m)
    if margin_mm:
        if g.shape == "ellipsoid":
            outer_radius = min(g.size) * 1000
            inner_radius = outer_radius - wall
            shell = scale(primitive(g), by=1 + margin_mm / outer_radius) - scale(
                inner_envelope(g, wall), by=1 - margin_mm / inner_radius
            )
        else:
            shell = inner_envelope(g, -margin_mm) - inner_envelope(g, wall + margin_mm)
    elif robot.config.camera.cassette_enabled:
        from .camera_mount import clearance

        shell -= clearance(robot)
    result = {}
    split = settings.split_x_m * 1000
    for side, sign in (("left", 1), ("right", -1)):
        band = Pos(0, sign * (inner_y + outer_y) / 2, (bottom + top) / 2) * Box(
            300, outer_y - inner_y + 2 * margin_mm, top - bottom + 2 * margin_mm
        )
        for end, direction in (("front", 1), ("rear", -1)):
            edge = split + direction * (SPLIT_GAP_MM / 2 + seam_mm - margin_mm)
            clip = Pos(edge + direction * 150, 0, 0) * Box(300, 300, 300)
            result[f"head_lower_{end}_{side}"] = shell & band & clip
    return result
