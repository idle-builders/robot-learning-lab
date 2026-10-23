"""Containing margin envelope for the ribbed stock recipe, not a mesh offset."""

from build123d import Box, Cylinder, Pos, scale

from lab.head_service import inner_envelope
from lab.mechanical import primitive
from lab.leg_mechanism import overlap


def envelope(robot, spec, row, actual, margin_mm=1.5):
    g = next(link for link in robot.links if link.name == "head").geometry[0]
    outer = primitive(g)
    if g.shape == "ellipsoid":
        expanded_outer = scale(outer, by=1 + margin_mm / (min(g.size) * 1000))
    else:
        expanded_outer = inner_envelope(g, -margin_mm)
    x, y, z = row["center_head_mm"]
    radius = spec["boss_radius_mm"]
    depth = spec["boss_depth_mm"]
    seat = spec["seat_thickness_mm"]
    boss = Pos(x, y, z + depth / 2) * Cylinder(radius, depth)
    bound = Pos(x, y, z + depth / 2) * Cylinder(
        radius + margin_mm, depth + 2 * margin_mm
    )
    tab = Pos(x, y, z - seat / 2) * Cylinder(radius, seat)
    tab_bound = Pos(x, y, z - seat / 2) * Cylinder(
        radius + margin_mm, seat + 2 * margin_mm
    )
    if spec.get("contour_tab", False):
        tab &= outer
        tab_bound &= expanded_outer
    if spec.get("rib_height_mm", 0):
        direction = 1 if "_front_" in row["cover"] else -1
        tip = direction * (robot.config.dimensions.head_depth_m * 500 + 1)
        center = Pos((x + tip) / 2, y, z + depth - spec["rib_height_mm"] / 2)
        rib = center * Box(abs(tip - x), spec["rib_width_mm"], spec["rib_height_mm"])
        rib_bound = center * Box(
            abs(tip - x) + 2 * margin_mm,
            spec["rib_width_mm"] + 2 * margin_mm,
            spec["rib_height_mm"] + 2 * margin_mm,
        )
        boss += rib & outer
        bound += rib_bound & expanded_outer
    expected = boss + tab
    if (
        abs(expected.volume - actual.volume) > 0.02
        or expected.volume - overlap(expected, actual) > 0.02
    ):
        raise ValueError("Reconstructed stock differs from source STEP")
    bound += tab_bound
    missing = actual.volume - overlap(actual, bound)
    if not bound.is_valid or not bound.solids() or missing > 0.02:
        raise ValueError("Primitive envelope invalid or omits source stock")
    return bound, dict(
        method="Union of expanded cylinder and rib/seat primitives; clipped pieces intersect a containing expanded head exterior",
        margin_mm=margin_mm,
        source_stock_identity="pass",
        missing_stock_mm3=max(0, missing),
        continuous_motion_status="not_run",
    )
