"""Candidate neck supports; millimetres, nominal fit only."""

from build123d import Box, Cylinder, Pos, Rot, Compound, RegularPolygon, extrude


def bridge_offsets():
    return (-16.5, 16.5)


def head_bridge_positions():
    return ((-5, 0), (5, 0))


def hardware_counts():
    from .pelvis import clamp_stack_specs

    clamp_screws = 2 * sum(2 for _ in clamp_stack_specs(False))
    bridge_screws = len(bridge_offsets()) + len(head_bridge_positions())
    return {
        "screw_m3_40": clamp_screws,
        "screw_m3_10": len(bridge_offsets()),
        "screw_m3_12": len(head_bridge_positions()),
        "washer_m3": 2 * (clamp_screws + bridge_screws),
        "nut_m3": clamp_screws + bridge_screws,
    }


def yaw_support(d):
    """Rear-shell pedestal and clamp plate, in torso coordinates."""
    axis = 1000 * (d.torso_height_m + d.neck_yaw_rise_m)
    bottom = 1000 * d.torso_height_m - 25
    top = axis - 10
    stock = Pos(-13, -16.865, axis) * Box(63.23, 3, 26)
    stock += Pos(-44.615, -15.365, (bottom + top) / 2) * Box(4, 6, top - bottom)
    for x in [-38.615, 13.615]:
        for z in [-8, 8]:
            stock -= Pos(x, -16.865, axis + z) * Rot(90, 0, 0) * Cylinder(1.65, 10)
    return stock


def fit_rear_shell(rear, d):
    result = rear + Pos(0, 0, -d.torso_height_m * 500) * yaw_support(d)
    if not result.is_valid or len(result.solids()) != 1:
        raise ValueError("neck pedestal is not connected to rear shell")
    return result


def pitch_support(d):
    """Yaw-yoke bridge to underside of pitch clamp, in yaw-link coordinates."""
    top = d.neck_pitch_rise_m * 1000 - 15.365
    shape = Pos(32, 0, (top - 20) / 2) * Box(3, 20, top + 20)
    shape += Pos((-44.615 + 33.5) / 2, 0, top - 1.5) * Box(78.115, 28, 3)
    for z in bridge_offsets():
        shape -= Pos(32, 0, z) * Rot(0, 90, 0) * Cylinder(1.65, 6)
    for x in [-38.615, 13.615]:
        for y in [-8, 8]:
            shape -= Pos(x, y, top - 1.5) * Cylinder(1.65, 6)
    return shape


def fit_head_shell(shell, outer, d):
    """Bottom access slot and integral bridge supporting the pitch output yoke."""
    # The opening is an allocation; cable flex and an installation sequence need review.
    shell -= Pos(0, 0, -64) * Box(d.head_depth_m * 1000 + 10, 58, 100)
    shell -= Pos(0, 0, d.neck_lower_port_top_m * 1000 - 50) * Box(
        d.head_depth_m * 1000 + 10, d.neck_lower_port_width_m * 1000, 100
    )
    bridge = (Pos(0, 0, 32) * Box(20, d.head_width_m * 1000, 3)) & outer
    shell += bridge
    for x, y in head_bridge_positions():
        shell -= Pos(x, y, 32) * Cylinder(1.65, 6)
        top = d.head_height_m * 500 + 5
        shell -= Pos(x, y, (34 + top) / 2) * Cylinder(3.5, top - 34)
    if not shell.is_valid or len(shell.solids()) != 1:
        raise ValueError("head bridge is not connected to shell")
    return shell


def bridge_hardware(length_mm=12, nut_clock_deg=0):
    """M3 through 6 mm, two washers and nut; Z=0 lower stock face."""
    if length_mm not in (10, 12):
        raise ValueError("bridge stack requires a selected 10 or 12 mm screw")
    washer = Cylinder(3.5, 0.5) - Cylinder(1.6, 1)
    nut = extrude(RegularPolygon(2.75, 6, major_radius=False), amount=2.4)
    nut -= Pos(0, 0, 1.2) * Cylinder(1.25, 4)
    nut = Rot(0, 0, nut_clock_deg) * nut
    return Compound(
        [
            Pos(0, 0, 6.5 - length_mm / 2) * Cylinder(1.5, length_mm),
            Pos(0, 0, 8) * Cylinder(2.75, 3),
            Pos(0, 0, 6.25) * washer,
            Pos(0, 0, -0.25) * washer,
            Pos(0, 0, -2.9) * nut,
        ]
    )
