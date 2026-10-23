"""ST3215 shoulder mounting shelf and nominal hardware, in torso millimetres.

The shelf joins the front shell through two ribs. This supplies an explicit
load path and assembly geometry, not strength or fall-survival qualification.
"""

from build123d import Box, Cylinder, Pos, RegularPolygon, extrude, Compound

CLAMP_FACE_MM = 15.365


def shoulder_axes(d):
    lateral = (
        d.torso_width_m / 2 + d.shoulder_outset_m - d.shoulder_pitch_offset_m
    ) * 1000
    return lateral, d.torso_height_m * 1000 - 18


def bolt_positions(d):
    lateral, height = shoulder_axes(d)
    return [
        (x, sign * (lateral + dy), height)
        for sign in [-1, 1]
        for dy in [-38.615, 13.615]
        for x in [-8, 8]
    ]


def mounting_stock(d):
    lateral, height = shoulder_axes(d)
    thickness = d.shoulder_mount_thickness_m * 1000
    top = height - CLAMP_FACE_MM
    bottom = top - thickness
    stock = Pos(0, 0, (top + bottom) / 2) * Box(28, 2 * (lateral + 20), thickness)
    end = d.torso_depth_m * 500
    for y in [-25, 25]:
        stock += Pos((12 + end) / 2, y, (top + bottom) / 2) * Box(
            end - 12, 6, thickness
        )
    for x, y, _ in bolt_positions(d):
        stock -= Pos(x, y, (top + bottom) / 2) * Cylinder(1.65, thickness + 2)
        pocket = extrude(RegularPolygon(2.85, 6, major_radius=False), amount=2.7)
        stock -= Pos(x, y, bottom - 0.1) * pocket
    return stock


def fit_front_shell(front, outer, d):
    """Arguments are centered torso solids, matching mechanical.service_parts."""
    if not d.shoulder_mount_thickness_m:
        return front
    lateral, height = shoulder_axes(d)
    offset = Pos(0, 0, -d.torso_height_m * 500)
    for sign in [-1, 1]:
        # A rectangular service window encloses the allocated circular sweep.
        # Circular Boolean cuts left residual material on one side of this shell;
        # both sides are checked against the actual yoke, not just BRep validity.
        span = d.shoulder_port_radius_m * 2000
        port = Pos(0, sign * lateral, height) * Box(49.5, span, span)
        clamp = Pos(0, sign * (lateral - 12.5), height) * Box(40, 60.23, 31.73)
        front -= offset * port
        front -= offset * clamp
    for x, y, z in bolt_positions(d):
        # Top access assumes the head/neck module has been removed.
        front -= offset * Pos(x, y, z + 35) * Cylinder(3.5, 40)
    front += (offset * mounting_stock(d)) & outer
    if not front.is_valid or len(front.solids()) != 1:
        raise ValueError("shoulder shelf is not connected to front shell")
    return front


def mounting_hardware(d):
    """Eight M3x40 screws, washers and underside captive nuts; threads omitted."""
    top = shoulder_axes(d)[1] - CLAMP_FACE_MM
    bottom = top - d.shoulder_mount_thickness_m * 1000
    pieces = []
    for x, y, height in bolt_positions(d):
        under_head = height + CLAMP_FACE_MM + 0.5
        shank = Pos(x, y, under_head - 20) * Cylinder(1.5, 40)
        head = Pos(x, y, under_head + 1.5) * Cylinder(2.75, 3)
        washer = Cylinder(3.5, 0.5) - Cylinder(1.6, 1)
        nut = extrude(RegularPolygon(2.75, 6, major_radius=False), amount=2.4)
        nut -= Pos(0, 0, 1.2) * Cylinder(1.25, 4)
        pieces.extend(
            [
                shank,
                head,
                Pos(x, y, under_head - 0.25) * washer,
                Pos(x, y, bottom + 0.1) * nut,
            ]
        )
    return Compound(pieces)


def top_tool_spaces(d):
    """6 mm driver shafts above screw heads; head/neck module removed."""
    return [
        Pos(x, y, h + CLAMP_FACE_MM + 3.5 + 0.1 + 30) * Cylinder(3, 60)
        for x, y, h in bolt_positions(d)
    ]
