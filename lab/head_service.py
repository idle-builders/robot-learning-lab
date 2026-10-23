"""Removable rear head cover and computer carrier; nominal head-local mm."""

from dataclasses import replace

from build123d import Box, Cylinder, Pos, Rot, RegularPolygon, extrude

from .logic_mount import (
    INSERT_DIAMETER_MM,
    INSERT_LENGTH_MM,
    PILOT_DIAMETER_MM,
    PILOT_DEPTH_MM,
)

SPLIT_GAP_MM = 0.25
REAR_SEAT_MM = 2.5
FRONT_STOCK_MM = 8.4
SCREW_LENGTH_MM = 8.0
PI_THICKNESS_MM = 1.6
PI_HOLE_DIAMETER_MM = 2.75
PI_POST_DEPTH_MM = 8.4
PI_RAIL_THICKNESS_MM = 3.0


def cylinder_y(x, y, z, radius, length):
    return Pos(x, y, z) * Rot(90, 0, 0) * Cylinder(radius, length)


def socket_tool_void():
    return extrude(RegularPolygon(1 / (3**0.5 / 2), 6), amount=1.5)


def pi_centers(settings):
    x, y, z = (v * 1000 for v in settings.pi_position_head_m)
    mx, mz = (29, 11.5) if settings.pi_orientation == "horizontal" else (11.5, 29)
    return [
        (x + dx, y + PI_THICKNESS_MM, z + dz) for dx in (-mx, mx) for dz in (-mz, mz)
    ]


def pi_parts(settings):
    """Unpopulated substrate and selected hardware; component CAD remains open."""
    x, y, z = (v * 1000 for v in settings.pi_position_head_m)
    width, height = (65, 30) if settings.pi_orientation == "horizontal" else (30, 65)
    board = Pos(x, y + PI_THICKNESS_MM / 2, z) * Box(width, PI_THICKNESS_MM, height)
    result = {}
    for i, (px, py, pz) in enumerate(pi_centers(settings)):
        board -= cylinder_y(px, y, pz, PI_HOLE_DIAMETER_MM / 2, 6)
        screw = cylinder_y(px, y + 3, pz, 1.25, 6) + cylinder_y(
            px, y - 1.25, pz, 2.25, 2.5
        )
        screw -= Pos(px, y - 2.75, pz) * Rot(-90, 0, 0) * socket_tool_void()
        result[f"pi_screw_{i}"] = screw
        result[f"pi_insert_{i}"] = cylinder_y(
            px, py + INSERT_LENGTH_MM / 2, pz, INSERT_DIAMETER_MM / 2, INSERT_LENGTH_MM
        ) - cylinder_y(px, py + INSERT_LENGTH_MM / 2, pz, 1.25, INSERT_LENGTH_MM + 1)
    result["pi_substrate"] = board
    return result


def pi_reservation(settings):
    """Planning component space, with assumed empty mounting zones explicitly cut."""
    x, y, z = (v * 1000 for v in settings.pi_position_head_m)
    width, height = (65, 30) if settings.pi_orientation == "horizontal" else (30, 65)
    space = Pos(x, y - 2.25, z) * Box(width, 11.5, height)
    for px, _, pz in pi_centers(settings):
        space -= cylinder_y(px, y, pz, 4.25, 30)
    return space


def inner_envelope(g, inset_mm):
    from .mechanical import primitive

    inset = inset_mm / 1000
    if g.shape == "ellipsoid":
        return primitive(replace(g, size=tuple(v - inset for v in g.size)))
    size = tuple(v - 2 * inset for v in g.size)
    radius = min(g.size) * g.rounding - inset
    return primitive(replace(g, size=size, rounding=radius / min(size)))


def add_pi_carrier(rear, outer, inner, d, settings):
    x, y, z = (v * 1000 for v in settings.pi_position_head_m)
    root = -d.head_depth_m * 500
    width, height = (65, 30) if settings.pi_orientation == "horizontal" else (30, 65)
    tip = x + width / 2 + 1
    mount_z = height / 2 - 3.5
    rear_zone = Pos(settings.split_x_m * 1000 - SPLIT_GAP_MM / 2 - 150, 0, 0) * Box(
        300, 300, 300
    )
    allowed = inner + (outer & rear_zone)
    for dz in (-mount_z, mount_z):
        rib = Pos((root + tip) / 2, y + 10, z + dz) * Box(
            tip - root, PI_RAIL_THICKNESS_MM, 8
        )
        rear += rib & allowed
    rear += (
        Pos(x + width / 2 - 3.5, y + 10, z)
        * Box(8, PI_RAIL_THICKNESS_MM, 2 * mount_z + 8)
    ) & allowed
    for px, py, pz in pi_centers(settings):
        rear += cylinder_y(px, py + PI_POST_DEPTH_MM / 2, pz, 4, PI_POST_DEPTH_MM)
        rear -= cylinder_y(
            px,
            py + PILOT_DEPTH_MM / 2 - 0.01,
            pz,
            PILOT_DIAMETER_MM / 2,
            PILOT_DEPTH_MM + 0.02,
        )
    if not rear.is_valid or len(rear.solids()) != 1:
        raise ValueError("Pi carrier is disconnected from rear cap")
    return rear


def cylinder_x(x, y, z, radius, length):
    return Pos(x, y, z) * Rot(0, 90, 0) * Cylinder(radius, length)


def centers(d):
    return [(sign * d.head_width_m * 380, z) for sign in (-1, 1) for z in (-15, 15)]


def split_parts(shell, outer, d, settings, head_geometry=None):
    split = settings.split_x_m * 1000
    front_start = split + SPLIT_GAP_MM / 2
    rear_end = split - SPLIT_GAP_MM / 2
    front = shell & (Pos(front_start + 150, 0, 0) * Box(300, 300, 300))
    rear = shell & (Pos(rear_end - 150, 0, 0) * Box(300, 300, 300))
    for y, z in centers(d):
        sign = 1 if y > 0 else -1
        end = d.head_width_m * 500 + 1
        inner = abs(y) - 4
        # Radial tabs reach actual curved stock. Clip to the original exterior.
        for name, start, length in (
            ("front", front_start, FRONT_STOCK_MM),
            ("rear", rear_end - REAR_SEAT_MM, REAR_SEAT_MM),
        ):
            tab = Pos(start + length / 2, sign * (inner + end) / 2, z) * Box(
                length, end - inner, 8
            )
            tab &= outer
            if name == "front":
                front += tab
            else:
                rear += tab
        # The cosmetic seam stays open while the circular bearing pad bridges
        # it. Otherwise tightening would pull the cap away from its CAD datum.
        front += cylinder_x(
            rear_end + (FRONT_STOCK_MM + SPLIT_GAP_MM) / 2,
            y,
            z,
            4,
            FRONT_STOCK_MM + SPLIT_GAP_MM,
        )
        front -= cylinder_x(
            rear_end + PILOT_DEPTH_MM / 2 - 0.01,
            y,
            z,
            PILOT_DIAMETER_MM / 2,
            PILOT_DEPTH_MM + 0.02,
        )
        rear -= cylinder_x(rear_end - 50, y, z, 1.35, 100)
        seat = rear_end - REAR_SEAT_MM
        rear -= cylinder_x(seat - 50, y, z, 2.5, 100)
    if settings.pi_mount_enabled:
        if head_geometry is None:
            raise ValueError("Pi carrier requires the shared head geometry recipe")
        rear = add_pi_carrier(
            rear,
            outer,
            inner_envelope(head_geometry, d.shell_wall_m * 1000 + 0.3),
            d,
            settings,
        )
    for name, shape in (("front", front), ("rear", rear)):
        if not shape.is_valid or len(shape.solids()) != 1:
            raise ValueError(f"head {name} cover disconnected or invalid")
    return dict(front=front, rear=rear)


def hardware(d, settings):
    split = settings.split_x_m * 1000
    seat = split - SPLIT_GAP_MM / 2 - REAR_SEAT_MM
    start = split - SPLIT_GAP_MM / 2
    result = {}
    for i, (y, z) in enumerate(centers(d)):
        screw = cylinder_x(seat + SCREW_LENGTH_MM / 2, y, z, 1.25, SCREW_LENGTH_MM)
        screw += cylinder_x(seat - 1.25, y, z, 2.25, 2.5)
        screw -= Pos(seat - 2.75, y, z) * Rot(0, 90, 0) * socket_tool_void()
        result[f"head_cover_screw_{i}"] = screw
        result[f"head_cover_insert_{i}"] = cylinder_x(
            start + INSERT_LENGTH_MM / 2, y, z, INSERT_DIAMETER_MM / 2, INSERT_LENGTH_MM
        ) - cylinder_x(start + INSERT_LENGTH_MM / 2, y, z, 1.25, INSERT_LENGTH_MM + 1)
    if settings.pi_mount_enabled:
        result.update(pi_parts(settings))
    return result


def interface(d, settings):
    return dict(
        schema_version=1,
        parent="head",
        units="mm",
        split_x_mm=settings.split_x_m * 1000,
        split_gap_mm=SPLIT_GAP_MM,
        mount_yz_mm=centers(d),
        screw="PTS A91202508 M2.5x8",
        insert="ruthex RX-M2.5x5.7",
        per_robot=dict(rear_cover=1, screw=4, insert=4),
        rear_seat_mm=REAR_SEAT_MM,
        front_stock_mm=FRONT_STOCK_MM,
        nominal_thread_overlap_mm=SCREW_LENGTH_MM - REAR_SEAT_MM,
        removal_axis="head -X at neutral; screws removed and cables disconnected",
        qualification="nominal candidate; material/thread/load/process unqualified",
        pi_carrier=dict(
            enabled=settings.pi_mount_enabled,
            orientation=settings.pi_orientation,
            board_to_head_rotation=(
                [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
                if settings.pi_orientation == "horizontal"
                else [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
            ),
            position_head_mm=[v * 1000 for v in settings.pi_position_head_m],
            mount_xyz_mm=pi_centers(settings),
            board_outline_mm=[65, 30],
            substrate_thickness_mm=PI_THICKNESS_MM,
            hole_diameter_mm=PI_HOLE_DIAMETER_MM,
            component_occupied_board_z_mm=[-3.5, 8],
            assumed_component_free_mount_radius_mm=4.25,
            post_depth_mm=PI_POST_DEPTH_MM,
            screw="PTS A91202506 M2.5x6",
            insert="ruthex RX-M2.5x5.7",
            per_robot=dict(substrate=1, screw=4, insert=4)
            if settings.pi_mount_enabled
            else {},
            evidence="outline/hole locations from drawing; hole size cross-family, thickness/component-free zones and heights unverified; connector/antenna keepouts absent",
        ),
    )
