"""Open-backed shin shell study, in the shared shin owner's local millimetres."""

from build123d import Box, Cylinder, Pos, RectangleRounded, Rot, extrude
from lab.logic_mount import (
    INSERT_DIAMETER_MM,
    INSERT_LENGTH_MM,
    PILOT_DEPTH_MM,
    PILOT_DIAMETER_MM,
    SCREW_HEAD_DIAMETER_MM,
    SCREW_HEAD_HEIGHT_MM,
)


def cylinder_x(x, y, z, radius, length):
    return Pos(x, y, z) * Rot(0, 90, 0) * Cylinder(radius, length)


def candidate(spine, dimensions, spec, profile, side):
    """Return a mounted front-shell hypothesis; original spine is replaced."""
    c = spec
    radius = c["profiles"][profile]["corner_radius"]
    wall = c["wall"]
    if side not in ("left", "right"):
        raise ValueError("Unknown side")
    if not 0 < wall < radius < min(c["outer_x"], c["outer_y"]) / 2:
        raise ValueError("Invalid rounded shell section")
    wheel_z = -dimensions.shin_m * 1000
    bottom = wheel_z - c["distal_extra_below_wheel_axis"]
    height = c["proximal_z"] - bottom
    shell = Pos(0, 0, bottom) * extrude(
        RectangleRounded(c["outer_x"], c["outer_y"], radius), amount=height
    )
    shell -= Pos(0, 0, bottom - 1) * extrude(
        RectangleRounded(
            c["outer_x"] - 2 * wall, c["outer_y"] - 2 * wall, radius - wall
        ),
        amount=height + 2,
    )
    shell &= Pos(c["rear_open_x"] + 100, 0, (bottom + c["proximal_z"]) / 2) * Box(
        200, 200, height + 2
    )
    wheel_sign = 1 if side == "left" else -1
    wheel_sweep = (
        Pos(0, wheel_sign * dimensions.wheel_outset_m * 1000, wheel_z)
        * Rot(90, 0, 0)
        * Cylinder(
            dimensions.wheel_radius_m * 1000 + c["wheel_clearance"],
            dimensions.wheel_width_m * 1000 + 2 * c["wheel_clearance"],
        )
    )
    shell -= wheel_sweep
    seat = c["outer_x"] / 2 - wall
    front = c["outer_x"] / 2
    if seat - c["spine_face_x"] < PILOT_DEPTH_MM:
        raise ValueError("Insufficient boss length for insert pilot")
    modified = spine
    screws, inserts, tools, mounts = {}, {}, {}, []
    for i, y in enumerate(c["mount_y"]):
        for j, z in enumerate(c["mount_z"]):
            key = f"{i}_{j}"
            if not bottom + c["boss_radius"] < z < c["proximal_z"] - c["boss_radius"]:
                raise ValueError("Mount lies outside shell stock")
            if abs(y) + c["boss_radius"] > c["outer_y"] / 2 - radius:
                raise ValueError("Mount leaves the flat front seating surface")
            start = c["spine_face_x"] - 0.5
            stock = cylinder_x((start + seat) / 2, y, z, c["boss_radius"], seat - start)
            modified += stock
            modified -= cylinder_x(
                seat - PILOT_DEPTH_MM / 2 + 0.01,
                y,
                z,
                PILOT_DIAMETER_MM / 2,
                PILOT_DEPTH_MM + 0.02,
            )
            shell -= cylinder_x(front, y, z, 1.35, 8)
            screws[key] = cylinder_x(front - 4, y, z, 1.25, 8) + cylinder_x(
                front + SCREW_HEAD_HEIGHT_MM / 2,
                y,
                z,
                SCREW_HEAD_DIAMETER_MM / 2,
                SCREW_HEAD_HEIGHT_MM,
            )
            inserts[key] = cylinder_x(
                seat - INSERT_LENGTH_MM / 2,
                y,
                z,
                INSERT_DIAMETER_MM / 2,
                INSERT_LENGTH_MM,
            ) - cylinder_x(
                seat - INSERT_LENGTH_MM / 2, y, z, 1.25, INSERT_LENGTH_MM + 1
            )
            tools[key] = cylinder_x(
                front + SCREW_HEAD_HEIGHT_MM + c["driver_length"] / 2,
                y,
                z,
                c["driver_radius"],
                c["driver_length"],
            )
            mounts.append(
                dict(
                    y_mm=y,
                    z_mm=z,
                    seat_x_mm=seat,
                    insert_thread_overlap_mm=min(8 - wall, INSERT_LENGTH_MM),
                    nominal_tip_space_mm=PILOT_DEPTH_MM - (8 - wall),
                )
            )
    for name, shape in {
        "cover": shell,
        "spine": modified,
        **screws,
        **{f"insert_{k}": v for k, v in inserts.items()},
    }.items():
        if not shape.is_valid or len(shape.solids()) != 1:
            raise ValueError(f"Invalid or disconnected candidate {name}")
    return dict(
        cover=shell,
        spine=modified,
        added_spine=modified - spine,
        screws=screws,
        inserts=inserts,
        tools=tools,
        wheel_sweep=wheel_sweep,
        mounts=mounts,
    )
