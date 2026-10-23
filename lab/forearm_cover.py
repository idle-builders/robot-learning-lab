"""Experimental split forearm enclosure in owner-local millimetres.

Open elbow and finger interfaces are intentional. This recipe is not enabled in
the production geometry. Insert installation and strength require hardware work.
"""

from build123d import (
    Box,
    Cylinder,
    Plane,
    Pos,
    Rectangle,
    RectangleRounded,
    Rot,
    extrude,
    loft,
)

from .logic_mount import (
    INSERT_DIAMETER_MM,
    INSERT_LENGTH_MM,
    PILOT_DEPTH_MM,
    PILOT_DIAMETER_MM,
    SCREW_HEAD_DIAMETER_MM,
    SCREW_HEAD_HEIGHT_MM,
)


def cylinder_y(x, y, z, radius, length):
    return Pos(x, y, z) * Rot(90, 0, 0) * Cylinder(radius, length)


def candidate(spine, spec, *, right=False):
    """Return separate cover halves, modified spine and nominal hardware.

    Input spine already carries the appropriate left/right recipe. Seating is at
    the inside lateral wall; no boss volume is shared by spine and cover.
    """
    c = spec

    def section(width, radius, inset):
        profile = RectangleRounded(
            width - 2 * inset,
            c["outer_y"] - 2 * inset,
            radius - inset,
        )
        if "rear_extent_x" not in c:
            return profile
        extent = c["rear_extent_x"]
        rear_radius = c["rear_corner_radius"]
        if not c["wall"] < rear_radius <= extent <= width / 2:
            raise ValueError(
                "rear profile requires wall < radius <= extent <= half width"
            )
        # Preserve the original front contour. The rear gets its own section
        # radius and negative-X extent without moving mounts or the split plane.
        rear = Pos((width / 2 - extent) / 2, 0) * RectangleRounded(
            width / 2 + extent - 2 * inset,
            c["outer_y"] - 2 * inset,
            rear_radius - inset,
        )
        return (profile & (Pos(100, 0) * Rectangle(200, 200))) + (
            rear & (Pos(-100, 0) * Rectangle(200, 200))
        )

    def envelope(inset):
        upper = section(c["upper_x"], c["upper_corner_radius"], inset)
        lower = section(c["outer_x"], c["corner_radius"], inset)
        # Overshoot the cavity ends so no accidental closing lip can remain.
        extension = 1 if inset else 0
        shape = Pos(0, 0, c["taper_upper_z"]) * extrude(
            upper, amount=c["proximal_z"] - c["taper_upper_z"] + extension
        )
        shape += Pos(0, 0, c["distal_z"] - extension) * extrude(
            lower, amount=c["taper_lower_z"] - c["distal_z"] + extension
        )
        shape += loft(
            [
                Pos(0, 0, c["taper_lower_z"]) * lower,
                Pos(0, 0, c["taper_upper_z"]) * upper,
            ],
            ruled=True,
        )
        return shape

    ring = envelope(0) - envelope(c["wall"])
    # Full-width opening exposes the moving finger on the medial distal side.
    ring -= Pos(0, c["finger_relief_y"] - 100, c["finger_relief_z"] - 100) * Box(
        200, 200, 200
    )
    # Open proximal front relief follows the measured folded shoulder-clamp
    # envelope. Rounded lower corners preserve a connected front shell.
    notch = (
        Pos(
            c["fold_relief_x"],
            0,
            (c["fold_relief_bottom_z"] + c["fold_relief_top_z"]) / 2,
        )
        * Rot(0, 90, 0)
        * extrude(
            RectangleRounded(
                c["fold_relief_top_z"] - c["fold_relief_bottom_z"],
                c["fold_relief_width_y"],
                c["fold_relief_radius"],
            ),
            amount=40,
        )
    )
    ring -= notch
    seat = c["outer_y"] / 2 - c["wall"]
    head_seat = c["outer_y"] / 2
    half_shapes = {}
    for name, sign in [("front", 1), ("rear", -1)]:
        half_shapes[name] = ring & (
            Pos(sign * (100 + c["split_gap"] / 2), 0, -70) * Box(200, 200, 200)
        )
    if "rear_guard_extent_x" in c:
        extent = c["rear_guard_extent_x"]
        minimum = max(-x for x in c["mount_x"]) + c["boss_radius"] + c["wall"]
        if "rear_extent_x" in c or not minimum <= extent <= c["outer_x"] / 2:
            raise ValueError(
                "rear guard requires original contour and intact mount margin"
            )
        # An explicitly open lateral guard: retain only the mounted side of the
        # existing shell. This removes material; it does not move a wall into
        # the gripper screw/finger envelope or imply complete rear protection.
        half_shapes["rear"] &= Pos(
            -extent / 2, c["outer_y"] / 2, (c["proximal_z"] + c["distal_z"]) / 2
        ) * Box(extent, c["outer_y"], c["proximal_z"] - c["distal_z"] + 2)

    # First construct the left stock; mirror each additive/cutting tool for right.
    def mirror(shape):
        return shape.mirror(Plane.XZ) if right else shape

    modified = spine
    screws, inserts, tools = {}, {}, {}
    for i, x in enumerate(c["mount_x"]):
        for j, z in enumerate(c["mount_z"]):
            key = f"{i}_{j}"
            start = c["spine_face_y"] - 0.5
            stock = cylinder_y(x, (start + seat) / 2, z, c["boss_radius"], seat - start)
            modified += mirror(stock)
            pilot = cylinder_y(
                x,
                seat - PILOT_DEPTH_MM / 2 + 0.01,
                z,
                PILOT_DIAMETER_MM / 2,
                PILOT_DEPTH_MM + 0.02,
            )
            modified -= mirror(pilot)
            for name in half_shapes:
                half_shapes[name] -= cylinder_y(x, head_seat, z, 1.35, 8)
            screw = cylinder_y(x, head_seat - 4, z, 1.25, 8)
            screw += cylinder_y(
                x,
                head_seat + SCREW_HEAD_HEIGHT_MM / 2,
                z,
                SCREW_HEAD_DIAMETER_MM / 2,
                SCREW_HEAD_HEIGHT_MM,
            )
            insert = cylinder_y(
                x,
                seat - INSERT_LENGTH_MM / 2,
                z,
                INSERT_DIAMETER_MM / 2,
                INSERT_LENGTH_MM,
            )
            insert -= cylinder_y(
                x, seat - INSERT_LENGTH_MM / 2, z, 1.25, INSERT_LENGTH_MM + 1
            )
            screws[key], inserts[key] = mirror(screw), mirror(insert)
            # Tool envelope is a declared design allowance, not a sourced tool.
            tools[key] = mirror(cylinder_y(x, head_seat + 18, z, 2.5, 30))
    halves = {name: mirror(shape) for name, shape in half_shapes.items()}
    for name, shape in {**halves, "spine": modified}.items():
        if not shape.is_valid or len(shape.solids()) != 1:
            raise ValueError(f"invalid/disconnected enclosure part: {name}")
    return dict(
        halves=halves,
        spine=modified,
        screws=screws,
        inserts=inserts,
        tools=tools,
        seat_y=(-seat if right else seat),
        screw_engagement_mm=8 - c["wall"],
        insert_thread_overlap_mm=min(8 - c["wall"], INSERT_LENGTH_MM),
    )
