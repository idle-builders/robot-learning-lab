"""Removable, open-backed upper-arm guard; no physical qualification.

Owned by upper_arm, covering its spine and the fixed elbow housing flanks.
It is distinct from the earlier shoulder_roll-owned shoulder shell. Millimetres.
"""

from build123d import Box, Pos, RectangleRounded, extrude
from lab.shin_cover import cylinder_x
from lab.logic_mount import (
    INSERT_DIAMETER_MM,
    INSERT_LENGTH_MM,
    PILOT_DEPTH_MM,
    PILOT_DIAMETER_MM,
    SCREW_HEAD_DIAMETER_MM,
    SCREW_HEAD_HEIGHT_MM,
)


def candidate(spine, dimensions, spec, profile, side="left"):
    c = spec
    wall = c["wall"]
    radius = c["profiles"][profile]["corner_radius"]
    bottom = -1000 * dimensions.upper_arm_m - c["distal_below_elbow"]
    top = c["proximal_z"]
    seat, front = c["outer_x"] / 2 - wall, c["outer_x"] / 2
    if not 0 < wall < radius < min(c["outer_x"], c["outer_y"]) / 2:
        raise ValueError("Invalid wall/radius")
    if seat - c["spine_face_x"] < PILOT_DEPTH_MM:
        raise ValueError("Insert pilot would enter original spine")
    if not bottom < c["fold_opening_top_z"] < top:
        raise ValueError("Fold opening outside guard")
    shell = Pos(0, 0, bottom) * extrude(
        RectangleRounded(c["outer_x"], c["outer_y"], radius), amount=top - bottom
    )
    shell -= Pos(0, 0, bottom - 1) * extrude(
        RectangleRounded(
            c["outer_x"] - 2 * wall, c["outer_y"] - 2 * wall, radius - wall
        ),
        amount=top - bottom + 2,
    )
    shell &= Pos(c["rear_open_x"] + 100, 0, (bottom + top) / 2) * Box(
        200, 200, top - bottom + 2
    )
    # Leave side guards; the front opens for the elbow's forward fold.
    shell -= Pos(100, 0, c["fold_opening_top_z"] - 100) * Box(
        200, c["fold_opening_width"], 200
    )
    if side not in ("left", "right"):
        raise ValueError("Unknown side")
    if "medial_opening_top_z" in c:
        # Remove the lower inboard wing that met the thigh screws in v1.
        sign = -1 if side == "left" else 1
        shell -= Pos(0, sign * 100, c["medial_opening_top_z"] - 100) * Box(
            200, 200, 200
        )
    modified = spine
    screws, inserts, tools, mounts = {}, {}, {}, []
    for i, y in enumerate(c["mount_y"]):
        for j, z in enumerate(c["mount_z"]):
            if (
                not c["fold_opening_top_z"] + c["boss_radius"]
                < z
                < top - c["boss_radius"]
            ):
                raise ValueError("Mount leaves front cuff stock")
            if abs(y) + c["boss_radius"] > c["outer_y"] / 2 - radius:
                raise ValueError("Mount leaves flat seating face")
            key = f"{i}_{j}"
            start = c["spine_face_x"] - 0.5
            stock = cylinder_x((start + seat) / 2, y, z, c["boss_radius"], seat - start)
            if sum(s.volume for s in (stock & spine).solids()) < 0.1:
                raise ValueError("Mount does not join source spine")
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
    for name, shape in dict(
        cover=shell,
        spine=modified,
        **screws,
        **{f"insert_{k}": v for k, v in inserts.items()},
    ).items():
        if not shape.is_valid or len(shape.solids()) != 1:
            raise ValueError(f"Invalid or disconnected guard part: {name}")
    return dict(
        cover=shell,
        spine=modified,
        added_spine=modified - spine,
        screws=screws,
        inserts=inserts,
        tools=tools,
        mounts=mounts,
    )
