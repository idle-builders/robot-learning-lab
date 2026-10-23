"""Scout removable visor, in head-local millimetres.

Nominal M2.5 cap screws and heat-set insert envelopes; no thread or material
strength qualification. Boss insertion interference is intentional and explicit.
"""

from dataclasses import replace

from build123d import Box, Cylinder, Pos, RegularPolygon, extrude, fillet

from .logic_mount import (
    INSERT_DIAMETER_MM,
    INSERT_LENGTH_MM,
    PILOT_DEPTH_MM,
    PILOT_DIAMETER_MM,
    POST_RADIUS_MM,
    SCREW_HEAD_DIAMETER_MM,
    SCREW_HEAD_HEIGHT_MM,
    SCREW_LENGTH_MM,
)

SHELL_GAP_MM = 0.3
SEAT_THICKNESS_MM = 2.5
POST_DEPTH_MM = 15.0
COUNTERBORE_DIAMETER_MM = 5.0
CLEARANCE_DIAMETER_MM = 2.7
VISOR_WALL_MM = 2.0


def centers(d):
    return [
        (
            d.head_depth_m * 1000 / 3,
            sign * d.head_width_m * 200,
            d.head_height_m * 500 - 2,
        )
        for sign in (-1, 1)
    ]


def envelope(g, gap_mm):
    """Parallel offset of a rounded box, including its corner radius."""
    from .mechanical import primitive

    if g.shape != "box":
        raise ValueError("visor mount requires rounded box head")
    size = tuple(v + 2 * gap_mm / 1000 for v in g.size)
    radius = min(g.size) * g.rounding + gap_mm / 1000
    return primitive(replace(g, size=size, rounding=radius / min(size)))


def fit_shell(shell, d):
    for x, y, z in centers(d):
        shell += Pos(x, y, z - POST_DEPTH_MM / 2) * Cylinder(
            POST_RADIUS_MM, POST_DEPTH_MM
        )
        shell -= Pos(x, y, z - PILOT_DEPTH_MM / 2 + 0.01) * Cylinder(
            PILOT_DIAMETER_MM / 2, PILOT_DEPTH_MM + 0.02
        )
    if not shell.is_valid or len(shell.solids()) != 1:
        raise ValueError("visor bosses disconnected from head shell")
    return shell


def visor(robot, g):
    from .mechanical import primitive

    head = next(link for link in robot.links if link.name == "head")
    shape = Pos(*(v * 1000 for v in g.pos)) * primitive(g)
    # Open-bottom cap: retain the rounded exterior, remove the solid allocation.
    # Extending the inner box downwards preserves its top and side wall offsets.
    inner_size = [v * 1000 - 2 * VISOR_WALL_MM for v in g.size]
    inner_size[2] += 10
    inner = fillet(
        Box(*inner_size).edges(), min(g.size) * 1000 * g.rounding - VISOR_WALL_MM
    )
    shape -= Pos(g.pos[0] * 1000, g.pos[1] * 1000, g.pos[2] * 1000 - 5) * inner
    for x, y, z in centers(robot.config.dimensions):
        shape += Pos(x, y, z + 1.5) * Cylinder(POST_RADIUS_MM, 3)
    shape -= envelope(head.geometry[0], SHELL_GAP_MM)
    for x, y, z in centers(robot.config.dimensions):
        # Relief below the flat seat admits each boss without side-wall contact.
        shape -= Pos(x, y, z - 30) * Cylinder(POST_RADIUS_MM + 0.2, 60)
        shape -= Pos(x, y, z + 10) * Cylinder(CLEARANCE_DIAMETER_MM / 2, 40)
        shape -= Pos(x, y, z + SEAT_THICKNESS_MM + 10) * Cylinder(
            COUNTERBORE_DIAMETER_MM / 2, 20
        )
    if not shape.is_valid or len(shape.solids()) != 1:
        raise ValueError("visor mating cut produces disconnected or invalid part")
    return shape


def hardware(d):
    result = {}
    for i, (x, y, z) in enumerate(centers(d)):
        seat = z + SEAT_THICKNESS_MM
        screw = Pos(x, y, seat - SCREW_LENGTH_MM / 2) * Cylinder(1.25, SCREW_LENGTH_MM)
        screw += Pos(x, y, seat + SCREW_HEAD_HEIGHT_MM / 2) * Cylinder(
            SCREW_HEAD_DIAMETER_MM / 2, SCREW_HEAD_HEIGHT_MM
        )
        # Socket recess is a hexagonal 2 mm across-flats allocation. Its depth
        # remains provisional; the selected screw source specifies key size.
        socket = extrude(RegularPolygon(1 / (3**0.5 / 2), 6), amount=1.5)
        screw -= Pos(x, y, seat + SCREW_HEAD_HEIGHT_MM - 1.25) * socket
        insert = Pos(x, y, z - INSERT_LENGTH_MM / 2) * (
            Cylinder(INSERT_DIAMETER_MM / 2, INSERT_LENGTH_MM)
            - Cylinder(1.25, INSERT_LENGTH_MM + 1)
        )
        result[f"visor_screw_{i}"] = screw
        result[f"visor_insert_{i}"] = insert
    return result


def interface(d):
    return {
        "schema_version": 1,
        "parent": "head",
        "units": "mm",
        "centers_mm": centers(d),
        "removal_axis": "+Z after screw removal",
        "shell_gap_mm": SHELL_GAP_MM,
        "visor_wall_mm": VISOR_WALL_MM,
        "seat_thickness_mm": SEAT_THICKNESS_MM,
        "post_depth_mm": POST_DEPTH_MM,
        "post_radius_mm": POST_RADIUS_MM,
        "pilot_diameter_mm": PILOT_DIAMETER_MM,
        "pilot_depth_mm": PILOT_DEPTH_MM,
        "counterbore_diameter_mm": COUNTERBORE_DIAMETER_MM,
        "clearance_diameter_mm": CLEARANCE_DIAMETER_MM,
        "screw": "PTS A91202506 M2.5x6",
        "insert": "ruthex RX-M2.5x5.7",
        "thread_engagement_nominal_mm": SCREW_LENGTH_MM - SEAT_THICKNESS_MM,
        "radial_insert_wall_mm": POST_RADIUS_MM - INSERT_DIAMETER_MM / 2,
        "per_robot": {"visor": 1, "screw": 2, "insert": 2},
        "evidence": "nominal candidate; socket depth, plastic and threads unqualified",
    }
