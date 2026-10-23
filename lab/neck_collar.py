"""Split fixed lower-neck collar and independent M2.5 mounting, in torso mm.

A partial yaw-mechanism cover. The open upper joint/head interfaces are not
finger-proof, and the nominal pilot/insert fit is not a strength qualification.
"""

from build123d import Box, Cylinder, Pos, fillet

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

TAB_THICKNESS_MM = 2.0
POST_TIP_WALL_MM = 1.5
SPLIT_GAP_MM = 0.2
MOUNTING_X_MM = (-27.5, 27.5)
MOUNTING_Y_MM = (-34.0, 34.0)


def centers(d):
    return [
        (x, y, d.torso_height_m * 1000) for x in MOUNTING_X_MM for y in MOUNTING_Y_MM
    ]


def collar_parts(d):
    radius = d.neck_collar_radius_m * 1000
    wall = d.neck_collar_wall_m * 1000
    height = d.neck_collar_height_m * 1000
    bottom = d.torso_height_m * 1000
    outer = Pos(0, 0, bottom + height / 2) * Cylinder(radius, height)
    ring = outer - Pos(0, 0, bottom + height / 2) * Cylinder(radius - wall, height + 2)
    ring = fillet(ring.edges(), min(wall * 0.3, 0.8))
    for x, y, z in centers(d):
        # A radial-width tab connects the seat to the outer wall. Trimming by the
        # exterior keeps the same silhouette as radius varies.
        sign = 1 if y > 0 else -1
        inner_y = abs(y) - 4
        end_y = radius + 1
        tab = Pos(x, sign * (inner_y + end_y) / 2, z + TAB_THICKNESS_MM / 2) * Box(
            8, end_y - inner_y, TAB_THICKNESS_MM
        )
        ring += tab & outer
        ring -= Pos(x, y, z + TAB_THICKNESS_MM / 2) * Cylinder(
            1.35, TAB_THICKNESS_MM + 2
        )
    result = {}
    for name, sign in (("front", 1), ("rear", -1)):
        half = ring & (
            Pos(sign * (100 + SPLIT_GAP_MM / 2), 0, bottom + height / 2)
            * Box(200, 200, height + 2)
        )
        if not half.is_valid or len(half.solids()) != 1:
            raise ValueError("collar half or tabs disconnected")
        result[name] = half
    return result


def fit_front_shell(front, d):
    """Independent bosses in the existing torso front shell; no clamp-stack reuse."""
    offset = Pos(0, 0, -d.torso_height_m * 500)
    stock_depth = PILOT_DEPTH_MM + POST_TIP_WALL_MM
    for x, y, z in centers(d):
        front += (
            offset
            * Pos(x, y, z - stock_depth / 2)
            * Cylinder(POST_RADIUS_MM, stock_depth)
        )
        front -= (
            offset
            * Pos(x, y, z - PILOT_DEPTH_MM / 2 + 0.01)
            * Cylinder(PILOT_DIAMETER_MM / 2, PILOT_DEPTH_MM + 0.02)
        )
    if not front.is_valid or len(front.solids()) != 1:
        raise ValueError("collar mounting posts disconnected from torso front")
    return front


def hardware(d):
    screws, inserts = [], []
    for x, y, z in centers(d):
        head_seat = z + TAB_THICKNESS_MM
        screw = Pos(x, y, head_seat - SCREW_LENGTH_MM / 2) * Cylinder(
            1.25, SCREW_LENGTH_MM
        )
        screw += Pos(x, y, head_seat + SCREW_HEAD_HEIGHT_MM / 2) * Cylinder(
            SCREW_HEAD_DIAMETER_MM / 2, SCREW_HEAD_HEIGHT_MM
        )
        insert = Pos(x, y, z - INSERT_LENGTH_MM / 2) * (
            Cylinder(INSERT_DIAMETER_MM / 2, INSERT_LENGTH_MM)
            - Cylinder(1.25, INSERT_LENGTH_MM + 1)
        )
        screws.append(screw)
        inserts.append(insert)
    return screws, inserts


def interface(d):
    return {
        "schema_version": 1,
        "parent": "torso",
        "units": "mm",
        "halves": ["front", "rear"],
        "split_axis": "+X front / -X rear",
        "split_gap_mm": SPLIT_GAP_MM,
        "outer_radius_mm": d.neck_collar_radius_m * 1000,
        "wall_mm": d.neck_collar_wall_m * 1000,
        "rim_radius_mm": min(d.neck_collar_wall_m * 300, 0.8),
        "height_mm": d.neck_collar_height_m * 1000,
        "mounting_centers_mm": centers(d),
        "tab_thickness_mm": TAB_THICKNESS_MM,
        "post_radius_mm": POST_RADIUS_MM,
        "pilot_diameter_mm": PILOT_DIAMETER_MM,
        "pilot_depth_mm": PILOT_DEPTH_MM,
        "pilot_tip_wall_mm": POST_TIP_WALL_MM,
        "nominal_insert_radial_wall_mm": POST_RADIUS_MM - INSERT_DIAMETER_MM / 2,
        "insert": "ruthex RX-M2.5x5.7",
        "screw": "PTS A91202506 M2.5x6",
        "per_robot": {"cover_half": 2, "insert": 4, "screw": 4},
        "nominal_thread_engagement_mm": SCREW_LENGTH_MM - TAB_THICKNESS_MM,
        "scope": "fixed lower-neck cover; separate fasteners; open top/head gap is not pinch protection",
    }


def clearance_envelope(d, margin_mm):
    """Conservative wall/tab clearance solid in torso mm; excludes no openings.

    Expanded radial/Z intervals enclose a Euclidean margin around the unfilleted
    ring. Expanded tab boxes enclose the tab stock before exterior trimming.
    Filling holes and the split can cause conservative failures, never justify
    a clearance pass by those openings. This is a checking solid, not a part.
    """
    import math

    if not math.isfinite(margin_mm) or not 0 < margin_mm < 5:
        raise ValueError("clearance margin must be finite and between 0 and 5 mm")
    radius = d.neck_collar_radius_m * 1000
    wall = d.neck_collar_wall_m * 1000
    height = d.neck_collar_height_m * 1000
    bottom = d.torso_height_m * 1000
    center = bottom + height / 2
    result = Pos(0, 0, center) * Cylinder(radius + margin_mm, height + 2 * margin_mm)
    result -= Pos(0, 0, center) * Cylinder(
        radius - wall - margin_mm, height + 2 * margin_mm + 2
    )
    for x, y, z in centers(d):
        sign = 1 if y > 0 else -1
        inner = abs(y) - 4
        end = radius + 1
        result += Pos(x, sign * (inner + end) / 2, z + TAB_THICKNESS_MM / 2) * Box(
            8 + 2 * margin_mm,
            end - inner + 2 * margin_mm,
            TAB_THICKNESS_MM + 2 * margin_mm,
        )
    if not result.is_valid:
        raise ValueError("invalid collar clearance envelope")
    return result
