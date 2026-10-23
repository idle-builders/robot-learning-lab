"""Printable interface specimens derived from existing robot mounting recipes.

Millimetres throughout. These parts test dimensional fit; their load capacity
and print process have not been identified. They are not robot replacements.
"""

from build123d import Box, Cylinder, Pos, Rot

from .joint_module import clamp_bolt_centers, joint_parts
from .logic_mount import (
    INSERT_DIAMETER_MM,
    INSERT_LENGTH_MM,
    PILOT_DEPTH_MM,
    SCREW_HEAD_DIAMETER_MM,
    SCREW_HEAD_HEIGHT_MM,
)


def insert_stack(cover, bench, pilot_diameter):
    """Copy the cover-to-spine axial stack into an isolated flat specimen."""
    cfg = bench["insert_coupon"]
    seat = cover["outer_y"] / 2 - cover["wall"]
    stand_off = seat - cover["spine_face_y"]
    top = cfg["base_thickness"] + stand_off
    boss_start = cfg["base_thickness"] - 0.5  # Same 0.5 mm shared stock as robot.
    if not 0 < pilot_diameter < INSERT_DIAMETER_MM < 2 * cover["boss_radius"]:
        raise ValueError("pilot/insert/boss diameters have invalid order")
    if top - PILOT_DEPTH_MM <= cfg["base_thickness"]:
        raise ValueError("pilot breaks through the coupon's attachment base")
    block = Pos(0, 0, cfg["base_thickness"] / 2) * Box(
        cfg["base_width"], cfg["base_width"], cfg["base_thickness"]
    )
    block += Pos(0, 0, (boss_start + top) / 2) * Cylinder(
        cover["boss_radius"], top - boss_start
    )
    block -= Pos(0, 0, top - PILOT_DEPTH_MM / 2 + 0.01) * Cylinder(
        pilot_diameter / 2, PILOT_DEPTH_MM + 0.02
    )
    lid = Pos(0, 0, top + cover["wall"] / 2) * Box(
        cfg["base_width"], cfg["base_width"], cover["wall"]
    )
    lid -= Pos(0, 0, top) * Cylinder(cfg["cover_clearance_diameter"] / 2, 10)
    insert = Pos(0, 0, top - INSERT_LENGTH_MM / 2) * Cylinder(
        INSERT_DIAMETER_MM / 2, INSERT_LENGTH_MM
    )
    insert -= Pos(0, 0, top - INSERT_LENGTH_MM / 2) * Cylinder(
        cfg["screw_diameter"] / 2, INSERT_LENGTH_MM + 1
    )
    head_seat = top + cover["wall"]
    screw = Pos(0, 0, head_seat - cfg["screw_length"] / 2) * Cylinder(
        cfg["screw_diameter"] / 2, cfg["screw_length"]
    )
    screw += Pos(0, 0, head_seat + SCREW_HEAD_HEIGHT_MM / 2) * Cylinder(
        SCREW_HEAD_DIAMETER_MM / 2, SCREW_HEAD_HEIGHT_MM
    )
    thread_entry = cfg["screw_length"] - cover["wall"]
    return dict(block=block, cover=lid, insert=insert, screw=screw), dict(
        pilot_diameter_mm=pilot_diameter,
        pilot_depth_mm=PILOT_DEPTH_MM,
        boss_diameter_mm=2 * cover["boss_radius"],
        boss_height_above_base_mm=stand_off,
        insert_length_mm=INSERT_LENGTH_MM,
        insert_max_diameter_mm=INSERT_DIAMETER_MM,
        cover_wall_mm=cover["wall"],
        screw_length_mm=cfg["screw_length"],
        nominal_thread_overlap_mm=min(thread_entry, INSERT_LENGTH_MM),
        tip_to_pilot_bottom_mm=PILOT_DEPTH_MM - thread_entry,
        minimum_nominal_radial_stock_mm=cover["boss_radius"] - INSERT_DIAMETER_MM / 2,
        cover_shaft_radial_clearance_mm=(
            cfg["cover_clearance_diameter"] - cfg["screw_diameter"]
        )
        / 2,
        insert_to_print_interference="intentional heat-install envelope; material flow not simulated",
        stack_height_mm=head_seat + SCREW_HEAD_HEIGHT_MM,
    )


def actuator_cradle(bench):
    """Reuse open clamp halves and add a fixture plate outside the case cavity."""
    c = bench["actuator_cradle"]
    parts = joint_parts(
        clearance_mm=c["case_clearance"], split_gap_mm=c["split_gap"], clamped=False
    )
    lower, upper = parts["clamp_lower"], parts["clamp_upper"]
    bounds = lower.bounding_box()
    cx = (bounds.min.X + bounds.max.X) / 2
    inner = bounds.min.Y + c["plate_overlap"]
    outer = inner - c["plate_thickness"]
    plate = Pos(cx, (outer + inner) / 2, 0) * Box(
        c["plate_width"], c["plate_thickness"], c["plate_depth"]
    )
    margin = (
        min(
            (c["plate_width"] - c["mounting_pitch_x"]) / 2,
            (c["plate_depth"] - c["mounting_pitch_z"]) / 2,
        )
        - c["mounting_hole_diameter"] / 2
    )
    if margin < c["hole_edge_margin_min"]:
        raise ValueError("fixture mounting holes violate the declared edge margin")
    for x in [-c["mounting_pitch_x"] / 2, c["mounting_pitch_x"] / 2]:
        for z in [-c["mounting_pitch_z"] / 2, c["mounting_pitch_z"] / 2]:
            plate -= (
                Pos(cx + x, (outer + inner) / 2, z)
                * Rot(90, 0, 0)
                * Cylinder(c["mounting_hole_diameter"] / 2, c["plate_thickness"] + 2)
            )
    lower += plate
    # The added fixture plate otherwise caps the original clamp's bolt bores.
    # Cut through the complete joined stock, sharing the module's bore pattern.
    for x, z in clamp_bolt_centers():
        lower -= Pos(x, (outer + bounds.max.Y) / 2, z) * Rot(90, 0, 0) * Cylinder(
            1.65, bounds.max.Y - outer + 2
        )
    # Rotate the fixture's outside plate face to a build plane at Z=0.
    transform = Pos(0, 0, -outer) * Rot(90, 0, 0)
    return {
        "cradle": transform * lower,
        "clamp_cap": transform * upper,
        "case_reference": transform * parts["servo_keepout"],
    }, dict(
        mounting_hole_diameter_mm=c["mounting_hole_diameter"],
        mounting_pitch_mm=[c["mounting_pitch_x"], c["mounting_pitch_z"]],
        edge_margin_mm=margin,
        case_clearance_per_side_mm=c["case_clearance"],
        open_split_mm=c["split_gap"],
        nominal_closed_split_mm=c["split_gap"] - 2 * c["case_clearance"],
        clamp_bolt_holes="Four shared Ø3.3 mm bores through the clamp and fixture plate; clamp preload remains unqualified",
        clamp_bolt_centers_xy_mm=[(x, -z) for x, z in clamp_bolt_centers()],
        case_reference="C018 catalog box; vendor STEP interchange and real case shape remain unverified",
        fixture_use="Dimensional fit cradle only. No powered or loaded operation qualification.",
    )
