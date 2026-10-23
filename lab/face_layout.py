"""Nominal eye integrity around the camera service opening; no lighting model."""

from math import degrees

from build123d import Pos, Rot

from .camera_mount import removal_clearance
from .leg_mechanism import overlap
from .mechanical import decorative_shape, part_shape


def check_face(robot, minimum_gap_mm=1.0):
    """Check uncut allocations, so Boolean trimming cannot hide an eye conflict."""
    if not robot.config.camera.cassette_enabled:
        return {"status": "not_run", "reason": "no camera service opening"}
    head = next(link for link in robot.links if link.name == "head")
    opening = removal_clearance(robot)
    brow = [
        Pos(*(v * 1000 for v in g.pos))
        * Rot(*(degrees(v) for v in g.rpy))
        * part_shape(robot, head, i)
        for i, g in enumerate(head.geometry)
        if g.role == "brow"
    ]
    rows = []
    for i, geometry in enumerate(head.geometry):
        if not geometry.role.startswith("eye_"):
            continue
        placement = Pos(*(v * 1000 for v in geometry.pos)) * Rot(
            *(degrees(v) for v in geometry.rpy)
        )
        raw = placement * decorative_shape(robot, geometry)
        made = placement * part_shape(robot, head, i)
        removed = overlap(raw, opening)
        gap = raw.distance_to(opening)
        volume_error = abs(raw.volume - made.volume)
        valid = made.is_valid and len(made.solids()) == 1
        brow_overlap = sum(overlap(raw, shape) for shape in brow)
        bounds = raw.bounding_box()
        d = robot.config.dimensions
        framing_margin = min(
            d.head_width_m * 500 - max(abs(bounds.min.Y), abs(bounds.max.Y)),
            d.head_height_m * 500 - max(abs(bounds.min.Z), abs(bounds.max.Z)),
        )
        rows.append(
            dict(
                geometry_index=i,
                role=geometry.role,
                center_head_m=list(geometry.pos),
                original_volume_mm3=raw.volume,
                removed_volume_mm3=removed,
                minimum_opening_gap_mm=gap,
                exported_volume_error_mm3=volume_error,
                connected_valid_solid=valid,
                brow_interference_mm3=brow_overlap,
                head_front_projection_margin_mm=framing_margin,
                status="pass"
                if valid
                and removed < 0.02
                and volume_error < 0.02
                and gap >= minimum_gap_mm
                and brow_overlap < 0.02
                and framing_margin >= minimum_gap_mm
                else "fail",
            )
        )
    expected = 6 if robot.config.appearance == "scout" else 2
    return dict(
        status="pass"
        if len(rows) == expected and all(row["status"] == "pass" for row in rows)
        else "fail",
        scope="uncut eye allocations versus nominal camera removal opening; no physical lighting or tolerance qualification",
        minimum_required_gap_mm=minimum_gap_mm,
        expected_eye_parts=expected,
        eyes=rows,
    )
