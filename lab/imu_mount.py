"""Two-post LSM6DSO board mount on the left torso wall; millimetre CAD interface.

Board STEP +X/+Y/+Z map to torso +X/+Z/-Y. Components face inward.
Retention/material/process remain unqualified. Recipe shares the sourced M2.5
insert/screw stack with the buck board, but owns its two-hole geometry.
"""

import numpy as np
from build123d import Cylinder, Pos, Rot
from .logic_mount import (
    POST_RADIUS_MM,
    INSERT_DIAMETER_MM,
    INSERT_LENGTH_MM,
    PILOT_DIAMETER_MM,
    PILOT_DEPTH_MM,
    SCREW_LENGTH_MM,
    SCREW_HEAD_DIAMETER_MM,
    SCREW_HEAD_HEIGHT_MM,
    PCB_THICKNESS_MM,
)

HOLES_PCB_MM = [[3, 17], [17, 17]]
BOARD_ROTATION = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]


def datum(d):
    return [d.imu_board_x_m * 1000, d.imu_board_y_m * 1000, d.imu_board_z_m * 1000]


def placement(d):
    return Pos(*datum(d)) * Rot(90, 0, 0) * Pos(-10, 10, 0)


def centers(d):
    x, y, z = datum(d)
    return [[x + px - 10, y, z + 10 - py] for px, py in HOLES_PCB_MM]


def validate_board(contract):
    if (
        contract["outline_mm"] != [20, 20]
        or sorted(contract["mounting_holes_mm"]) != sorted(HOLES_PCB_MM)
        or not np.isclose(contract["mounting_hole_diameter_mm"], 2.7)
        or not np.isclose(contract["thickness_mm"], PCB_THICKNESS_MM)
        or contract["step_frame"]
        != {
            "xy_from_pcb": [[1, 0], [0, -1]],
            "datum": "substrate bottom z=0; components toward positive z",
        }
    ):
        raise ValueError("compiled board differs from the two-post IMU interface")

    # The shared sensor site assumes U1 at the board centre and footprint 0°.
    # Reject a moved/rotated package rather than silently retaining that site.
    for pin, xy in {
        "U1.1": [8.8375, 9.25],
        "U1.4": [8.8375, 10.75],
        "U1.8": [11.1625, 10.75],
        "U1.11": [11.1625, 9.25],
    }.items():
        if not np.allclose(contract["pads"][pin]["xy_mm"], xy, atol=1e-6, rtol=0):
            raise ValueError(
                "sensor package moved or rotated relative to the mounting frame"
            )


def cylinder_y(x, y, z, radius, length):
    return Pos(x, y, z) * Rot(90, 0, 0) * Cylinder(radius, length)


def fit_front_shell(front, outer, d):
    if not d.imu_mount_enabled:
        return front
    offset = Pos(0, 0, -d.torso_height_m * 500)
    end = d.torso_width_m * 500
    for x, y, z in centers(d):
        stock = cylinder_y(x, (y + end) / 2, z, POST_RADIUS_MM, end - y)
        front += (offset * stock) & outer
        pilot = cylinder_y(
            x,
            y + PILOT_DEPTH_MM / 2 - 0.01,
            z,
            PILOT_DIAMETER_MM / 2,
            PILOT_DEPTH_MM + 0.02,
        )
        front -= offset * pilot
    if not front.is_valid or len(front.solids()) != 1:
        raise ValueError("IMU posts must join the front shell as one valid solid")
    return front


def hardware(d):
    screws, inserts = [], []
    for x, y, z in centers(d):
        seat = y - PCB_THICKNESS_MM
        screw = cylinder_y(x, seat + SCREW_LENGTH_MM / 2, z, 1.25, SCREW_LENGTH_MM)
        screw += cylinder_y(
            x,
            seat - SCREW_HEAD_HEIGHT_MM / 2,
            z,
            SCREW_HEAD_DIAMETER_MM / 2,
            SCREW_HEAD_HEIGHT_MM,
        )
        insert = cylinder_y(
            x, y + INSERT_LENGTH_MM / 2, z, INSERT_DIAMETER_MM / 2, INSERT_LENGTH_MM
        )
        insert -= cylinder_y(x, y + INSERT_LENGTH_MM / 2, z, 1.25, INSERT_LENGTH_MM + 1)
        screws.append(screw)
        inserts.append(insert)
    return screws, inserts
