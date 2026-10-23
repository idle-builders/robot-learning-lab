"""Camera cassette candidate in millimetres; purchased-part depth remains assumed."""

import numpy as np
from build123d import Box, Cylinder, Pos, Rot, fillet

from .vision import load_vision


def datum(robot):
    return np.array(robot.camera_position) * 1000


def mounting_centers(robot):
    x, y, z = datum(robot)
    return [(x - 14, y + side * 19, z - 6) for side in (-1, 1)]


def cassette_local():
    """Local origin at the allocated front optical point, +X looking out."""
    outer = Box(18, 30, 28.8)
    outer = Pos(-6, 0, -2.4) * fillet(outer.edges(), 1.2)
    # Clearance around the conservative whole-module allocation, not exact PCB CAD.
    outer -= Pos(-6.2, 0, -2.4) * Box(13.4, 26, 25)
    outer -= Pos(2, 0, 0) * Rot(0, 90, 0) * Cylinder(6.5, 6)
    # Four drawing-based holes through the assumed rear mounting plate.
    camera = load_vision().camera
    ox, oy = camera.optic_datum_drawing_mm
    for x, y in camera.mounting_holes_drawing_mm:
        outer -= Pos(-14, x - ox, y - oy) * Rot(0, 90, 0) * Cylinder(1.1, 6)
    # Initial flex exit reservation; bend radius/mating remain unverified.
    outer -= Pos(-9, 0, -16) * Box(5, 17, 4)
    for side in (-1, 1):
        ear = Pos(-12.5, side * 17.5, -6) * Box(3, 11, 8)
        ear -= Pos(-12.5, side * 19, -6) * Rot(0, 90, 0) * Cylinder(1.35, 6)
        outer += ear
    # The lug roots must not enter the reserved module corners.
    outer -= Pos(-6.2, 0, -2.4) * Box(13.4, 26, 25)
    # Open the lower rear plate around the existing pitch-bridge screw sweep.
    # This is between the camera's four mounting holes, outside its reservation.
    outer -= Pos(-14, 0, -14.5) * Box(4, 7, 8)
    if not outer.is_valid or len(outer.solids()) != 1:
        raise ValueError("camera cassette must be one valid connected part")
    return outer


def cassette(robot):
    if robot.config.camera.support_enabled:
        from .camera_support import combined_carrier, specification

        return Pos(*datum(robot)) * combined_carrier(
            specification(robot.config.camera.support_profile)
        )
    return Pos(*datum(robot)) * cassette_local()


def clearance(robot):
    # Removal clearance follows the cassette envelope and its attachment ears.
    x, y, z = datum(robot)
    opening = Pos(x - 6, y, z - 2.4) * Box(19, 31, 29.8)
    for side in (-1, 1):
        opening += Pos(x - 12.5, y + side * 17.5, z - 6) * Box(4, 12, 9)
    if robot.config.camera.support_enabled:
        from .camera_support import centers

        # Internal clearance for rear nuts and projecting screw tips.
        for cy, cz in centers():
            opening += Pos(x - 16, y + cy, z + cz) * Rot(0, 90, 0) * Cylinder(3, 7)
    return opening


def removal_clearance(robot):
    """Open the face ahead of the cassette so its lugs can slide out."""
    x, y, z = datum(robot)
    result = Pos(x + 9, y, z - 2.4) * Box(49, 31, 29.8)
    for side in (-1, 1):
        result += Pos(x + 3, y + side * 17.5, z - 6) * Box(35, 12, 9)
    return result


def fit_shell(shell, robot):
    shell -= clearance(robot)
    for x, y, z in mounting_centers(robot):
        # Rear-facing blind-pilot stock joined to the physical shell.
        boss = Pos(x - 4, y, z) * Rot(0, 90, 0) * Cylinder(3.8, 8)
        boss -= Pos(x - 2.5, y, z) * Rot(0, 90, 0) * Cylinder(1.05, 5)
        shell += boss
    if not shell.is_valid or len(shell.solids()) != 1:
        raise ValueError("camera mounting bosses must join the head shell")
    return shell


def screws(robot):
    result = []
    for x, y, z in mounting_centers(robot):
        # Provisional M2.5 x 8 self-tapping stack, 3 mm lug and 5 mm engagement.
        screw = Pos(x - 1, y, z) * Rot(0, 90, 0) * Cylinder(1.25, 8)
        screw += Pos(x + 4.2, y, z) * Rot(0, 90, 0) * Cylinder(2.25, 2.4)
        result.append(screw)
    return result


def interface(robot):
    result = dict(
        schema_version=1,
        owner="head",
        optical_datum_head_mm=datum(robot).tolist(),
        cassette_seats_head_mm=mounting_centers(robot),
        drawing_to_head=[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        camera_mount_holes_diameter_mm=2.2,
        shell_lug_clearance_diameter_mm=2.7,
        shell_pilot_diameter_mm=2.1,
        shell_fasteners="2 provisional M2.5 x 8 self-tapping screws",
        camera_mount_depth_status="assumed 12.4 mm behind optical front; no populated sample",
        not_run=[
            "camera fastener stack",
            "thread pullout",
            "connector mating and flex bend",
            "thermal",
            "tolerances",
        ],
    )
    if robot.config.camera.support_enabled:
        from .camera_support import interface as support_interface, specification

        spec = specification(robot.config.camera.support_profile)
        result["camera_mount_depth_status"] = spec.get(
            "depth_basis", "drawing dimension-chain candidate; unmeasured"
        )
        result["camera_support"] = support_interface(spec)
    return result
