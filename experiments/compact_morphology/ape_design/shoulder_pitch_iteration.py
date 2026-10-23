"""Recessed shoulder-pitch mechanism screen carrying the saved elbow/paw.

The motor and bearings are source-envelope placements. A coaxial coupling and
moving hub are CAD envelopes; spline retention, roll/yaw carrier and skin
continuity remain open. This advances the old two-box static shoulder study.
"""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

import sketch_shell as shell
from ankle_pitch_iteration import volume, cylinder_y

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-shoulder-pitch-iteration-20261003"
ELBOW = ROOT / "artifacts/ape-elbow-pitch-iteration-20261003"
AXIS = (-5, 79, 250)
POSES = (-90, -60, -30, 0, 30, 45)


def turned(shape: bd.Shape, deg: float):
    return bd.Pos(*AXIS) * bd.Rot(0, deg, 0) * bd.Pos(*(-v for v in AXIS)) * shape


def add(scene: trimesh.Scene, name: str, part: bd.Shape, color):
    verts, faces = part.tessellate(.3, .3)
    mesh = trimesh.Trimesh(np.array([[v.X, v.Y, v.Z] for v in verts]),
                           np.asarray(faces), process=False)
    mesh.visual.vertex_colors = color
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    torso = next(p.shape for p in shell.PARTS if p.name == "torso_shell")
    original_upper = bd.import_step(ELBOW / "upper-arm-with-elbow-sleeve.step")
    paw = bd.import_step(ELBOW / "paw-inner-with-elbow-cheek.step").fuse(
        bd.import_step(ELBOW / "paw-outer-with-elbow-cheek.step"))
    # True 12.2 mm output offset is along x. The motor is 20 mm farther
    # inboard than the first pitch screen, leaving 24 mm to the arm axis.
    motor = bd.Pos(-17.2, 37.5, 250) * bd.Box(45.22, 35, 24.72)
    motor_clearance = bd.Pos(-17.2, 37.5, 250) * bd.Box(45.82, 35.6, 25.32)
    moving_hub = cylinder_y(11, 4, -5, 85, 250)
    hub_web = bd.Pos(-5, 98, 250) * bd.Box(20, 26, 18)
    # Remove the source arm's inner skin locally before installing a hub;
    # otherwise the shell appears to pass through stationary bearings.
    upper = original_upper - cylinder_y(12, 27, -5, 66, 250)
    moving = upper.fuse(moving_hub, hub_web)
    output_shaft = cylinder_y(4, 37, -5, 73.5, 250)
    moving = moving.fuse(output_shaft)

    bearings = [cylinder_y(8, 5, -5, y, 250) - cylinder_y(4.05, 6, -5, y, 250)
                for y in (63, 73)]
    seats = [cylinder_y(11, 7, -5, y, 250) - cylinder_y(8.1, 8, -5, y, 250)
             for y in (63, 73)]
    support = bd.Pos(-5, 67, 259) * bd.Box(22, 27, 16)
    fixed = torso.fuse(*seats, support) - motor_clearance
    for deg in range(-90, 46, 15):
        fixed = fixed - turned(moving, deg)
    fixed_solids = sorted(fixed.solids(), key=lambda x: x.volume, reverse=True)
    discarded_mm3 = round(sum(x.volume for x in fixed_solids[1:]), 2)
    fixed = fixed_solids[0]

    poses = []
    for deg in POSES:
        arm = turned(moving, deg)
        hand = turned(paw, deg)
        poses.append({
            "shoulder_pitch_deg": deg,
            "upper_vs_fixed_mm3": round(volume(arm.intersect(fixed)), 2),
            "upper_vs_motor_mm3": round(volume(arm.intersect(motor)), 2),
            "paw_vs_fixed_mm3": round(volume(hand.intersect(fixed)), 2),
        })
        scene = trimesh.Scene()
        add(scene, "torso_with_pitch_socket_and_bearing_seats", fixed, (232, 222, 205, 255))
        add(scene, "saved_elbow_upper_arm_with_pitch_hub", arm, (242, 233, 215, 255))
        add(scene, "saved_paw_at_zero_elbow", hand, (247, 236, 218, 255))
        add(scene, "ST3250_recessed_case", motor, (77, 112, 148, 180))
        for i, bearing in enumerate(bearings):
            add(scene, f"bearing_688_{i+1}", bearing, (92, 98, 102, 255))
        scene.export(OUT / f"shoulder-{deg:+04d}.glb")

    upper_mass_g = moving.volume * .00124 + 18  # includes elbow motor; solid-PLA screen
    paw_report = json.loads((ELBOW / "report.json").read_text())
    distal_mass_g = upper_mass_g + paw_report["distal_paw_mass_estimate_g"]
    report = {
        "stage": "recessed shoulder pitch mechanism screen; roll/yaw carrier, enclosure and fasteners remain open",
        "source_servo": "Feetech ST3250-C001, 45.22 x 24.72 x 35 mm, 74.5 g, output 12.2 mm from case centre, 1.569 Nm rated at 12 V",
        "source_url": "https://www.feetechrc.com/en/562636.html",
        "bearing_source": "https://www.igus.com/product/xiros_radial_B180_ES_PA_C",
        "pitch_axis_mm": AXIS,
        "motor_case_center_mm": [-17.2, 37.5, 250],
        "motor_inside_torso_pct_before_socket": round(100 * volume(motor.intersect(shell.INTERIORS["torso_shell"])) / motor.volume, 1),
        "source_outer_case_face_y_mm": 55,
        "pitch_axis_y_mm": 79,
        "coaxial_coupler_span_mm": 24,
        "shaft_bearing_centres_y_mm": [63, 73],
        "motor_to_pitch_map": "1:1 coaxial spline-to-shaft envelope; coupling, fastener and zero calibration not designed",
        "fixed_connected_solids": len(fixed.solids()),
        "discarded_boolean_skin_chips_mm3": discarded_mm3,
        "distal_arm_mass_screen_g_solid_PLA": round(distal_mass_g, 1),
        "illustrative_gravity_moment_Nm_at_70mm_COM": round(distal_mass_g / 1000 * 9.81 * .07, 3),
        "rated_motor_torque_Nm": 1.569,
        "poses": poses,
        "open": ["source motor must be carried by shoulder roll/yaw module; world-fixed torso coupon cannot represent combined axes", "socket may break the smooth sloped deltoid and must receive independent aesthetic review", "coax coupler and opposed bearing retainers require actual load-rated design", "hard stops and cable loop", "servo duty and roll/yaw combined reach/crawl sweep", "retrain or replay learned policy after actual axis transform"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("shoulder_pitch_viewer.html").read_text())
    bd.export_step(fixed, OUT / "torso-pitch-socket.step")
    bd.export_step(moving, OUT / "upper-arm-pitch-hub.step")
    return report


if __name__ == "__main__":
    report = main()
    print(report["fixed_connected_solids"], report["poses"])
