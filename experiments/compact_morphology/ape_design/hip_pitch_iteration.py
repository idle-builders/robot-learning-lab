"""Parent-mounted hip-pitch coupon carrying the saved knee/ankle/boot.

The independent hip roll/yaw carrier, servo thermal duty and compliant pelvic
socket remain open; this screens a physical pitch axis and the full distal CAD.
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
OUT = ROOT / "artifacts/ape-hip-pitch-iteration-20261003"
KNEE = ROOT / "artifacts/ape-knee-pitch-iteration-20261003"
ANKLE = ROOT / "artifacts/ape-ankle-pitch-iteration-20261003"
AXIS = (0, 41, 170)
POSES = (-60, -40, -20, 0, 20, 40)


def turned(shape: bd.Shape, deg: float):
    return bd.Pos(*AXIS) * bd.Rot(0, deg, 0) * bd.Pos(*(-v for v in AXIS)) * shape


def add(scene: trimesh.Scene, name: str, part: bd.Shape, color):
    verts, faces = part.tessellate(.34, .34)
    mesh = trimesh.Trimesh(np.array([[v.X, v.Y, v.Z] for v in verts]),
                           np.asarray(faces), process=False)
    mesh.visual.vertex_colors = color
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    torso = next(p.shape for p in shell.PARTS if p.name == "torso_shell")
    original_thigh = bd.import_step(KNEE / "relieved-thigh-and-bearing-housings.step")
    shin = bd.import_step(KNEE / "shin-and-knee-yoke.step")
    boot = bd.import_step(ANKLE / "boot-and-yoke.step")
    outsole = bd.import_step(ANKLE / "oval-outsole.step")

    # The official output lies 12.2 mm below case centre when flipped into
    # the pelvis. Two mirrored motors at y=+/-41 have a 47 mm case-to-case
    # gap at the inner faces, rather than occupying the same central bay.
    motor = bd.Pos(0, 41, 182.2) * bd.Box(24.72, 35, 45.22)
    motor_clearance = bd.Pos(0, 41, 182.2) * bd.Box(25.32, 35.6, 45.82)
    # An upper-thigh cut at z=165 clears the source case through the sampled
    # -60..+40 pitch poses. It is only 5 mm below the hip axis; cheek webs
    # bridge that edge to the co-axial supported output.
    thigh = original_thigh & (bd.Pos(0, 41, 65) * bd.Box(200, 200, 200))
    cheeks = []
    for y in (18, 64):
        cheek = (bd.Pos(0, y, 159) * bd.Box(14, 4, 22)).fuse(
            cylinder_y(10, 4, 0, y, 170))
        bridge_y = 17 if y == 18 else 65
        cheek = cheek.fuse(bd.Pos(0, bridge_y, 156) * bd.Box(16, 8, 15))
        cheeks.append(cheek)
    moving = thigh.fuse(*cheeks)
    moving = moving.fuse(*[cylinder_y(10, 3, 0, y, 170) for y in (21.5, 60.5)],
                         *[cylinder_y(4, 7, 0, y, 170) for y in (12.5, 69.5)])

    bearings = [cylinder_y(8, 5, 0, y, 170) - cylinder_y(4.05, 6, 0, y, 170)
                for y in (9.5, 72.5)]
    seats = [cylinder_y(11.5, 6, 0, y, 170) - cylinder_y(8.05, 7, 0, y, 170)
             for y in (9, 73)]
    support_webs = [bd.Pos(0, y, 183) * bd.Box(22, 18, 20) for y in (16, 66)]
    fixed = torso.fuse(*seats, *support_webs) - motor_clearance
    # The pelvic lip must be a real swept socket. A simple overlay would hide
    # geometry that penetrates the leg during deep pitch.
    for deg in range(-60, 41, 20):
        fixed = fixed - turned(moving, deg)
    fixed_solids = sorted(fixed.solids(), key=lambda x: x.volume, reverse=True)
    discarded_mm3 = round(sum(x.volume for x in fixed_solids[1:]), 2)
    fixed = fixed_solids[0]

    poses = []
    for deg in POSES:
        leg = turned(moving, deg)
        calf = turned(shin, deg)
        foot = turned(boot.fuse(outsole), deg)
        poses.append({
            "hip_pitch_deg": deg,
            "thigh_vs_fixed_mm3": round(volume(leg.intersect(fixed)), 2),
            "thigh_vs_motor_mm3": round(volume(leg.intersect(motor)), 2),
            "calf_vs_fixed_mm3": round(volume(calf.intersect(fixed)), 2),
        })
        scene = trimesh.Scene()
        add(scene, "V_pelvis_with_pitch_socket_and_bearing_seats", fixed, (232, 222, 205, 255))
        add(scene, "saved_knee_thigh_with_hip_yoke", leg, (242, 233, 215, 255))
        add(scene, "saved_ankle_shin", calf, (247, 236, 218, 255))
        add(scene, "saved_boot_outsole", foot, (227, 217, 195, 255))
        add(scene, "ST3250_parent_fixed_case", motor, (77, 112, 148, 180))
        for i, bearing in enumerate(bearings):
            add(scene, f"bearing_688_{i+1}", bearing, (92, 98, 102, 255))
        scene.export(OUT / f"hip-{deg:+04d}.glb")

    knee_report = json.loads((KNEE / "report.json").read_text())
    moving_mass_g = moving.volume * .00124 + knee_report["distal_module_mass_estimate_g_excluding_hardware_and_wires"] + 74.5
    report = {
        "stage": "35 cm parent-mounted hip pitch coupon; roll/yaw, balanced load, external skin and printed retention remain open",
        "source_servo": "Feetech ST3250-C001, 45.22 x 24.72 x 35 mm, 74.5 g, actual output offset 12.2 mm, 1.569 Nm rated at 12 V",
        "source_url": "https://www.feetechrc.com/en/562636.html",
        "bearing_source": "https://www.igus.com/product/xiros_radial_B180_ES_PA_C",
        "hip_pitch_axis_mm": AXIS,
        "motor_case_center_mm": [0, 41, 182.2],
        "motor_inside_original_torso_pct": round(100 * volume(motor.intersect(shell.INTERIORS["torso_shell"])) / motor.volume, 1),
        "bilateral_motor_case_gap_mm": 47,
        "motor_to_joint_map": "1:1 dual-horn direct pitch; zero and polarity to calibrate",
        "removed_thigh_above_z_mm": 165,
        "fixed_connected_solids": len(fixed.solids()),
        "moving_connected_solids": len(moving.solids()),
        "discarded_boolean_skin_chips_mm3": discarded_mm3,
        "distal_mass_screen_g_solid_PLA": round(moving_mass_g, 1),
        "static_single_support_examples": [{"mass_kg": mass,
                                             "offset_mm": 30,
                                             "moment_Nm": round(mass * 9.81 * .03, 3),
                                             "fraction_of_servo_rated": round(mass * 9.81 * .03 / 1.569, 2)}
                                            for mass in (2, 3)],
        "poses": poses,
        "open": ["hip roll/yaw carrier must support and move this case; parent-fixed pitch coupon alone is not a three-axis hip", "pelvic socket/gaiter must be visually acceptable under sketch-06 V lip", "bearing stub and horn fastening must have measured load path and hard stops", "squat pose requires coupled knee and ankle sweep, not only straight-leg hip pitch", "servo heat/duty and full-body mass need physical or sourced verification", "real learned-policy transfer"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("hip_pitch_viewer.html").read_text())
    bd.export_step(fixed, OUT / "V-pelvis-with-pitch-socket.step")
    bd.export_step(moving, OUT / "thigh-with-hip-pitch-yoke.step")
    return report


if __name__ == "__main__":
    result = main()
    print(result["fixed_connected_solids"], result["moving_connected_solids"], result["poses"])
