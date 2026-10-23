"""Fast source-axis screen for the next inward ape joints.

This does not pretend the shoulder or hip is engineered: it carries the actual
saved distal CAD through pitch poses, compares true-offset ST3250 envelopes,
and writes the clash geometry that the next mechanism iteration must solve.
"""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

import sketch_shell as shell
from ankle_pitch_iteration import volume


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-proximal-pitch-screen-20261003"
ELBOW = ROOT / "artifacts/ape-elbow-pitch-iteration-20261003"
KNEE = ROOT / "artifacts/ape-knee-pitch-iteration-20261003"
ANKLE = ROOT / "artifacts/ape-ankle-pitch-iteration-20261003"


def rotated(part: bd.Shape, axis: tuple[float, float, float], deg: float) -> bd.Shape:
    return bd.Pos(*axis) * bd.Rot(0, deg, 0) * bd.Pos(*(-v for v in axis)) * part


def percent_inside(shape: bd.Shape, interior: bd.Shape) -> float:
    return round(100 * volume(shape.intersect(interior)) / shape.volume, 1)


def painted(scene, name, shape, color):
    verts, faces = shape.tessellate(.35, .35)
    mesh = trimesh.Trimesh(np.array([[v.X, v.Y, v.Z] for v in verts]),
                           np.asarray(faces), process=False)
    mesh.visual.vertex_colors = color
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    torso = next(p.shape for p in shell.PARTS if p.name == "torso_shell")
    # The initial case at y=57.5 intruded 2.0–2.8 cm3 into the upper arm.
    # Recessing it to y=37.5 removes that clash in the sampled pitch poses,
    # but requires a supported outboard shaft/coupler to reach the arm axis.
    shoulder_axis = (-5, 79, 250)
    shoulder_case = bd.Pos(-17.2, 37.5, 250) * bd.Box(45.22, 35, 24.72)
    upper = bd.import_step(ELBOW / "upper-arm-with-elbow-sleeve.step")
    lower = bd.import_step(ELBOW / "paw-inner-with-elbow-cheek.step").fuse(
        bd.import_step(ELBOW / "paw-outer-with-elbow-cheek.step"))

    # A thigh-carried motor puts its actual source output 12.2 mm above case
    # centre at (0,41,170). Hip roll/yaw carriers are NOT represented yet.
    hip_axis = (0, 41, 170)
    hip_case = bd.Pos(0, 41, 157.8) * bd.Box(24.72, 35, 45.22)
    thigh = bd.import_step(KNEE / "relieved-thigh-and-bearing-housings.step")
    shin = bd.import_step(KNEE / "shin-and-knee-yoke.step")
    boot = bd.import_step(ANKLE / "boot-and-yoke.step")

    shoulder_poses = []
    for deg in (-90, -60, -30, 0, 30, 45):
        a = rotated(upper, shoulder_axis, deg)
        b = rotated(lower, shoulder_axis, deg)
        shoulder_poses.append({"pitch_deg": deg,
                               "upper_vs_torso_mm3": round(volume(a.intersect(torso)), 1),
                               "paw_vs_torso_mm3": round(volume(b.intersect(torso)), 1),
                               "upper_vs_shoulder_case_mm3": round(volume(a.intersect(shoulder_case)), 1)})
        scene = trimesh.Scene()
        painted(scene, "torso_visual_shell", torso, (228, 220, 205, 150))
        painted(scene, "upper_arm_saved_elbow_module", a, (240, 232, 213, 255))
        painted(scene, "paw_saved_elbow_module", b, (248, 239, 223, 255))
        painted(scene, "candidate_ST3250_torso_case", shoulder_case, (69, 106, 144, 220))
        scene.export(OUT / f"shoulder-{deg:+04d}.glb")

    hip_poses = []
    for deg in (-60, -40, -20, 0, 20, 40):
        a = rotated(thigh, hip_axis, deg)
        b = rotated(shin, hip_axis, deg)
        c = rotated(boot, hip_axis, deg)
        case = rotated(hip_case, hip_axis, deg)
        hip_poses.append({"pitch_deg": deg,
                          "thigh_vs_torso_mm3": round(volume(a.intersect(torso)), 1),
                          "shin_vs_torso_mm3": round(volume(b.intersect(torso)), 1),
                          "case_vs_torso_mm3": round(volume(case.intersect(torso)), 1)})
        scene = trimesh.Scene()
        painted(scene, "torso_visual_shell", torso, (228, 220, 205, 150))
        painted(scene, "thigh_saved_knee_module", a, (240, 232, 213, 255))
        painted(scene, "shin_saved_ankle_module", b, (248, 239, 223, 255))
        painted(scene, "boot_saved_ankle_module", c, (231, 222, 205, 255))
        painted(scene, "candidate_ST3250_thigh_case", case, (69, 106, 144, 220))
        scene.export(OUT / f"hip-{deg:+04d}.glb")

    report = {
        "stage": "source-axis shoulder and hip pitch screen, not supported joint CAD",
        "source_motor": "Feetech ST3250-C001, 45.22 x 24.72 x 35 mm, output 12.2 mm from case centre, 74.5 g, 1.569 Nm rated at 12 V",
        "source_url": "https://www.feetechrc.com/en/562636.html",
        "shoulder_axis_mm": shoulder_axis,
        "shoulder_case_center_mm": (-17.2, 37.5, 250),
        "shoulder_source_case_outer_face_y_mm": 55,
        "shoulder_axis_y_mm": 79,
        "shoulder_unsupported_span_if_no_added_bearings_mm": 24,
        "shoulder_required_architecture": "supported coaxial 1:1 shaft/coupler from the recessed output to the pitch hub; the case must be carried by subsequent shoulder roll/yaw, not assumed fixed to the world",
        "shoulder_case_inside_torso_pct": percent_inside(shoulder_case, shell.INTERIORS["torso_shell"]),
        "shoulder_poses": shoulder_poses,
        "hip_axis_mm": hip_axis,
        "hip_case_center_mm": (0, 41, 157.8),
        "hip_bilateral_note": "Moving-thigh direct case is not a complete hip. Putting two parent-fixed cases at the same central pelvis location would make them overlap; lateral displacement, yaw/roll carrier and body-load support remain unresolved.",
        "hip_case_inside_original_thigh_pct": percent_inside(hip_case, shell.INTERIORS["left_thigh_shell"]),
        "hip_poses": hip_poses,
        "interpretation": "A motor case that fits in one link does not define the yoke, transverse bearing path, cable loop, stop, skin seam, paired roll/yaw axes, support torque duty or policy compatibility. Nonzero torso overlaps identify required socket relief and should not be silently Booleaned away.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("proximal_pitch_viewer.html").read_text())
    return report


if __name__ == "__main__":
    result = main()
    print("shoulder", result["shoulder_case_inside_torso_pct"], result["shoulder_poses"])
    print("hip", result["hip_case_inside_original_thigh_pct"], result["hip_poses"])
