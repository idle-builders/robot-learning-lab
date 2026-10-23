"""Ankle/foot case and contact-footprint screen for sketch-06.

The retained ape MJCF uses one pitch ankle and a 90x60 mm sole at 35 cm.
This coupon exposes the difference between that sole and the visible CAD foot.
"""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import trimesh

import sketch_shell as shell
from hip_joint_coupon import add, case, cyl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-ankle-joint-coupon-20261002"
HEIGHTS = (350, 500, 600)
ANGLES = (-45, 0, 45)
ST3250_CASE_MM = (45.22, 24.72, 35.0)


def rotate_foot(shape: bd.Shape, scale: float, angle: int) -> bd.Shape:
    pivot = (0, 41 * scale, 27 * scale)
    return bd.Pos(*pivot) * bd.Rot(0, angle, 0) * bd.Pos(*(-v for v in pivot)) * shape.scale(scale)


def overlap_mm3(a: bd.Shape, b: bd.Shape) -> float:
    result = a.intersect(b)
    if result is None:
        return 0.0
    return round(sum(s.volume for s in result), 1)


def screen(height: int) -> dict:
    scale = height / 350
    ankle_z = 27 * scale
    direct = case("direct_ST3250", (0, 41 * scale, ankle_z), ST3250_CASE_MM,
                  ("left_foot_shell", "left_shin_shell"), scale)
    remote = case("shin_ST3250", (0, 41 * scale, 65 * scale), ST3250_CASE_MM,
                  ("left_shin_shell",), scale)
    shin = next(p.shape for p in shell.PARTS if p.name == "left_shin_shell").scale(scale)
    foot = next(p.shape for p in shell.PARTS if p.name == "left_foot_shell")
    poses = [{"ankle_pitch_deg": angle,
              "original_rigid_shin_foot_overlap_mm3": overlap_mm3(
                  shin, rotate_foot(foot, scale, angle))} for angle in ANGLES]
    # The gym's sole is pos x=14 mm, half-length 45 mm, half-width 30 mm.
    # Sketch shell section at z=10 mm: x centre 7 mm, diameter 78 mm.
    # This is a plan-view bound, not a shell/ground intersection proof.
    sole_front = (14 + 45) * scale
    sole_rear = (14 - 45) * scale
    shell_front = (7 + 78 / 2) * scale
    shell_rear = (7 - 78 / 2) * scale
    return {
        "height_mm": height,
        "source_motor": "Feetech ST3250-C001 case envelope; 1.569 Nm published rated torque",
        "gym_model": "ape_sourced_v3_elbows_small_head.xml: one ankle pitch per side, sole box x center 14mm, half-length 45mm, half-width 30mm at 35cm",
        "direct_case": direct,
        "remote_case": remote,
        "sole_x_rear_front_mm": [round(sole_rear, 1), round(sole_front, 1)],
        "visible_shell_x_rear_front_at_z10_mm": [round(shell_rear, 1), round(shell_front, 1)],
        "gym_sole_toe_beyond_shell_at_z10_mm": round(sole_front - shell_front, 1),
        "gym_sole_heel_beyond_shell_at_z10_mm": round(max(0, shell_rear - sole_rear), 1),
        "thin_outsole_candidate_thickness_mm": round(2.5 * scale, 1),
        "thin_outsole_candidate_plan_bounds_match_gym": True,
        "poses": poses,
        "ankle_roll_in_retained_gym": False,
        "warning": "static case-corner and original rigid-skin checks; remote 1:1 transmission, bearing, cable, toe contact, speed-dependent motor limit and optional ankle roll not engineered",
    }


def render(report: dict, mode: str, angle: int):
    scale = report["height_mm"] / 350
    scene = trimesh.Scene()
    shin = next(p.shape for p in shell.PARTS if p.name == "left_shin_shell")
    foot = next(p.shape for p in shell.PARTS if p.name == "left_foot_shell")
    add(scene, "approved_shin_skin", shin.scale(scale), "shell")
    add(scene, "approved_foot_skin", rotate_foot(foot, scale, angle), "shell")
    selected = report["direct_case" if mode == "direct" else "remote_case"]
    motor = bd.Pos(*selected["center_mm"]) * bd.Box(*ST3250_CASE_MM)
    if mode == "direct":
        motor = rotate_foot(motor, 1, angle) if scale == 1 else (
            bd.Pos(0, 41 * scale, 27 * scale) * bd.Rot(0, angle, 0) *
            bd.Pos(0, -41 * scale, -27 * scale) * motor)
    add(scene, f"{mode}_ST3250_case", motor, "motor")
    add(scene, "independently_supported_ankle_shaft_concept",
        cyl(4, 48 * scale, (0, 41 * scale, 27 * scale), "y"), "shaft")
    for offset in (-17, 17):
        add(scene, f"bearing_y{offset}",
            cyl(8, 5, (0, (41 + offset) * scale, 27 * scale), "y"), "bearing")
    # Transparentable gym contact sole; dimensions copied from the MJCF.
    sole = bd.Pos(14 * scale, 41 * scale, 4 * scale) * bd.Box(90 * scale, 60 * scale, 4 * scale)
    add(scene, "retained_gym_sole_projection", sole, "gear")
    # Appearance-first correction to the contact *outline*, with no claim
    # about material, sole pressure, collision gap or learned-policy transfer.
    outsole = bd.Pos(14 * scale, 41 * scale, 1.25 * scale) * bd.Box(
        90 * scale, 60 * scale, 2.5 * scale)
    outsole = bd.fillet(outsole.edges().filter_by(bd.Axis.Z), 7 * scale)
    # Roll the *upper* exposed toe overhang into the cover while preserving
    # the flat bottom contact bound at x=+59 mm (35 cm scale).
    toe_edges = [edge for edge in outsole.edges()
                 if edge.center().X > 58 * scale and
                 edge.center().Z > 2.4 * scale and
                 edge.length > 20 * scale]
    outsole = bd.fillet(toe_edges, 1.2 * scale)
    add(scene, "thin_gym_footprint_outsole_candidate", outsole, "frame")
    scene.export(OUT / f"ankle-{report['height_mm']}-{mode}-{angle}.glb")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    reports = [screen(height) for height in HEIGHTS]
    for report in reports:
        for mode in ("direct", "remote"):
            for angle in ANGLES:
                render(report, mode, angle)
        print(report["height_mm"], "mm: direct/remote failed corners",
              len(report["direct_case"]["outside_corners"]),
              len(report["remote_case"]["outside_corners"]),
              "gym toe protrusion mm", report["gym_sole_toe_beyond_shell_at_z10_mm"], flush=True)
    (OUT / "screen.json").write_text(json.dumps(reports, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("ankle_joint_viewer.html").read_text())


if __name__ == "__main__":
    main()
