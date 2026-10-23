"""Isolated knee architecture screen under sketch-06 shells.

The case size is a published ST3250 envelope. The equal-crank four-bar is an
explicit *hypothesis* inspired by ToddlerBot's proximal knee motor, not a copy
of its CAD. A BRep shell sweep exposes unresolved rigid-skin interference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import build123d as bd
import trimesh

import sketch_shell as shell
from hip_joint_coupon import add, case, cyl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-knee-joint-coupon-20261002"
HEIGHTS = (350, 500, 600)
ANGLES = (0, 30, 60, 90, 120)
ST3250_CASE_MM = (45.22, 24.72, 35.0)
ST3250_MASS_KG = .0745


def rotate_about_knee(shape: bd.Shape, scale: float, side: int, angle: int) -> bd.Shape:
    knee = (0, side * 41 * scale, 103 * scale)
    return bd.Pos(*knee) * bd.Rot(0, angle, 0) * bd.Pos(*(-v for v in knee)) * shape.scale(scale)


def overlap_mm3(a: bd.Shape, b: bd.Shape) -> float:
    result = a.intersect(b)
    if result is None:
        return 0.0
    return round(sum(s.volume for s in result), 1)


def screen(height: int) -> dict:
    scale = height / 350
    knee_z = 103 * scale
    hip_z = 170 * scale
    motor_prox_z = knee_z + 35 * scale
    direct = case("direct_ST3250", (0, 41 * scale, knee_z), ST3250_CASE_MM,
                  ("left_thigh_shell", "left_shin_shell"), scale)
    proximal = case("thigh_ST3250", (0, 41 * scale, motor_prox_z), ST3250_CASE_MM,
                    ("left_thigh_shell",), scale)
    inertia_delta = ST3250_MASS_KG * (((hip_z - knee_z) / 1000) ** 2 -
                                     ((hip_z - motor_prox_z) / 1000) ** 2)
    skin = []
    thigh = next(p.shape for p in shell.PARTS if p.name == "left_thigh_shell").scale(scale)
    shin = next(p.shape for p in shell.PARTS if p.name == "left_shin_shell")
    for angle in ANGLES:
        phi = math.radians(30 + angle)
        radius_m = 15 * scale / 1000
        rod_force_per_Nm = 1 / (radius_m * abs(math.sin(phi)))
        skin.append({
            "knee_flex_deg": angle,
            "motor_crank_deg": 30 + angle,
            "ideal_motor_to_knee_angle_ratio": 1,
            "rod_axial_force_N_per_Nm_knee_torque": round(rod_force_per_Nm, 1),
            "original_rigid_thigh_shin_overlap_mm3": overlap_mm3(
                thigh, rotate_about_knee(shin, scale, 1, angle)),
        })
    return {
        "height_mm": height,
        "source_case": "Feetech ST3250-C001 45.22x24.72x35 mm, 74.5 g, 1.569 Nm rated",
        "direct_case": direct,
        "proximal_case": proximal,
        "hypothetical_linkage": "equal 15mm-at-35cm crank radii, 35mm-at-35cm centre span, 30deg phase; true ToddlerBot pin coordinates not claimed",
        "estimated_reduction_in_motor_point_swing_inertia_about_hip_kgm2": round(inertia_delta, 7),
        "poses": skin,
        "warning": "case-corner and original rigid-skin sweep only; no motor horn location, pin/bearing rating, rod section, thermal duty, shell fix, or learned-policy outcome",
    }


def render(report: dict, angle: int):
    scale = report["height_mm"] / 350
    theta = math.radians(30 + angle)
    center_y = 41 * scale
    knee_z = 103 * scale
    link_span = 35 * scale
    crank_radius = 15 * scale
    motor_z = knee_z + link_span
    scene = trimesh.Scene()
    thigh = next(p.shape for p in shell.PARTS if p.name == "left_thigh_shell")
    shin = next(p.shape for p in shell.PARTS if p.name == "left_shin_shell")
    add(scene, "approved_thigh_skin", thigh.scale(scale), "shell")
    add(scene, "approved_shin_skin", rotate_about_knee(shin, scale, 1, angle), "shell")
    add(scene, "proximal_ST3250_case", bd.Pos(0, center_y, motor_z) * bd.Box(*ST3250_CASE_MM), "motor")
    add(scene, "supported_knee_shaft", cyl(4, 49 * scale, (0, center_y, knee_z), "y"), "shaft")
    add(scene, "knee_idler_bearing", cyl(8, 5, (0, center_y + 18 * scale, knee_z), "y"), "bearing")
    add(scene, "knee_drive_bearing", cyl(8, 5, (0, center_y - 18 * scale, knee_z), "y"), "bearing")
    for name, z in (("motor", motor_z), ("knee", knee_z)):
        mid = (crank_radius * math.sin(theta) / 2, center_y,
               z + crank_radius * math.cos(theta) / 2)
        crank = bd.Pos(*mid) * bd.Rot(0, 30 + angle, 0) * bd.Cylinder(2, crank_radius)
        add(scene, f"{name}_hypothesis_crank", crank, "gear")
    rod_x = crank_radius * math.sin(theta)
    rod_z = knee_z + link_span / 2 + crank_radius * math.cos(theta)
    add(scene, "parallel_link_hypothesis", bd.Pos(rod_x, center_y, rod_z) * bd.Cylinder(2, link_span), "gear")
    scene.export(OUT / f"knee-{report['height_mm']}-{angle}.glb")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    reports = [screen(height) for height in HEIGHTS]
    for report in reports:
        for angle in (0, 60, 120):
            render(report, angle)
        print(report["height_mm"], "mm: direct/proximal failed corners",
              len(report["direct_case"]["outside_corners"]),
              len(report["proximal_case"]["outside_corners"]),
              "shell overlap at 120deg",
              report["poses"][-1]["original_rigid_thigh_shin_overlap_mm3"], flush=True)
    (OUT / "screen.json").write_text(json.dumps(reports, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("knee_joint_viewer.html").read_text())


if __name__ == "__main__":
    main()
