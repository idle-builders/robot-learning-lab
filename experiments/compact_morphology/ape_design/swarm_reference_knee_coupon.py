"""Sketch-shell knee coupon with a real-size direct knee motor.

This intentionally starts with the independently designed 350 mm shell, rather
than applying a cover to the old open-cage mechanism. Run from the repo root:

  .venv/bin/python experiments/compact_morphology/ape_design/swarm_reference_knee_coupon.py

Units are mm. The study uses X lateral, Y rearward, Z up. The imported aesthetic
shell is not an engineering reference for fasteners or joint limits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/swarm-reference-knee-20261004"
STUDY = ROOT / "artifacts/sketch-ape-shell/exports/350mm/ape_shell_assembly.step"
AXIS = (42, 13, 95)
ANGLES = tuple(range(0, 121, 10))
VIEW_ANGLES = (0, 30, 60, 90, 120)


def volume(shape) -> float:
    if shape is None:
        return 0.0
    return sum(s.volume for s in shape.solids())


def flex(shape, angle: int):
    return bd.Pos(*AXIS) * bd.Rot(angle, 0, 0) * bd.Pos(*(-x for x in AXIS)) * shape


def cyl_x(radius: float, x0: float, x1: float, y: float, z: float):
    return bd.Solid.make_cylinder(radius, x1-x0,
                                  bd.Plane(origin=(x0, y, z), z_dir=(1, 0, 0)))


def box_at(center, size):
    return bd.Pos(*center) * bd.Box(*size)


def elliptical_wire(z: float, cx: float, cy: float, rx: float, ry: float):
    return bd.Wire.make_ellipse(rx, ry, bd.Plane(origin=(cx, cy, z)))


def blended_lower_thigh():
    """Replace only the sketch skin below the existing 124 mm loft station.

    The lower thigh curves continuously into the upper source shell. Its extra
    fore-aft depth accommodates the real ST3250 case; the design cost is a
    fuller knee, which is visible in the comparison viewer.
    """
    stations = (
        (77.7, 42, 2, 21, 25),
        (82, 42, 2, 27.5, 32.5),
        (95, 42, 2, 28, 33),
        (110, 41, 3, 27.5, 31.5),
        (124, 39, 3, 24, 25),
        (129, 38, 3, 23.5, 24.5),
    )
    exterior = bd.Solid.make_loft([elliptical_wire(*row) for row in stations])
    interior = bd.Solid.make_loft([
        elliptical_wire(z+2 if i == 0 else z-2 if i == len(stations)-1 else z,
                        cx, cy, rx-2, ry-2)
        for i, (z, cx, cy, rx, ry) in enumerate(stations)
    ])
    return exterior - interior


def colored(scene, name, part, rgba):
    verts, faces = part.tessellate(.20, .22)
    mesh = trimesh.Trimesh(
        vertices=np.array([[p.X, p.Y, p.Z] for p in verts]),
        faces=np.asarray(faces), process=False,
    )
    mesh.visual.vertex_colors = rgba
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assembly = bd.import_step(STUDY)
    p = {part.label: part for part in assembly.children}
    study_thigh = p["R_thigh_front"].fuse(p["R_thigh_back"])
    # Keep the exact source upper-thigh surface and front/back seam; the lower
    # envelope is a single accessible moulded saddle in this coupon.
    upper_thigh = study_thigh & box_at((42, 3, 250), (180, 180, 252))  # Z >= 124
    thigh = upper_thigh.fuse(blended_lower_thigh())
    shin_raw = p["R_shin_front"].fuse(p["R_shin_back"])
    foot = p["R_foot_dorsal"].fuse(
        p["R_toe_1"], p["R_toe_2"], p["R_toe_3"], p["R_sole"])

    # ST3250 published 24.72 x 35 x 45.22 mm, 74.5 g, 1.569 Nm rated.
    # Rotate the long radial direction into Y, so only 24.72 mm lies in Z.
    # Its 12.2 mm output offset is carried exactly: case centre Y=0.8,
    # output Y=13. The case protrudes 4.4 mm below the original thigh shell.
    motor = box_at((42, .8, 95), (35, 45.22, 24.72))
    motor_keepout = box_at((42, .8, 95), (35.8, 46.02, 25.52))
    tray = box_at((42, .8, 95), (39.8, 50.02, 29.52)) - motor_keepout
    # The interior tray is joined to the thigh by opposed high webs. These
    # volumes are conceptual printed ribs; fastener holes are still absent.
    webs = [box_at((x, 13, 112), (5, 16, 31)) for x in (21, 63)]

    # The original shin is cut only 5 mm at its top, then bridged to the 95 mm
    # knee output by a lateral two-cheek structural yoke.
    shin = shin_raw & box_at((42, 3, 24), (180, 180, 108))  # top Z = 78
    bridge = box_at((42, 22, 74), (50, 8, 8))
    front_bridge = box_at((42, -18, 74), (48, 8, 8))
    cross_webs = [box_at((x, 2, 74), (6, 42, 8)) for x in (30, 54)]
    cheeks = [box_at((x, 13, 85), (4, 11, 22)) for x in (17.5, 66.5)]
    horns = [cyl_x(9, a, b, 13, 95) for a, b in ((15.5, 19.5), (64.5, 68.5))]
    moving = shin.fuse(bridge, front_bridge, *cross_webs, *cheeks, *horns)
    shaft = cyl_x(4, 11.5, 72.5, 13, 95)
    bearings = [cyl_x(8, a, a+5, 13, 95)-cyl_x(4.1, a-1, a+6, 13, 95)
                for a in (10.5, 68.5)]
    housings = [cyl_x(11.5, a, a+6, 13, 95)-cyl_x(8.1, a-1, a+7, 13, 95)
                for a in (10, 69)]
    bearing_webs = [box_at((x, 13, 113), (9, 13, 24)) for x in (15, 72)]

    fixed_raw = thigh.fuse(tray, *webs, *housings, *bearing_webs) - motor_keepout
    # Use an actual 10 degree BRep sweep to relieve the moving path. A soft
    # bridge can later close the aperture; zero-volume contact is not a gap.
    fixed = fixed_raw
    removed = []
    for angle in ANGLES:
        before = volume(fixed)
        moved = flex(moving, angle)
        f = flex(foot, angle)
        fixed = fixed - moved - f
        removed.append({"angle_deg": angle,
                        "fixed_material_removed_mm3": round(before-volume(fixed), 2)})
    fixed_parts = sorted(fixed.solids(), key=lambda s: s.volume, reverse=True)
    fixed_core = fixed_parts[0]

    rows = []
    for angle in ANGLES:
        moved = flex(moving, angle)
        f = flex(foot, angle)
        rows.append({
            "flex_deg": angle,
            "fixed_vs_moving_mm3": round(volume(fixed_core.intersect(moved)), 3),
            "fixed_vs_foot_mm3": round(volume(fixed_core.intersect(f)), 3),
            "motor_vs_moving_mm3": round(volume(motor.intersect(moved)), 3),
            "motor_vs_foot_mm3": round(volume(motor.intersect(f)), 3),
        })
        if angle not in VIEW_ANGLES:
            continue
        scene = trimesh.Scene()
        colored(scene, "sketch_shell_thigh_guard_and_motor_mount", fixed_core,
                (226, 219, 204, 255))
        colored(scene, "sketch_shell_shin_with_structural_yoke", moved,
                (238, 232, 217, 255))
        colored(scene, "sketch_shell_foot", f, (221, 211, 193, 255))
        colored(scene, "source_size_ST3250_case", motor, (88, 116, 140, 220))
        for i, bearing in enumerate(bearings):
            colored(scene, f"bearing_688_envelope_{i}", bearing, (93, 91, 88, 255))
        scene.export(OUT / f"knee-{angle:03d}.glb")

    report = {
        "stage": "aesthetic source-shell plus sideways source-size direct knee coupon",
        "source_study": str(STUDY.relative_to(ROOT)),
        "source_study_warning": "The 67-part source is an aesthetic skin study without mechanics.",
        "axis_mm_xyz": list(AXIS),
        "source_motor": "Feetech ST3250-C001, 24.72 x 35 x 45.22 mm, 74.5 g, 1.569 Nm rated at 12 V",
        "source_motor_url": "https://www.feetechrc.com/en/562636.html",
        "motor_placement": "35 mm across lateral X, 45.22 mm fore-aft Y, 24.72 mm vertical Z; 12.2 mm shaft offset along Y",
        "original_shell_knee_cuff": "discarded because it directly occupies the motor and rotating yoke envelope",
        "original_shin_z_max_mm": 83,
        "trimmed_shin_z_max_mm": 78,
        "knee_to_original_visual_cuff_offset_mm": 9,
        "motor_below_original_thigh_z_mm": 4.36,
        "motor_vs_unrelieved_study_thigh_mm3": round(volume(motor.intersect(study_thigh)), 2),
        "motor_vs_unrelieved_study_shin_mm3": round(volume(motor.intersect(shin_raw)), 2),
        "motor_vs_new_lower_thigh_skin_mm3": round(volume(motor.intersect(thigh)), 2),
        "motor_vs_fixed_screened_core_mm3": round(volume(motor.intersect(fixed_core)), 3),
        "fixed_solids_after_sweep": len(fixed_parts),
        "discarded_sweep_fragments_mm3": round(sum(s.volume for s in fixed_parts[1:]), 2),
        "moving_solids_after_yoke_fuse": len(moving.solids()),
        "sweep_removal": removed,
        "poses": rows,
        "unresolved": [
            "Large thigh-fixed guard is an initial rounded shape, not aesthetic approval.",
            "Motor mount ribs are envelopes without screws, shell bosses or assembly path.",
            "No enclosed flexible knee gaiter or routed cable strain/retention exists.",
            "688 bearings are envelope placeholders without load/life calculation.",
            "Booleans at 10 degree intervals have no positive clearance or continuous-motion proof.",
            "No torso/hip/other-leg collision, electronics, fatigue, impact or thermal duty check.",
        ],
    }
    bd.export_step(fixed_core, OUT / "sketch-derived-fixed-thigh-and-guard.step")
    bd.export_step(moving, OUT / "sketch-derived-moving-shin-and-yoke.step")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(
        Path(__file__).with_name("swarm_reference_knee_viewer.html").read_text())
    print(json.dumps({k: report[k] for k in (
        "fixed_solids_after_sweep", "discarded_sweep_fragments_mm3",
        "moving_solids_after_yoke_fuse", "poses")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
