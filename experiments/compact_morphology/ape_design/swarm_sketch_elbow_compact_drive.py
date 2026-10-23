"""16:32 belt elbow screen on the independent 350 mm sketch shell.

The 16T pulley is *not* bolted flat to the XL330's 12 mm bolt circle. A
source-sized 4-hole flange would step axially to a smaller pulley; this coupon
only reserves its envelope. Exact tooth, hub fasteners, bearings and rear cover
are still engineering gates. Dimensions are millimetres.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh
from shapely.geometry import Point
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/sketch_ape_shell"))
from build import Study  # noqa: E402

OUT = ROOT / "artifacts/swarm-sketch-elbow-16-32-20261004"
AXIS = (108, 2, 194.5)       # +12 mm outboard relative to the sculpted cuff center
CASE = (90, 1, 232.55)
DRIVER = (90, 26, 236.55)
OUTPUT = (108, 26, 194.5)
R1 = 16 / math.pi             # 16 teeth at 2 mm pitch / (2*pi)
R2 = 32 / math.pi
POSES = (0, 30, 60, 90, 115)


def volume(s):
    return sum(p.volume for p in s) if isinstance(s, bd.ShapeList) else (s.volume if s else 0.0)


def cyl_y(r, depth, x, y, z):
    return bd.Pos(x, y, z) * bd.Rot(90, 0, 0) * bd.Cylinder(r, depth)


def flex(shape, deg):
    return bd.Pos(*AXIS) * bd.Rot(0, -deg, 0) * bd.Pos(*(-v for v in AXIS)) * shape


def belt_envelope(clearance=0.0):
    centers = ((DRIVER[0], DRIVER[2], R1), (OUTPUT[0], OUTPUT[2], R2))
    outer = unary_union([Point(x, z).buffer(r + .76 + clearance, 32)
                         for x, z, r in centers]).convex_hull
    inner = unary_union([Point(x, z).buffer(r - .76 - clearance, 32)
                         for x, z, r in centers]).convex_hull
    wire = lambda poly: bd.Wire.make_polygon([(x, 24, z) for x, z in poly.exterior.coords])
    return bd.extrude(bd.Face(wire(outer), [wire(inner)]), amount=4,
                      dir=(0, 1, 0))


def mesh(shape, color):
    v, f = shape.tessellate(.22, .25)
    m = trimesh.Trimesh(np.array([[a.X, a.Y, a.Z] for a in v]),
                        np.asarray(f), process=False)
    m.merge_vertices()
    m.visual.vertex_colors = color
    return m


def add(scene, name, shape, color):
    scene.add_geometry(mesh(shape, color), geom_name=name, node_name=name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    study = Study(height=350)
    study.build()
    p = {s["name"]: s["shape"] for s in study.parts}
    upper = [p[n] for n in ("R_upper_arm_front", "R_upper_arm_back")]
    fore = [p[n] for n in ("R_forearm_front", "R_forearm_back",
                              "R_palm_front", "R_palm_back")]
    cuff = p["R_elbow_cuff"]
    motor = bd.Pos(*CASE) * bd.Box(20, 26, 34)
    motor_output_boss = cyl_y(3, 6, DRIVER[0], 16, DRIVER[2])
    input_flange = cyl_y(8, 2.5, DRIVER[0], 19.25, DRIVER[2])
    input_hub = cyl_y(3, 4, DRIVER[0], 22.5, DRIVER[2])
    input_pulley = cyl_y(R1, 4, *DRIVER)
    output_pulley = cyl_y(R2, 4, *OUTPUT)
    loop = belt_envelope()
    hardware = {"XL330_case": motor, "XL330_output_boss_envelope": motor_output_boss,
                "stepped_hub_flange": input_flange,
                "stepped_hub_neck": input_hub, "16T_driver_envelope": input_pulley,
                "32T_output_envelope": output_pulley,
                "Gates_140_2MGT_04_belt_envelope": loop}
    center = math.dist((DRIVER[0], DRIVER[2]), (OUTPUT[0], OUTPUT[2]))
    pitch = (2 * math.sqrt(center**2-(R2-R1)**2) + math.pi*(R1+R2) +
             2*(R2-R1)*math.asin((R2-R1)/center))
    report = {
        "stage": "unmodified independent sketch-shell compact-drive screen; not a cleared elbow",
        "shell_source": "experiments/sketch_ape_shell/build.py Study(height=350)",
        "manufacturer_sources": {
            "motor": "https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/",
            "belt_catalog": "https://www.gates.com/content/dam/gates/home/knowledge-center/resource-library/catalogs/power-transmission-catalog.pdf",
            "belt_design_manual": "https://www.gates.com/content/dam/documents-library/catalogs/light-power-and-precision-manual.pdf"},
        "motor_size_mm": [20, 26, 34], "motor_mass_g": 18,
        "motor_stall_Nm_5V_not_continuous": .52,
        "output_axis_mm": AXIS, "source_cuff_center_mm": [96, 2, 194.5],
        "outboard_axis_shift_mm": 12,
        "motor_case_center_mm": CASE, "motor_pulley_center_mm": DRIVER,
        "belt_center_distance_mm": round(center, 3),
        "belt_pitch_path_length_mm": round(pitch, 3),
        "catalog_belt_pitch_length_mm": 140,
        "needed_center_adjustment_mm_approx": round((140-pitch)/2, 3),
        "Gates_16T_100rpm_6mm_base_torque_Nm": 1.58,
        "Gates_4mm_width_factor": .6,
        "Gates_140mm_length_factor": .8,
        "corrected_belt_input_torque_Nm_at_100rpm": round(1.58*.6*.8, 3),
        "belt_rating_interpretation": "Catalog rating for a 16T small sprocket at 100 rpm; printed pulley teeth, hub, thermal servo limit and dynamic duty are unverified.",
        "motor_to_elbow_map": "16:32, elbow angle=motor angle/2 ideal; elbow torque=2x motor torque before losses",
        "motor_no_load_rpm_5V": 103,
        "elbow_no_load_rpm_ideal": 51.5,
        "hardware_vs_original_upper_shell_mm3": {
            name: round(sum(volume(h.intersect(s)) for s in upper), 2)
            for name, h in hardware.items()},
        "hardware_vs_original_forearm_shell_at_zero_mm3": {
            name: round(sum(volume(h.intersect(s)) for s in fore), 2)
            for name, h in hardware.items()},
        "hardware_vs_original_cuff_mm3": {
            name: round(volume(h.intersect(cuff)), 2)
            for name, h in hardware.items()},
        "poses": [],
    }
    for deg in range(0, 116, 5):
        moving = [flex(s, deg) for s in fore]
        row = {"angle": deg,
               "forearm_vs_original_upper_mm3": round(sum(volume(a.intersect(b))
                   for a in moving for b in upper), 2),
               "forearm_vs_32T_output_mm3": round(sum(volume(a.intersect(output_pulley))
                   for a in moving), 2),
               "forearm_vs_belt_mm3": round(sum(volume(a.intersect(loop))
                   for a in moving), 2),
               "forearm_vs_source_cuff_mm3": round(sum(volume(a.intersect(cuff))
                   for a in moving), 2)}
        report["poses"].append(row)
        print(row, flush=True)
        if deg in POSES:
            scene = trimesh.Scene()
            for i, s in enumerate(upper):
                add(scene, f"fixed_source_upper_{i}", s, (229, 222, 208, 255))
            for i, s in enumerate(moving):
                add(scene, f"moving_source_fore_palm_{i}", s, (241, 234, 218, 255))
            add(scene, "source_cosmetic_cuff_unmodified", cuff, (111, 111, 107, 160))
            for name, h in hardware.items():
                col = (67, 102, 143, 160) if name == "XL330_case" else (199, 127, 79, 180)
                add(scene, name, h, col)
            scene.export(OUT / f"elbow-{deg:03d}.glb")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("swarm_sketch_elbow_compact_drive_viewer.html").read_text())
    return report


if __name__ == "__main__":
    out = main()
    print("max source-skin collision", max(v["forearm_vs_original_upper_mm3"] for v in out["poses"]))
