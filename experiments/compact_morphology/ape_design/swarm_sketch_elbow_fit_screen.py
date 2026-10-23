"""Source-size XL330/32:64 elbow fit against the independent sketch shell.

This is intentionally an unmodified-shell rejection screen. It uses the
actual 350 mm Study BReps, not the earlier mechanical proxy shell, and reports
where source-size hardware or articulated skins require redesign.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import build123d as bd
from shapely.geometry import Point
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/sketch_ape_shell"))
from build import Study  # noqa: E402

OUT = ROOT / "artifacts/swarm-sketch-elbow-fit-20261004"
AXIS = (96, 2, 194.5)
MOTOR = (87, 3, 234)
DRIVER = (87, 22, 238)
OUTPUT = (96, 22, 194.5)
POSES = tuple(range(0, 116, 10)) + (115,)


def volume(s):
    return sum(p.volume for p in s) if isinstance(s, bd.ShapeList) else (s.volume if s else 0.0)


def cyl_y(r, depth, x, y, z):
    return bd.Pos(x, y, z) * bd.Rot(90, 0, 0) * bd.Cylinder(r, depth)


def flex(shape, deg):
    return bd.Pos(*AXIS) * bd.Rot(0, -deg, 0) * bd.Pos(*(-v for v in AXIS)) * shape


def belt():
    circles = ((DRIVER[0], DRIVER[2], 10.19), (OUTPUT[0], OUTPUT[2], 20.37))
    outer = unary_union([Point(x, z).buffer(r + .69, 32) for x, z, r in circles]).convex_hull
    inner = unary_union([Point(x, z).buffer(r - .69, 32) for x, z, r in circles]).convex_hull
    wire = lambda poly: bd.Wire.make_polygon([(x, 20, z) for x, z in poly.exterior.coords])
    return bd.extrude(bd.Face(wire(outer), [wire(inner)]), amount=4, dir=(0, 1, 0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    study = Study(height=350)
    study.build()
    parts = {p["name"]: p["shape"] for p in study.parts}
    upper = [parts[n] for n in ("R_upper_arm_front", "R_upper_arm_back")]
    fore = [parts[n] for n in ("R_forearm_front", "R_forearm_back",
                                 "R_palm_front", "R_palm_back")]
    cuff = parts["R_elbow_cuff"]
    motor = bd.Pos(*MOTOR) * bd.Box(20, 26, 34)
    pulley_input = cyl_y(10.19, 4, *DRIVER)
    pulley_output = cyl_y(20.37, 4, *OUTPUT)
    loop = belt()
    hardware = {"XL330_case": motor, "32T_driver": pulley_input,
                "64T_output": pulley_output, "4mm_belt_swept_envelope": loop}
    report = {
        "stage": "unmodified independent sketch-shell hardware and motion fit rejection screen",
        "shell_source": "experiments/sketch_ape_shell/build.py Study(height=350)",
        "vendor_motor": "ROBOTIS XL330-M288-T source-sized 20 x 26 x 34 mm, motor mass 18 g",
        "vendor_motor_url": "https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/",
        "elbow_axis_xyz_mm": AXIS, "motor_case_xyz_mm": MOTOR,
        "motor_output_xyz_mm": DRIVER,
        "belt_axis_center_distance_mm": round(((DRIVER[0]-AXIS[0])**2 +
                                                 (DRIVER[2]-AXIS[2])**2)**.5, 3),
        "hardware_vs_upper_shell_mm3": {
            name: round(sum(volume(shape.intersect(s)) for s in upper), 2)
            for name, shape in hardware.items()},
        "hardware_vs_static_forearm_shell_mm3": {
            name: round(sum(volume(shape.intersect(s)) for s in fore), 2)
            for name, shape in hardware.items()},
        "hardware_vs_cosmetic_elbow_cuff_mm3": {
            name: round(volume(shape.intersect(cuff)), 2)
            for name, shape in hardware.items()},
        "poses": [],
    }
    for deg in POSES:
        moved = [flex(s, deg) for s in fore]
        shell_clash = [a.intersect(b) for a in moved for b in upper]
        row = {"angle": deg,
               "forearm_vs_upper_shell_mm3": round(sum(volume(s) for s in shell_clash), 2),
               "forearm_vs_64T_output_mm3": round(sum(volume(a.intersect(pulley_output))
                   for a in moved), 2),
               "forearm_vs_belt_mm3": round(sum(volume(a.intersect(loop))
                   for a in moved), 2),
               "forearm_vs_original_cuff_mm3": round(sum(volume(a.intersect(cuff))
                   for a in moved), 2)}
        if deg == 115:
            row["shell_clash_bounds_mm"] = [
                {"min": [round(v, 2) for v in s.bounding_box().min],
                 "max": [round(v, 2) for v in s.bounding_box().max],
                 "volume_mm3": round(volume(s), 2)}
                for clash in shell_clash if clash
                for s in clash.solids() if s.volume > 1]
        report["poses"].append(row)
        print(row, flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "poses"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
