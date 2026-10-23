"""Isolated hip-yaw relocation / paired-axis packaging study, not a buildable joint.

The driven yaw shaft is separately supported. Published motor *case* dimensions
are exact; gear pitch circles, bearing arrangement and 2XC horn positions are
concept dimensions pending vendor CAD and load rating. The source sketch shell
is used only to ask whether the fixed case corners fit its nominal interior.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

import sketch_shell as shell

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-hip-joint-coupon-20261002"
HEIGHTS = (350, 500, 600)
MASS_CASES = {350: 1.2, 500: 3.4, 600: 4.0}  # hypotheses, except ToddlerBot 500/3.4 is a size analogue
OFFSET_MM = {350: 20, 500: 30, 600: 40}  # illustrative COM-to-support offsets
COLORS = {
    "shell": [235, 224, 207, 105],
    "frame": [112, 128, 141, 255],
    "motor": [59, 99, 140, 255],
    "paired": [198, 119, 73, 255],
    "bearing": [221, 192, 87, 255],
    "shaft": [188, 204, 210, 255],
    "gear": [123, 190, 156, 255],
    "axis": [227, 79, 78, 255],
}


def corners(center: tuple[float, float, float], size: tuple[float, float, float]):
    return [tuple(center[i] + sign[i] * size[i] / 2 for i in range(3))
            for sign in product((-1, 1), repeat=3)]


def inside(point: tuple[float, float, float], names: tuple[str, ...], scale: float) -> bool:
    query = bd.Vector(*(value / scale for value in point))
    return any(shell.INTERIORS[name].is_inside(query) for name in names)


def case(name: str, center: tuple[float, float, float], size: tuple[float, float, float],
         candidates: tuple[str, ...], scale: float) -> dict:
    failed = [tuple(round(v, 2) for v in corner) for corner in corners(center, size)
              if not inside(corner, candidates, scale)]
    return {"name": name, "center_mm": list(center), "size_mm": list(size),
            "outside_corners": failed, "nominal_corner_fit": not failed,
            "interiors": candidates}


def overlaps(a: dict, b: dict, gap: float = 1.0) -> bool:
    return all(abs(a["center_mm"][i] - b["center_mm"][i]) <
               (a["size_mm"][i] + b["size_mm"][i]) / 2 + gap for i in range(3))


def screen(height: int) -> dict:
    s = height / 350
    entries = []
    for side, sign in (("left", 1), ("right", -1)):
        hip_y = sign * 40 * s
        yaw_motor_y = sign * (40 * s - 20)  # exactly 20 mm pitch-centre relocation
        entries.append(case(f"{side}_yaw_XL330", (0, yaw_motor_y, 206 * s),
                            (34, 20, 26), ("torso_shell",), s))
        entries.append(case(f"{side}_roll_pitch_2XC_envelope", (0, hip_y, 168 * s),
                            (36, 36, 46.5),
                            ("torso_shell", f"{side}_thigh_shell"), s))
    clashes = [[a["name"], b["name"]] for index, a in enumerate(entries)
               for b in entries[index + 1:] if overlaps(a, b)]
    moment = MASS_CASES[height] * 9.81 * OFFSET_MM[height] / 1000
    return {
        "height_mm": height, "mass_hypothesis_kg": MASS_CASES[height],
        "single_support_lateral_offset_hypothesis_mm": OFFSET_MM[height],
        "static_hip_roll_moment_Nm": round(moment, 3),
        "cases": entries, "case_pairs_with_less_than_1mm_clearance": clashes,
        "candidate_corner_fit": all(item["nominal_corner_fit"] for item in entries) and not clashes,
        "yaw_drive": "1:1 20 mm pitch-centre spur relocation; independent output shaft/bearings; gear tooth form and load rating undecided",
        "paired_drive": "2XC430-W250-T outer case envelope only; vendor horn locations, idler/bearing, cable exit, shaft travel, and moving-case sweep NOT checked",
        "meaning": "static axis/case packaging hypothesis only; not a claim of buildability, rated walking torque, shell clearance through motion, or learned policy transfer",
    }


def add(scene: trimesh.Scene, name: str, shape: bd.Shape, color: str):
    vertices, faces = shape.tessellate(0.2, 0.2)
    mesh = trimesh.Trimesh(vertices=np.array([[v.X, v.Y, v.Z] for v in vertices]),
                           faces=np.asarray(faces), process=False)
    mesh.visual.vertex_colors = COLORS[color]
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def cyl(radius: float, height: float, at: tuple[float, float, float], axis="z"):
    shape = bd.Cylinder(radius, height)
    if axis == "x":
        shape = bd.Rot(0, 90, 0) * shape
    elif axis == "y":
        shape = bd.Rot(90, 0, 0) * shape
    return bd.Pos(*at) * shape


def render(report: dict):
    s = report["height_mm"] / 350
    scene = trimesh.Scene()
    # The approved torso is ghosted as a contextual envelope, not modified.
    torso = next(part.shape for part in shell.PARTS if part.name == "torso_shell")
    add(scene, "approved_torso_skin_context", torso.scale(s), "shell")
    for item in report["cases"]:
        color = "paired" if "2XC" in item["name"] else "motor"
        add(scene, item["name"], bd.Pos(*item["center_mm"]) * bd.Box(*item["size_mm"]), color)
    for side, sign in (("left", 1), ("right", -1)):
        y = sign * 40 * s
        my = sign * (40 * s - 20)
        z = 206 * s
        # Steel output concept: torso frame carries opposed 8x16x5 bearings.
        add(scene, f"{side}_output_shaft_d8_concept", cyl(4, 52, (0, y, z - 1)), "shaft")
        for dz in (-7, 15):
            add(scene, f"{side}_bearing_OD16_z{dz}", cyl(8, 5, (0, y, z + dz)), "bearing")
        for suffix, cy in (("motor", my), ("output", y)):
            add(scene, f"{side}_{suffix}_pitch_circle_D20", cyl(10, 3, (0, cy, z - 17)), "gear")
        # Reference axes: actual 2XC horn coordinates must replace these markers.
        add(scene, f"{side}_roll_axis_reference", cyl(1.5, 54, (0, y, 174 * s), "x"), "axis")
        add(scene, f"{side}_pitch_axis_reference", cyl(1.5, 54, (0, y, 166 * s), "y"), "axis")
    scene.export(OUT / f"hip-{report['height_mm']}.glb")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    reports = [screen(height) for height in HEIGHTS]
    for report in reports:
        render(report)
        print(report["height_mm"], "mm:",
              sum(len(item["outside_corners"]) for item in report["cases"]),
              "outside case corners,", len(report["case_pairs_with_less_than_1mm_clearance"]),
              "case pairs <1mm")
    (OUT / "screen.json").write_text(json.dumps(reports, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("hip_joint_viewer.html").read_text())


if __name__ == "__main__":
    main()
