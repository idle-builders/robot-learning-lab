"""Exact BRep screen of a tempting but inadequate diagonal joint-seam fix.

Trimming both original rigid covers symmetrically is a falsification experiment:
small cuts clear solid overlap but still produce zero geometric clearance in
deep flex. It is not an articulated skin design or a product exterior.
"""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import trimesh

import sketch_shell as shell
from hip_joint_coupon import add, cyl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-moving-skin-clearance-20261002"
SLANT_DEG = 15
MODULES = {
    "knee": ("left_thigh_shell", "left_shin_shell", (0, 41, 103), (0, 60, 120), (5, 20, 40)),
    "ankle": ("left_shin_shell", "left_foot_shell", (0, 41, 27), (-45, 0, 45), (5, 15, 25)),
    "elbow": ("left_upper_arm_shell", "left_forearm_shell", (0, 96, 200), (0, 60, 120), (5, 20, 40)),
}


def single(shape: bd.Shape | bd.ShapeList) -> bd.Shape:
    return bd.Compound(children=list(shape)) if isinstance(shape, bd.ShapeList) else shape


def diagonal_trim(shape: bd.Shape, pivot: tuple[int, int, int], cut: int,
                  upper: bool) -> bd.Shape:
    # A rotated half-space whose forward-facing edge slopes below the back.
    z_mid = (cut + 200) if upper else (-cut - 200)
    halfspace = bd.Pos(*pivot) * bd.Rot(0, SLANT_DEG, 0) * bd.Pos(0, 0, z_mid) * bd.Box(500, 500, 400)
    return single(shape.intersect(halfspace))


def move(shape: bd.Shape, pivot: tuple[int, int, int], angle: int) -> bd.Shape:
    return bd.Pos(*pivot) * bd.Rot(0, angle, 0) * bd.Pos(*(-v for v in pivot)) * shape


def run() -> list[dict]:
    shell.build()
    shapes = {p.name: p.shape for p in shell.PARTS}
    result = []
    for name, (upper_name, lower_name, pivot, angles, cuts) in MODULES.items():
        rows = []
        for cut in cuts:
            upper = diagonal_trim(shapes[upper_name], pivot, cut, True)
            lower = diagonal_trim(shapes[lower_name], pivot, cut, False)
            for angle in angles:
                moving_angle = -angle if name == "elbow" else angle
                moved = move(lower, pivot, moving_angle)
                crossing = upper.intersect(moved)
                intersect_volume = sum(x.volume for x in crossing) if crossing else 0
                distance = upper.distance_to(moved)
                row = {
                    "bilateral_trim_mm_at_35cm": cut,
                    "pose_deg": angle,
                    "hard_cover_overlap_mm3": round(intersect_volume, 2),
                    "minimum_exact_surface_clearance_mm": round(distance, 2),
                    "upper_cover_material_retained_percent": round(100*upper.volume/shapes[upper_name].volume, 1),
                    "lower_cover_material_retained_percent": round(100*lower.volume/shapes[lower_name].volume, 1),
                }
                rows.append(row)
                scene = trimesh.Scene()
                add(scene, "diagonally_trimmed_upper_hard_skin", upper, "shell")
                add(scene, "diagonally_trimmed_lower_hard_skin", moved, "paired")
                add(scene, "joint_axis_reference", cyl(1.5, 52, pivot, "y"), "axis")
                scene.export(OUT / f"skin-{name}-{cut}-{angle}.glb")
        result.append({
            "module": name, "pivot_mm_at_35cm": pivot,
            "upper_cover": upper_name, "lower_cover": lower_name,
            "diagonal_cut_plane_deg": SLANT_DEG,
            "rows": rows,
            "conclusion": "This simple oblique planar trim is a rejected final design. Zero overlap volume does not imply a running gap when exact minimum distance is zero. Deep flex may require a nested cap, shaped relief, gaiter or motion limit.",
        })
        print(name, [(cut, min(r["minimum_exact_surface_clearance_mm"] for r in rows
                               if r["bilateral_trim_mm_at_35cm"] == cut)) for cut in cuts], flush=True)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    (OUT / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("moving_skin_clearance_viewer.html").read_text())


if __name__ == "__main__":
    main()
