"""Sketch-06 exterior study. No actuator or battery dimensions enter this model.

Run from the repository root:
  .venv/bin/python experiments/compact_morphology/ape_design/sketch_shell.py

Coordinates: x forward, y left, z up; dimensions are millimetres. The closed
2 mm skins are visual/volume studies, not printable joint or fastening designs.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-sketch-shell-v2-20261002"

COLORS = {
    "porcelain": [231, 226, 214, 255],
    "warm": [239, 228, 210, 255],
    "face": [244, 222, 194, 255],
    "shadow": [75, 74, 70, 255],
    "eye": [36, 39, 39, 255],
    "glint": [250, 247, 238, 255],
    "accent": [224, 112, 75, 255],
}


@dataclass
class Part:
    name: str
    region: str
    shape: bd.Shape
    color: str
    purpose: str


PARTS: list[Part] = []
INTERIORS: dict[str, bd.Shape] = {}
DEFAULT_WALL_MM = 2.0


def zloft(sections: list[tuple[float, float, float, float, float]]) -> bd.Shape:
    """z, x centre, y centre, x diameter, y diameter."""
    return bd.loft([
        bd.Plane(origin=(cx, cy, z)) * bd.Ellipse(dx / 2, dy / 2)
        for z, cx, cy, dx, dy in sections
    ])


def xloft(sections: list[tuple[float, float, float, float, float]]) -> bd.Shape:
    """x, y centre, z centre, y diameter, z diameter."""
    return bd.loft([
        bd.Plane(origin=(x, cy, cz), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
        * bd.Ellipse(dy / 2, dz / 2)
        for x, cy, cz, dy, dz in sections
    ])


def skin(sections: list[tuple[float, float, float, float, float]], wall: float | None = None) -> bd.Shape:
    wall = DEFAULT_WALL_MM if wall is None else wall
    outer = zloft(sections)
    return outer - inner_skin(sections, wall)


def inner_skin(sections: list[tuple[float, float, float, float, float]], wall: float | None = None) -> bd.Shape:
    wall = DEFAULT_WALL_MM if wall is None else wall
    return zloft([
        (z + (wall if i == 0 else -wall if i == len(sections) - 1 else 0),
         cx, cy, dx - 2 * wall, dy - 2 * wall)
        for i, (z, cx, cy, dx, dy) in enumerate(sections)
    ])


def add(name: str, region: str, shape: bd.Shape, color: str, purpose: str) -> None:
    if shape.volume <= 1:
        raise ValueError(f"Empty shape: {name}")
    PARTS.append(Part(name, region, shape, color, purpose))


def add_skin(name: str, region: str,
             sections: list[tuple[float, float, float, float, float]],
             color: str, purpose: str, wall: float | None = None) -> None:
    """Keep the shell and its nominal free interior from the same CAD sections."""
    add(name, region, skin(sections, wall), color, purpose)
    INTERIORS[name] = inner_skin(sections, wall)


def build(wall_mm: float = 2.0) -> None:
    global DEFAULT_WALL_MM
    if not 0.6 <= wall_mm <= 3.0:
        raise ValueError("Cosmetic wall study is limited to 0.6–3.0 mm")
    PARTS.clear()
    INTERIORS.clear()
    DEFAULT_WALL_MM = wall_mm
    # Sketch 06: head is about one quarter of the figure height. The shoulder
    # line falls below it; the back and belly stay full rather than conical.
    head_sections = [
        (270, -4, 0, 30, 36), (276, -4, 0, 54, 58),
        (285, -3, 0, 75, 77), (304, -3, 0, 87, 88),
        (324, -4, 0, 90, 89), (340, -6, 0, 75, 77),
        (348, -7, 0, 42, 44), (350, -7, 0, 12, 14),
    ]
    head = skin(head_sections)
    INTERIORS["head_shell"] = inner_skin(head_sections)
    face_region = bd.extrude(
        bd.Plane(origin=(10, 0, 314), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
        * bd.Ellipse(36, 28), amount=70,
    )
    add("head_shell", "head", head - face_region, "porcelain",
        "Closed domed head skin with an inlaid face opening")
    add("face_mask", "head", head & face_region, "face",
        "Flush face skin conforming to the head, not a raised disc")
    add("neck_shadow", "head", zloft([
        (269, -5, 0, 45, 49), (274, -5, 0, 53, 55),
        (279, -5, 0, 52, 54), (283, -5, 0, 46, 49),
    ]), "shadow", "Protected visual neck seam")
    add("muzzle", "head", xloft([
        (24, 0, 299, 46, 35), (31, 0, 297, 45, 32),
        (39, 0, 295, 42, 28), (46, 0, 294, 36, 23),
        (51, 0, 294, 25, 16), (53, 0, 294, 8, 6),
    ]), "face", "Low, narrow muzzle blended into the face")
    for side, sign in (("left", 1), ("right", -1)):
        add(f"{side}_eye", "face", bd.Pos(37, sign * 17, 319) * bd.Sphere(5.7),
            "eye", "Dark eye set into the face")
        add(f"{side}_eye_glint", "face", bd.Pos(42.2, sign * 18.2, 321.0) * bd.Sphere(1.2),
            "glint", "Eye highlight")

    # A single continuous shell keeps the V-shaped chest connected to its
    # broad rounded pelvis. The lower front sits ahead of the upper thighs.
    add_skin("torso_shell", "torso", [
        (127, 5, 0, 84, 90), (134, 5, 0, 90, 101),
        (148, 3, 0, 95, 112), (163, 0, 0, 98, 121),
        (175, -2, 0, 95, 125), (200, -3, 0, 97, 134),
        (225, -4, 0, 101, 145), (244, -4, 0, 101, 149),
        (260, -4, 0, 92, 135), (270, -5, 0, 72, 104),
        (277, -5, 0, 48, 57),
    ], "porcelain", "One continuous V chest and rounded pelvis")
    add("chest_light", "torso", bd.Pos(47.2, 0, 237) *
        bd.fillet(bd.Box(3.8, 4.8, 11).edges(), 1.7), "accent",
        "Single restrained status light")

    for side, sign in (("left", 1), ("right", -1)):
        # The moving cap reaches the torso contour, so the arm looks supported.
        # Narrow dark cuffs mark provisional, non-validated joint interfaces.
        add_skin(f"{side}_upper_arm_shell", "arms", [
            (197, 0, sign * 95, 36, 42),
            (202, 0, sign * 95, 44, 49),
            (218, -1, sign * 92, 53, 56),
            (237, -3, sign * 88, 58, 58),
            (252, -5, sign * 83, 55, 55),
            (263, -6, sign * 79, 42, 43),
            (268, -6, sign * 77, 20, 24),
        ], "porcelain", "Sloping deltoid continuing into the upper arm")
        add(f"{side}_elbow_shadow", "arms", zloft([
            (196, 1, sign * 96, 38, 42),
            (200, 1, sign * 96, 43, 46),
            (203, 1, sign * 96, 40, 43),
        ]), "shadow", "Thin protected elbow seam")
        add_skin(f"{side}_forearm_shell", "arms", [
            (122, 7, sign * 103, 34, 36),
            (128, 6, sign * 103, 44, 46),
            (145, 5, sign * 102, 52, 50),
            (167, 3, sign * 100, 50, 51),
            (190, 2, sign * 97, 44, 47),
            (199, 1, sign * 96, 37, 42),
            (202, 1, sign * 96, 29, 32),
        ], "warm", "Full forearm hanging near the knee")
        add(f"{side}_wrist_shadow", "arms", zloft([
            (120, 7, sign * 107, 32, 36),
            (124, 7, sign * 107, 38, 41),
            (127, 7, sign * 107, 34, 37),
        ]), "shadow", "Narrow wrist seam")
        add_skin(f"{side}_hand_shell", "hands", [
            (78, 13, sign * 108, 24, 30),
            (84, 13, sign * 108, 41, 43),
            (97, 11, sign * 108, 49, 46),
            (113, 9, sign * 108, 44, 44),
            (123, 7, sign * 107, 32, 35),
        ], "porcelain", "Simple smooth paw shell without sculpted digits", 1.7)

        # Hip caps disappear into the lower torso instead of sitting apart.
        add_skin(f"{side}_thigh_shell", "legs", [
            (99, -2, sign * 41, 48, 49),
            (108, -2, sign * 41, 57, 56),
            (128, -2, sign * 40, 64, 60),
            (150, -2, sign * 40, 64, 61),
            (166, -3, sign * 40, 56, 55),
            (175, -4, sign * 40, 39, 41),
        ], "porcelain", "Curved thigh nested under pelvis")
        add(f"{side}_knee_shadow", "legs", zloft([
            (99, -1, sign * 41, 47, 48),
            (105, -1, sign * 41, 51, 51),
            (109, -1, sign * 41, 47, 48),
        ]), "shadow", "Thin knee seam")
        add_skin(f"{side}_shin_shell", "legs", [
            (24, -2, sign * 41, 31, 36),
            (31, -2, sign * 41, 43, 46),
            (55, -1, sign * 41, 51, 54),
            (80, -1, sign * 41, 53, 54),
            (101, -1, sign * 41, 47, 48),
            (107, -1, sign * 41, 37, 40),
        ], "warm", "Soft taper from calf into protected ankle")
        add_skin(f"{side}_foot_shell", "feet", [
            (0.5, 5, sign * 41, 46, 49),
            (3, 6, sign * 41, 67, 64),
            (10, 7, sign * 41, 78, 70),
            (20, 5, sign * 41, 74, 67),
            (27, 1, sign * 41, 55, 57),
            (33, -2, sign * 41, 31, 39),
        ], "porcelain", "Short broad foot pad with a continuous forefoot")


def mesh(part: Part) -> trimesh.Trimesh:
    vertices, faces = part.shape.tessellate(0.07, 0.07)
    result = trimesh.Trimesh(
        vertices=np.array([[v.X, v.Y, v.Z] for v in vertices]),
        faces=np.asarray(faces), process=False,
    )
    result.merge_vertices()
    result.fix_normals()
    result.visual.vertex_colors = COLORS[part.color]
    return result


def export() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    step_dir = OUT / "parts-step"
    step_dir.mkdir(exist_ok=True)
    scene = trimesh.Scene()
    manifest = []
    for part in PARTS:
        bd.export_step(part.shape, str(step_dir / f"{part.name}.step"))
        scene.add_geometry(mesh(part), node_name=part.name, geom_name=part.name)
        box = part.shape.bounding_box()
        manifest.append({
            "name": part.name, "region": part.region, "color": part.color,
            "purpose": part.purpose,
            "volume_cm3": round(part.shape.volume / 1000, 2),
            "bounds_mm": [[round(v, 2) for v in box.min],
                          [round(v, 2) for v in box.max]],
            "step": f"parts-step/{part.name}.step",
        })
    scene.export(OUT / "sketch-06-shell.glb")
    bounds = scene.bounds
    summary = {
        "design_stage": "sketch-first exterior only",
        "source_sketch": "docs/design/ape-concepts/06-tumble-compact-shoulder.png",
        "height_mm": round(float(bounds[1, 2] - bounds[0, 2]), 1),
        "width_mm": round(float(bounds[1, 1] - bounds[0, 1]), 1),
        "depth_mm": round(float(bounds[1, 0] - bounds[0, 0]), 1),
        "part_count": len(PARTS), "parts": manifest,
        "caveat": "Provisional seams. See separate static case screen; no joint-sweep, full 22-axis packaging, structural, electrical or printability validation yet.",
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copyfile(Path(__file__).with_name("sketch_shell_viewer.html"), OUT / "index.html")
    print(f"Exported {len(PARTS)} parts: {summary['height_mm']} high, "
          f"{summary['width_mm']} wide, {summary['depth_mm']} deep -> {OUT}")


if __name__ == "__main__":
    build()
    export()
