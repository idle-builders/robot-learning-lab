"""Make static comparison views from the B-rep source and exported knee GLBs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightweight_ape_xl330_knee as coupon  # noqa: E402

OUT = coupon.OUT
WIDTH, HEIGHT = 460, 600
SCALE = .28  # mm per pixel; matched between reference and coupon
VIEWS = {"front": (0, -1, .22), "side": (1, 0, .17),
         "threequarter": (1, -1, .5)}
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
SMALL = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)


def as_mesh(shape, rgba):
    v, f = shape.tessellate(.5, .4)
    result = trimesh.Trimesh(np.array([(p.X,p.Y,p.Z) for p in v]),
                             np.asarray(f), process=False)
    result.visual.face_colors = rgba
    return result


def source_parts():
    parts = {p.label: p for p in bd.import_step(coupon.SOURCE).children}
    fixed = [as_mesh(parts[n], color) for n,color in (
        ("R_thigh_front", (223,215,193,255)),
        ("R_thigh_back", (223,215,193,255)),
        ("R_knee_cuff", (83,81,75,255)))]
    moving = [as_mesh(parts[n], (228,210,177,255)) for n in
              ("R_shin_front", "R_shin_back")]
    return fixed, moving


def projected(meshes, view):
    look = np.asarray(view, dtype=float); look /= np.linalg.norm(look)
    up = np.array([0.,0.,1.]); right = np.cross(up, look); right /= np.linalg.norm(right)
    up = np.cross(look, right)
    target = np.array([42., 13., 93.])
    polygons = []
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices) - target
        xy = np.stack((WIDTH/2 + vertices @ right / SCALE,
                       HEIGHT/2 - vertices @ up / SCALE), axis=1)
        depth = vertices @ look
        colors = mesh.visual.face_colors
        normals = mesh.face_normals
        for i, tri in enumerate(mesh.faces):
            pts = xy[tri]
            if np.all((pts[:,0] < -5) | (pts[:,0] > WIDTH+5)) or np.all((pts[:,1] < -5) | (pts[:,1] > HEIGHT+5)):
                continue
            shade = .68 + .32*max(0., float(normals[i] @ np.array([.35,-.55,.76])))
            c = tuple(int(max(0,min(255,v*shade))) for v in colors[i,:3])
            polygons.append((float(depth[tri].mean()), pts, c))
    polygons.sort(key=lambda row: row[0])
    image = Image.new("RGB", (WIDTH,HEIGHT), "#e6e2d9")
    draw = ImageDraw.Draw(image)
    for _, pts, color in polygons:
        draw.polygon([tuple(p) for p in pts], fill=color)
    return image


def main():
    report = json.loads((OUT/"report.json").read_text())
    fixed, moving = source_parts()
    for angle in (0,60,120):
        transform = trimesh.transformations.rotation_matrix(
            np.deg2rad(angle), (1,0,0), coupon.AXIS)
        original = fixed + [m.copy().apply_transform(transform) for m in moving]
        built = list(trimesh.load(OUT/f"knee-{angle:03d}.glb", force="scene").dump())
        pose = next(row for row in report["poses"] if row["flexion_deg"] == angle)
        for name, look in VIEWS.items():
            left = projected(original, look)
            right = projected(built, look)
            canvas = Image.new("RGB", (WIDTH*2+24, HEIGHT+80), "#f8f6f0")
            canvas.paste(left, (8,42)); canvas.paste(right, (WIDTH+16,42))
            draw = ImageDraw.Draw(canvas)
            draw.text((10,8), f"{angle}° {name} — exact source exterior", font=FONT, fill="#242924")
            draw.text((WIDTH+18,8), "XL330 coupon", font=FONT, fill="#242924")
            summary = (f"Source thigh/shin: {pose['original_thigh_vs_original_shin_overlap_mm3']:.1f} mm³ overlap"
                       f"   •   Coupon hard collision: {'YES' if not pose['hard_collision_pass'] else 'NO'}"
                       f"   •   1 mm shell")
            draw.text((10, HEIGHT+50), summary, font=SMALL, fill="#44504a")
            canvas.save(OUT/f"preview-{name}-{angle:03d}.png")
            print(OUT/f"preview-{name}-{angle:03d}.png", flush=True)


if __name__ == "__main__":
    main()
