"""Opaque orthographic contact sheet for the bounded right-hip coupon."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
ART = ROOT / "artifacts/hip-v-hood-20261006"
HERE = Path(__file__).resolve().parent
VIEWS = {"front": (0, -1, .18), "side": (1, 0, .10),
         "back": (0, 1, .18), "three-quarter": (1, -1, .35)}
POSES = ("stand", "yaw_out", "stride", "squat")


def render(glb, view, size=(580, 600)):
    scene = trimesh.load(glb, force="scene")
    direction = np.asarray(VIEWS[view], dtype=float)
    direction /= np.linalg.norm(direction)
    up = np.array((0., 0., 1.))
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    all_vertices = np.concatenate([g.vertices for g in scene.geometry.values()])
    center = np.array((30., 0., 290.))
    proj = np.stack(((all_vertices-center) @ right, (all_vertices-center) @ up), axis=1)
    extent = max(np.ptp(proj[:, 0]), np.ptp(proj[:, 1]), 215)
    scale = min(size)*.83/extent
    img = Image.new("RGB", size, (237, 236, 231))
    draw = ImageDraw.Draw(img)
    polygons = []
    light = np.array((.4, -.5, .75)); light /= np.linalg.norm(light)
    for name, geom in scene.geometry.items():
        p = geom.vertices
        q = np.stack(((p-center) @ right, (p-center) @ up), axis=1)
        q = q*scale + np.array((size[0]/2, size[1]/2))
        q[:, 1] = size[1]-q[:, 1]
        depth = (p-center) @ direction
        normals = geom.face_normals
        for face_index, face in enumerate(geom.faces):
            shade = .78 + .20*abs(float(normals[face_index] @ light))
            base = np.array((226, 218, 201))
            polygons.append((float(np.mean(depth[face])),
                             tuple(map(tuple, q[face])),
                             tuple(np.clip(base*shade, 0, 255).astype(int))))
    for _, poly, color in sorted(polygons, key=lambda x: x[0]):
        draw.polygon(poly, fill=color)
    return img


def main():
    sheet = Image.new("RGB", (580*len(VIEWS), 600*len(POSES)), "white")
    for i, pose in enumerate(POSES):
        for j, view in enumerate(VIEWS):
            img = render(ART/f"opaque-{pose}.glb", view)
            ImageDraw.Draw(img).text((16, 14), f"{pose} · {view}", fill=(30, 36, 37))
            sheet.paste(img, (j*580, i*600))
    sheet.save(HERE/"hip-v-hood-contact-sheet.png")


if __name__ == "__main__":
    main()
