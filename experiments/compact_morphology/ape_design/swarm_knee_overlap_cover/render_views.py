"""Render the overlap trial GLBs as opaque orthographic inspection sheets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
VIEWS = {
    "front": (0, -1, 0.18),
    "side": (1, 0, 0.10),
    "back": (0, 1, 0.18),
    "three-quarter": (1, -1, 0.35),
}
ANG = (0, 60, 115)


def render(glb, view, size=(640, 600)):
    scene = trimesh.load(glb, force="scene")
    direction = np.asarray(VIEWS[view], dtype=float)
    direction /= np.linalg.norm(direction)
    up = np.array((0., 0., 1.))
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    all_vertices = np.concatenate([g.vertices for g in scene.geometry.values()])
    center = np.array((65., 12., 150.))
    proj = np.stack(((all_vertices-center)@right, (all_vertices-center)@up), axis=1)
    extent = max(np.ptp(proj[:, 0]), np.ptp(proj[:, 1]), 125)
    scale = min(size)*0.81/extent
    img = Image.new("RGB", size, (237, 236, 231))
    draw = ImageDraw.Draw(img)
    polygons = []
    light = np.array((0.4, -0.5, 0.75)); light /= np.linalg.norm(light)
    for name, geom in scene.geometry.items():
        p = geom.vertices
        q = np.stack(((p-center)@right, (p-center)@up), axis=1)
        q = q*scale + np.array((size[0]/2, size[1]/2))
        q[:, 1] = size[1]-q[:, 1]
        depth = (p-center)@direction
        is_cover = "gore" in name
        normals = geom.face_normals
        for face_index, face in enumerate(geom.faces):
            normal = normals[face_index]
            shade = 0.78 + 0.20*abs(float(normal@light))
            if is_cover:
                base = np.array((48, 56, 58))
            else:
                base = np.array((226, 218, 201))
            color = tuple(np.clip(base*shade, 0, 255).astype(int))
            poly = tuple(map(tuple, q[face]))
            polygons.append((float(np.mean(depth[face])), poly, color, is_cover))
    for _, poly, color, is_cover in sorted(polygons, key=lambda x:x[0]):
        draw.polygon(poly, fill=color)
        if is_cover:
            draw.line((*poly,poly[0]), fill=(38, 44, 46), width=1)
    return img


def main():
    tiles = []
    for angle in ANG:
        row = []
        for view in VIEWS:
            img = render(HERE/f"overlap-{angle}.glb", view)
            ImageDraw.Draw(img).text((16, 14), f"{angle}° · {view}", fill=(30,36,37))
            img.save(HERE/f"overlap-{angle}-{view}.png")
            row.append(img)
        tiles.append(row)
    sheet = Image.new("RGB", (640*4, 600*3), "white")
    for i,row in enumerate(tiles):
        for j,img in enumerate(row):
            sheet.paste(img,(j*640,i*600))
    sheet.save(HERE/"overlap-contact-sheet.png")


if __name__ == "__main__":
    main()
