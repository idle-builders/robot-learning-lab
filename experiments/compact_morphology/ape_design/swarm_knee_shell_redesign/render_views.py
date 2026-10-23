"""Render the two opaque contour coupons in identical twelve-view grids."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
VIEWS = {"front": (0, -1, .18), "side": (1, 0, .10),
         "back": (0, 1, .18), "three-quarter": (1, -1, .35)}


def render(glb, view, size=(640,600)):
    scene=trimesh.load(glb,force="scene")
    direction=np.asarray(VIEWS[view],dtype=float)
    direction/=np.linalg.norm(direction)
    up=np.array((0.,0.,1.))
    right=np.cross(up,direction);right/=np.linalg.norm(right)
    up=np.cross(direction,right)
    center=np.array((65.,12.,150.))
    scale=1.8  # shared across all poses and both waist profiles
    image=Image.new("RGB",size,(237,236,231))
    draw=ImageDraw.Draw(image)
    light=np.array((.4,-.5,.75));light/=np.linalg.norm(light)
    polygons=[]
    for geom in scene.geometry.values():
        p=geom.vertices
        q=np.stack(((p-center)@right,(p-center)@up),axis=1)
        q=q*scale+np.array((size[0]/2,size[1]/2))
        q[:,1]=size[1]-q[:,1]
        depth=(p-center)@direction
        for index,face in enumerate(geom.faces):
            shade=.78+.20*abs(float(geom.face_normals[index]@light))
            color=tuple(np.clip(np.array((226,218,201))*shade,0,255).astype(int))
            polygons.append((float(np.mean(depth[face])),tuple(map(tuple,q[face])),color))
    for _,polygon,color in sorted(polygons,key=lambda x:x[0]):
        draw.polygon(polygon,fill=color)
    return image


def main():
    for label in ("waist_low", "waist_high"):
        sheet = Image.new("RGB", (640*4, 600*3), "white")
        for row, angle in enumerate((0, 60, 115)):
            for column, view in enumerate(VIEWS):
                img = render(HERE/f"{label}-{angle}.glb", view)
                ImageDraw.Draw(img).text((16, 14), f"{label} · {angle}° · {view}", fill=(30,36,37))
                img.save(HERE/f"{label}-{angle}-{view}.png")
                sheet.paste(img, (column*640, row*600))
        sheet.save(HERE/f"{label}-contact-sheet.png")


if __name__ == "__main__":
    main()
