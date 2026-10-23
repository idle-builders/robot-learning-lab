"""Render matched opaque orthographic views from the generated GLB shells.

Pillow's small triangle painter is deliberately deterministic and requires no
OpenGL/CDN.  These images judge silhouette only; exact clearances stay in the
BRep report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/"artifacts/rated-leg-skin-v2-20261005"
SIZE=800
BG=(220,217,207)
LIGHT=np.array([.35,-.70,.63]);LIGHT/=np.linalg.norm(LIGHT)


def render(mesh, view, height):
    vertices=np.asarray(mesh.vertices)
    faces=np.asarray(mesh.faces)
    normals=np.asarray(mesh.face_normals)
    tri=vertices[faces]
    centre=height/500
    if view=="front":
        u=tri[:,:,0]-62*centre;depth=tri[:,:,1].mean(axis=1)
        face_front=-normals[:,1]
        order=np.argsort(-depth)
    elif view=="back":
        u=62*centre-tri[:,:,0];depth=tri[:,:,1].mean(axis=1)
        face_front=normals[:,1]
        order=np.argsort(depth)
    else:
        u=-(tri[:,:,1]-0);depth=tri[:,:,0].mean(axis=1)
        face_front=normals[:,0]
        order=np.argsort(depth)
    v=tri[:,:,2]-140
    # Fixed physical scale across 50/60 cm; source/revised share exactly the
    # same framing, so enlargement remains visible.
    xy=np.stack((SIZE/2+u*2.65,SIZE/2-v*2.65),axis=2)
    colors=np.asarray(mesh.visual.face_colors)[:,:3]
    canvas=Image.new("RGB",(SIZE,SIZE),BG)
    draw=ImageDraw.Draw(canvas)
    for i in order:
        if face_front[i]<=0.01:continue
        shade=.68+.32*max(0,float(normals[i]@LIGHT))
        color=tuple(int(np.clip(x*shade,0,255)) for x in colors[i])
        draw.polygon([tuple(p) for p in xy[i]],fill=color)
    return canvas


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--heights",type=int,nargs="+",choices=(500,600),default=(500,600))
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for height in args.heights:
        panels=[]
        for variant in ("source","revised"):
            path=OUT/f"{variant}-opaque-{height}.glb"
            scene=trimesh.load(path)
            mesh=scene.to_geometry()
            for view in ("front","side","back"):
                image=render(mesh,view,height)
                image.save(OUT/f"{variant}-{height}-{view}.png")
                panels.append((variant,view,image))
        sheet=Image.new("RGB",(SIZE*3,SIZE*2+68),(248,246,240))
        draw=ImageDraw.Draw(sheet)
        for index,(variant,view,image) in enumerate(panels):
            x=(index%3)*SIZE;y=(index//3)*(SIZE+34)+34
            draw.text((x+14,y-25),f"{height} mm · {variant} · {view}",fill=(38,52,46))
            sheet.paste(image,(x,y))
        sheet.save(OUT/f"matched-opaque-{height}.png")
        print(OUT/f"matched-opaque-{height}.png")


if __name__=="__main__":main()
