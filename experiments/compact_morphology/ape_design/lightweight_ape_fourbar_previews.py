"""Paired orthographic preview sheets for the right-leg four-bar coupon."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"artifacts/lightweight-ape-fourbar-knee-20261005"
POSES=(0,60,115)
VIEWS={"front":(0,-1,0),"side":(1,0,0),"rear":(0,1,0),
       "threequarter":(1,-1,.27)}
W,H=960,1060


def geometry(view):
    eye=np.asarray(VIEWS[view],dtype=float);eye/=np.linalg.norm(eye)
    forward=-eye
    right=np.cross(forward,np.array((0.,0.,1.)));right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    scenes={p:trimesh.load(OUT/f"fourbar-{p:03d}.glb",force="scene") for p in POSES}
    vertices=np.concatenate([m.vertices for scene in scenes.values() for m in scene.geometry.values()])
    x=vertices@right;y=vertices@up
    pad=8
    xmin,xmax=float(x.min()-pad),float(x.max()+pad)
    ymin,ymax=float(y.min()-pad),float(y.max()+pad)
    scale=min((W-110)/(xmax-xmin),(H-150)/(ymax-ymin))
    return eye,right,up,scenes,((xmin+xmax)/2,(ymin+ymax)/2,scale)


def draw(view,pose,geo):
    eye,right,up,scenes,(cx,cy,scale)=geo
    image=Image.new("RGB",(W,H),"#202a30")
    d=ImageDraw.Draw(image)
    try:font=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",21)
    except OSError:font=ImageFont.load_default()
    triangles=[]
    light=np.array((.38,-.58,.72));light/=np.linalg.norm(light)
    for mesh in scenes[pose].geometry.values():
        verts=np.asarray(mesh.vertices)
        xy=np.column_stack((W/2+(verts@right-cx)*scale,
                            H/2-(verts@up-cy)*scale))
        depth=verts@eye
        colors=np.asarray(mesh.visual.vertex_colors)[:,:3]
        for face in mesh.faces:
            p=verts[face]
            normal=np.cross(p[1]-p[0],p[2]-p[0]);length=np.linalg.norm(normal)
            if length<1e-8:continue
            normal/=length
            brightness=.58+.36*abs(float(normal@light))
            rgb=tuple(int(np.clip(v*brightness,0,255)) for v in colors[face].mean(axis=0))
            triangles.append((float(depth[face].mean()),xy[face],rgb))
    triangles.sort(key=lambda item:item[0])
    for _,points,color in triangles:
        d.polygon([tuple(p) for p in points],fill=color)
    d.rounded_rectangle((16,16,575,82),radius=8,fill="#10191f")
    d.text((29,27),f"Source-shell four-bar knee  |  {pose}°  |  {view}",font=font,fill="#f2ede4")
    d.text((28,91),"Ivory: exact thigh/shin   ·   dark: first lapped cuff   ·   copper: links",fill="#d5d7d5")
    path=OUT/f"preview-{pose:03d}-{view}.png"
    image.save(path)
    print(path)


def main():
    for view in VIEWS:
        geo=geometry(view)
        for pose in POSES:draw(view,pose,geo)


if __name__=="__main__":main()
