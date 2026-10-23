"""Draw orthographic review sheets from the coupon's exported GLB meshes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/lightweight-ape-ft3332-knee-20261005"
CAMERAS = {
    "front": (0,-1,0),
    "side": (1,0,0),
    "rear": (0,1,0),
    "three-quarter": (1,-1,.28),
}
WIDTH,HEIGHT = 920,1100


def draw_pose(angle,view,eye):
    scene=trimesh.load(OUT/f"pose-{angle:03d}/ft3332_knee.glb",force="scene")
    meshes=list(scene.geometry.values())
    eye=np.asarray(eye,dtype=float); eye/=np.linalg.norm(eye)
    forward=-eye
    right=np.cross(forward,np.array([0.,0.,1.])); right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    all_points=np.concatenate([m.vertices for m in meshes])
    px=all_points@right; py=all_points@up
    # Keep one scale and origin across all three poses for each view.
    if view == "front":
        xmin,xmax=-25,105
    elif view == "rear":
        xmin,xmax=-105,25
    else:
        xmin,xmax=-85,105
    ymin,ymax=0,180
    span=max((xmax-xmin)/(WIDTH-100),(ymax-ymin)/(HEIGHT-140))
    scale=1/span
    cx=(xmin+xmax)/2;cy=(ymin+ymax)/2
    def xy(points):
        return np.column_stack([WIDTH/2+(points@right-cx)*scale,
                                HEIGHT/2-(points@up-cy)*scale])
    im=Image.new('RGB',(WIDTH,HEIGHT),'#202a30')
    dr=ImageDraw.Draw(im)
    try:font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',20)
    except OSError:font=ImageFont.load_default()
    triangles=[]
    light=np.array([.35,-.65,.68]);light/=np.linalg.norm(light)
    for mesh in meshes:
        verts=np.asarray(mesh.vertices)
        proj=xy(verts)
        depths=verts@eye
        cols=np.asarray(mesh.visual.vertex_colors)[:,:3]
        for face in mesh.faces:
            p=verts[face]
            n=np.cross(p[1]-p[0],p[2]-p[0])
            ln=np.linalg.norm(n)
            if ln<1e-8:continue
            n/=ln
            brightness=.59+.33*abs(np.dot(n,light))
            rgb=tuple(int(np.clip(x*brightness,0,255)) for x in cols[face].mean(axis=0))
            triangles.append((float(depths[face].mean()),proj[face],rgb))
    triangles.sort(key=lambda a:a[0])
    for _,pts,rgb in triangles:
        dr.polygon([tuple(p) for p in pts],fill=rgb)
    dr.rounded_rectangle((16,16,450,75),radius=8,fill='#111a20')
    dr.text((30,28),f'FT3332 knee | {angle}° | {view}',font=font,fill='#f0ede4')
    dr.text((30,81),'Exact 1 mm source-shell right leg; grey/copper parts are provisional',
            fill='#d1d3d0')
    path=OUT/f'preview-{angle:03d}-{view}.png';im.save(path)
    print(path)


def main():
    for angle in (0,60,120):
        for view,eye in CAMERAS.items():draw_pose(angle,view,eye)


if __name__=='__main__':main()
