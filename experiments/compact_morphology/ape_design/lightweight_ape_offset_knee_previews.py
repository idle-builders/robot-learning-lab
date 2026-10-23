"""Paired fixed-scale source/coupon pose sheets for the offset knee."""
from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

import lightweight_ape_offset_knee as coupon


OUT = coupon.OUT
W,H = 480,570
MM_PER_PIXEL = .28
VIEWS = {"front":(0,-1,.16),"side":(1,0,.12),"rear":(0,1,.16),
         "threequarter":(1,-1,.38)}


def source_mesh(shape,color):
    v,f=shape.tessellate(.5,.4)
    mesh=trimesh.Trimesh(np.asarray([tuple(p) for p in v]),
                         np.asarray(f),process=False)
    mesh.visual.face_colors=color
    return mesh


def project(meshes, eye):
    eye=np.asarray(eye,dtype=float)
    eye/=np.linalg.norm(eye)
    right=np.cross(np.array([0.,0.,1.]),eye)
    right/=np.linalg.norm(right)
    up=np.cross(eye,right)
    target=np.array([42.,15.,95.])
    polygons=[]
    light=np.array([.36,-.58,.73]);light/=np.linalg.norm(light)
    for mesh in meshes:
        vertices=np.asarray(mesh.vertices)-target
        xy=np.stack((W/2+vertices@right/MM_PER_PIXEL,
                     H/2-vertices@up/MM_PER_PIXEL),axis=1)
        depth=vertices@eye
        colors=np.asarray(mesh.visual.face_colors)
        for i,tri in enumerate(mesh.faces):
            pts=xy[tri]
            if (np.all(pts[:,0]<-5) or np.all(pts[:,0]>W+5) or
                np.all(pts[:,1]<-5) or np.all(pts[:,1]>H+5)):
                continue
            brightness=.67+.33*max(0.,float(mesh.face_normals[i]@light))
            color=tuple(int(np.clip(c*brightness,0,255)) for c in colors[i,:3])
            polygons.append((float(depth[tri].mean()),pts,color))
    polygons.sort(key=lambda x:x[0])
    image=Image.new("RGB",(W,H),"#e8e3d8")
    draw=ImageDraw.Draw(image)
    for _,pts,color in polygons:
        draw.polygon([tuple(p) for p in pts],fill=color)
    return image


def main():
    report=json.loads((OUT/"report.json").read_text())
    source={p.label:p for p in bd.import_step(coupon.SOURCE).children}
    fixed=[source_mesh(source[n],(225,215,194,255)) for n in
           ("R_thigh_front","R_thigh_back")]
    move=[source_mesh(source[n],(230,211,179,255)) for n in
          ("R_shin_front","R_shin_back")]
    move.append(source_mesh(source["R_knee_cuff"],(68,70,69,255)))
    font=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf",18)
    small=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf",14)
    for degrees in coupon.POSES:
        transform=trimesh.transformations.rotation_matrix(
            np.deg2rad(degrees),(1,0,0),coupon.AXIS)
        original=fixed+[m.copy().apply_transform(transform) for m in move]
        built=list(trimesh.load(OUT/f"offset-knee-{degrees:03d}.glb",force="scene").dump())
        row=next(r for r in report["poses"] if r["flexion_deg"]==degrees)
        for view,eye in VIEWS.items():
            canvas=Image.new("RGB",(W*2+24,H+98),"#f8f6f0")
            canvas.paste(project(original,eye),(8,43))
            canvas.paste(project(built,eye),(W+16,43))
            draw=ImageDraw.Draw(canvas)
            draw.text((10,10),f"{degrees}° {view} | source exterior",font=font,fill="#26302d")
            draw.text((W+18,10),"offset axis + inner support",font=font,fill="#26302d")
            draw.text((10,H+53),
                      f"Axis Y14 Z106 mm | source shin top Y{row['shin_top_center_yz_mm'][0]:.1f} mm"
                      f" | rear excursion {row['skin_rear_excursion_past_source_thigh_at_axis_mm']:.1f} mm",
                      font=small,fill="#46504d")
            draw.text((10,H+73),"Orange: moving yoke  |  blue: XL330-size envelope  |  grey: fixed carrier",
                      font=small,fill="#46504d")
            path=OUT/f"preview-{view}-{degrees:03d}.png"
            canvas.save(path)
            print(path)


if __name__=="__main__":
    main()
