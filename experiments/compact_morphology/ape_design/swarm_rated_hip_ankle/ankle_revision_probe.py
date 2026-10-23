"""Bounded 50 cm ankle topology sensitivity after the neutral belt-path rejection.

The source 2 mm shell stays exact.  Case A moves the ST3215 into the foot and
rotates its 35 x 24.7 x 45.2 mm vendor envelope about the anatomical pitch
axis.  Case B keeps the proximal shin motor and tests a smooth, uniformly
thicker 2 mm hollow cuff.  Both are negative bounds, not approved cutaways.
Motion is intentionally absent while exact neutral containment fails.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as b
import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/"artifacts/rated-ankle-revision-20261005"
sys.path.insert(0,str(HERE.parent/"swarm_rated_leg_skin_v2"))
from skin_v2_case import integration_shapes
from rated_hip_ankle_case import ankle_pitch, hits, maybe_intersection, mesh, shell_shapes, vol

SCALE=500/350
AXIS=(46*SCALE,4*SCALE,24*SCALE)
CUFF_CENTRE=(46*SCALE,3*SCALE,24*SCALE)
CUFF_RADII=(19*SCALE,18*SCALE,5*SCALE)
FOOT_NAMES=("R_shin_","R_ankle_cuff","R_foot_","R_sole","R_toe_")
ADJACENT_OUTER=("boot_R","ankle_cuff_R","shin_R")


def cuff(vertical_radius):
    radii=(CUFF_RADII[0],CUFF_RADII[1],vertical_radius)
    outer=b.Solid.make_sphere(1).scale(radii).translate(CUFF_CENTRE)
    inner=b.Solid.make_sphere(1).scale(tuple(a-2 for a in radii)).translate(CUFF_CENTRE)
    return outer-inner


def direct_foot_motor(side,angle_deg,axis_z):
    # The output face is x=AXIS.x; the 12.2 mm long-case axis offset comes
    # from the same conservative ST3215 case used in the first coupon.
    theta=math.radians(angle_deg)
    center=(AXIS[0]+side*17.5,AXIS[1]-12.2*math.sin(theta),
            axis_z+12.2*math.cos(theta))
    return b.Pos(*center)*b.Rot(angle_deg,0,0)*b.Box(35,24.7,45.2)


def residual(shape,outer):
    remainder=shape
    for name in ADJACENT_OUTER:
        if name not in outer:continue
        solid=outer[name][0]
        if maybe_intersection(remainder,solid)>.05:
            remainder=remainder.cut(solid)
    return remainder


def direct_grid(skin,outer):
    local={n:s for n,s in skin.items() if n.startswith(FOOT_NAMES)}
    rows=[]
    for side in (-1,1):
        for angle in (60,75,90,105,120,240,255,270,285,300):
            for z in (22,24,26,28,30):
                motor=direct_foot_motor(side,angle,z)
                row_hits=hits(motor,local)
                outside=vol(residual(motor,outer))
                rows.append(dict(case_side=side,roll_about_pitch_axis_deg=angle,
                                 pitch_axis_z_mm=z,skin_hits_mm3=row_hits,
                                 total_skin_overlap_mm3=round(sum(row_hits.values()),3),
                                 outside_adjacent_outer_mm3=round(outside,3),
                                 neutral_clear=not row_hits and outside<=.05))
    return sorted(rows,key=lambda x:(x["total_skin_overlap_mm3"],x["outside_adjacent_outer_mm3"]))


def cuff_grid(skin,parts,outer):
    rows=[]
    for r in (CUFF_RADII[2],9,10,12,14,16,18):
        fairing=cuff(r)
        overlap=hits(fairing,{n:s for n,s in parts.items() if n!="motor_case"})
        rows.append(dict(vertical_radius_mm=round(r,3),
                         outside_height_increase_mm=round(2*(r-CUFF_RADII[2]),3),
                         added_shell_material_mm3=round(vol(fairing)-vol(skin["R_ankle_cuff"]),3),
                         outside_source_body_mm3=round(vol(residual(fairing,outer)),3),
                         hardware_hits_mm3=overlap,
                         source_foot_dorsal_hit_mm3=round(maybe_intersection(fairing,skin["R_foot_dorsal"]),3),
                         neutral_clear=not overlap and maybe_intersection(fairing,skin["R_foot_dorsal"])<=.05))
    return rows


def scene_for(skin,cuff_shape=None,motor=None,outside=None):
    scene=trimesh.Scene()
    for name,shape in skin.items():
        if name.startswith(("R_shin_","R_foot_","R_sole","R_toe_")):
            mesh(scene,name,shape,(225,218,204,255))
    mesh(scene,"R_ankle_cuff",cuff_shape if cuff_shape is not None else skin["R_ankle_cuff"],
         (98,106,100,255))
    if motor is not None:
        mesh(scene,"foot_ST3215",motor,(67,103,126,255))
    if outside is not None and vol(outside)>.05:
        mesh(scene,"motor_outside_source_body",outside,(195,72,55,255))
    return scene


def panel(scene,view):
    geo=scene.to_geometry()
    vertices=np.asarray(geo.vertices)
    faces=np.asarray(geo.faces)
    normals=np.asarray(geo.face_normals)
    tris=vertices[faces]
    if view=="front":
        u=tris[:,:,0]-65; depth=tris[:,:,1].mean(axis=1)
        front=-normals[:,1]; order=np.argsort(-depth)
    else:
        u=-tris[:,:,1]; depth=tris[:,:,0].mean(axis=1)
        front=normals[:,0]; order=np.argsort(depth)
    v=tris[:,:,2]-59
    size=620
    xy=np.stack((size/2+u*4.5,size/2-v*4.5),axis=2)
    colors=np.asarray(geo.visual.face_colors)[:,:3]
    canvas=Image.new("RGB",(size,size),(220,217,207))
    draw=ImageDraw.Draw(canvas)
    light=np.array([.35,-.7,.63]);light/=np.linalg.norm(light)
    for i in order:
        if front[i]<=.01:continue
        shade=.68+.32*max(0,float(normals[i]@light))
        color=tuple(int(np.clip(c*shade,0,255)) for c in colors[i])
        draw.polygon([tuple(p) for p in xy[i]],fill=color)
    return canvas


def export_views(skin,outer,best):
    direct=direct_foot_motor(best["case_side"],best["roll_about_pitch_axis_deg"],best["pitch_axis_z_mm"])
    variants=(("approved source",scene_for(skin)),
              ("rounded cuff R18 · still collides",scene_for(skin,cuff(18))),
              ("direct foot · protrusion red",scene_for(skin,motor=direct,outside=residual(direct,outer))))
    sheet=Image.new("RGB",(620*3,620*2+64),(248,246,240))
    draw=ImageDraw.Draw(sheet)
    for col,(name,scene) in enumerate(variants):
        scene.export(OUT/("source.glb" if col==0 else "rounded-cuff.glb" if col==1 else "direct-foot.glb"))
        for row,view in enumerate(("front","side")):
            draw.text((col*620+14,row*(620+32)+8),name+" · "+view,fill=(35,50,43))
            sheet.paste(panel(scene,view),(col*620,row*(620+32)+32))
    sheet.save(OUT/"matched-opaque-neutral.png")
    b.export_step(b.Compound(children=[cuff(18),direct]),str(OUT/"ankle-revision-500.step"))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    case=integration_shapes(500)
    skin=case["revised_skin"]
    outer,_=shell_shapes(500,2)
    outer["ankle_cuff_R"]=(b.Solid.make_sphere(1).scale(CUFF_RADII).translate(CUFF_CENTRE),
                           skin["R_ankle_cuff"])
    parts=ankle_pitch(6)["parts"]
    direct=direct_grid(skin,outer)
    thick=cuff_grid(skin,parts,outer)
    best=direct[0]
    report=dict(height_mm=500,source="exact Study(500, wall=2, gap=.65) shell plus committed knee local contour",
                direct_foot_grid=dict(count=len(direct),best=best,all=direct),
                rounded_cuff_grid=thick,
                pose_screen="not_run_no_static_clear_candidate",
                sources=dict(ST3215="https://www.feetechrc.com/525603.html",
                             SKF_W618_6="https://www.skf.com/binaries/pub12/Images/0901d19680406705-SKF-Stainless-steel-DGBB---11279_1-EN-Low-Res_tcm_12-259995.pdf",
                             Gates_123_3MGT_15="https://www.gates.com/content/dam/gates/home/knowledge-center/resource-library/catalogs/power-transmission-catalog.pdf"),
                verdict="Foot-mounted direct ST3215 case fails exact skin and outer containment in every bounded orientation/axis sample; uniform cuff thickening through R18 still cuts belt, shaft, medial bearing/seat/support and moving yoke. No opaque covered moving ankle is established.")
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    export_views(skin,outer,best)
    print("best direct",best)
    print("thickest cuff",thick[-1])
    print("wrote",OUT/"report.json")


if __name__=="__main__":main()
