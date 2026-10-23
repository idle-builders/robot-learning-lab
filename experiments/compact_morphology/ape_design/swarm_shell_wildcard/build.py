"""Independent 500 mm mantle/haunch ape concept. All coordinates are mm.

X is right, Y is rear, Z is up. This is an outer-form and source-package
reservation study, not a fitted mechanism or a clearance certificate.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from build123d import Compound, Plane, Solid, Vertex, Wire, export_step


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "artifacts" / "ape-shell-wildcard-500"
WALL = 2.0

PALETTE = {
    "cream": "#e7dfcf", "head": "#e9e1d1", "face": "#c6ae91",
    "muzzle": "#d8c4a8", "eye": "#282c2a", "sole": "#9a8d7b",
    "flex": "#82786b", "orange": "#db784b", "steel": "#6d8896",
    "pack": "#8b9b64", "service": "#8f71a1",
}


@dataclass
class Part:
    name: str
    shape: Solid
    color: str
    link: str
    layer: str = "skin"


def loft(stations):
    """Station tuples: z, x, y, x radius, y radius."""
    return Solid.make_loft([
        Wire.make_ellipse(rx, ry, Plane(origin=(x, y, z)))
        for z, x, y, rx, ry in stations
    ])


def hollow(stations, wall=WALL):
    outer = loft(stations)
    inner = loft([(z, x, y, max(rx-wall, 1), max(ry-wall, 1))
                  for z, x, y, rx, ry in stations])
    return outer - inner


def ellipsoid(center, radii, wall=0):
    outer = Solid.make_sphere(1).scale(radii).translate(center)
    if wall:
        inner = Solid.make_sphere(1).scale(tuple(r-wall for r in radii)).translate(center)
        return outer - inner
    return outer


def box(center, size):
    return Solid.make_box(*size, Plane(origin=tuple(c-s/2 for c, s in zip(center, size))))


def cylinder(center, radius, length, direction=(1, 0, 0)):
    start = tuple(c - d*length/2 for c, d in zip(center, direction))
    return Solid.make_cylinder(radius, length, Plane(origin=start, z_dir=direction))


def model():
    p: list[Part] = []
    add = lambda name, shape, color, link, layer="skin": p.append(Part(name, shape, color, link, layer))

    # One continuous fixed roof crosses the shoulder centres. The mobile arms
    # leave beneath its rim; the seam is deliberately in the axilla.
    body = loft([(251,0,6,43,31), (276,0,7,61,40), (316,0,7,76,48),
                 (355,0,7,83,53), (386,0,8,75,48), (406,0,9,44,32)])
    cavity = loft([(251,0,6,41,29), (276,0,7,59,38), (316,0,7,74,46),
                   (355,0,7,81,51), (386,0,8,73,46), (406,0,9,42,30)])
    for side in (-1, 1):
        body = body.fuse(ellipsoid((side*77, 7, 374), (35, 47, 32)))
        cavity = cavity.fuse(ellipsoid((side*77, 7, 374), (33, 45, 30)))
    mantle = body - cavity
    add("mantle_front", mantle & box((0,-65.15,330),(260,130,230)), "cream", "torso")
    add("mantle_rear", mantle & box((0,65.15,330),(260,130,230)), "cream", "torso")
    add("saddle_front", hollow([(241,0,8,35,25),(257,0,8,48,32),
                                (275,0,8,57,38)] ) & box((0,-54,260),(160,108,65)), "cream", "pelvis")
    add("saddle_rear", hollow([(241,0,8,35,25),(257,0,8,48,32),
                               (275,0,8,57,38)]) & box((0,54,260),(160,108,65)), "cream", "pelvis")

    add("neck", hollow([(394,0,6,20,19),(408,0,6,20,19),(421,0,6,18,18)]), "flex", "torso")
    add("cranium", ellipsoid((0,1,448),(55,49,52),WALL), "head", "torso")
    add("face_mask", ellipsoid((0,-43,447),(40,13,34),2), "face", "torso")
    add("muzzle", ellipsoid((0,-54,424),(27,19,17),2), "muzzle", "torso")
    for side in (-1,1):
        add(f"eye_{side:+d}", ellipsoid((side*19,-55,453),(7,5,8)), "eye", "torso")
        add(f"ear_{side:+d}", ellipsoid((side*53,6,450),(8,12,17),2), "face", "torso")

    for side in (-1,1):
        tag = "L" if side < 0 else "R"
        def arm_st(st): return [(z,side*x,y,rx,ry) for z,x,y,rx,ry in st]
        add(f"{tag}_upper_arm", hollow(arm_st([
            (244,108,1,23,24),(260,107,0,28,29),(302,104,1,33,32),
            (340,101,2,31,32),(367,99,3,20,23)])), "cream", f"{tag}_upper")
        add(f"{tag}_elbow_flex", hollow(arm_st([
            (235,108,1,21,22),(249,108,1,23,24),(259,107,1,23,24)])), "flex", f"{tag}_fore")
        add(f"{tag}_forearm", hollow(arm_st([
            (144,113,-5,24,25),(164,112,-4,29,29),(204,110,-2,30,30),
            (239,108,0,23,25)])), "cream", f"{tag}_fore")
        add(f"{tag}_palm", ellipsoid((side*113,-4,128),(29,28,24),WALL), "cream", f"{tag}_hand")
        for i in range(3):
            add(f"{tag}_finger_{i}", ellipsoid((side*(99+i*12),-20,110),
                (8,13,12),1.6), "cream", f"{tag}_hand")
        add(f"{tag}_thumb", ellipsoid((side*88,-13,132),(9,12,14),1.6), "cream", f"{tag}_hand")

        # The mobile haunch owns the rear and side hip fullness. The fixed
        # saddle is short; it does not leave an outboard C-shaped cheek.
        thigh = [(143,54,4,25,26),(163,54,5,31,33),(203,53,7,37,38),
                 (250,52,9,41,41),(283,51,11,36,36),(305,50,13,23,23)]
        add(f"{tag}_haunch_thigh", hollow(arm_st(thigh)), "cream", f"{tag}_thigh")
        add(f"{tag}_knee_membrane", bridge_shape(side,"knee","stand"), "flex", "world")
        add(f"{tag}_shin", hollow(arm_st([
            (58,56,5,21,22),(72,56,5,28,30),(105,55,5,31,33),
            (134,55,4,27,29),(158,55,4,22,23)])), "cream", f"{tag}_shin")
        add(f"{tag}_ankle_membrane", bridge_shape(side,"ankle","stand"), "flex", "world")
        add(f"{tag}_boot", ellipsoid((side*56,-15,32),(43,57,29),WALL), "cream", f"{tag}_boot")
        add(f"{tag}_boot_throat", hollow(arm_st([
            (40,56,5,24,26),(54,56,5,25,28),(76,56,5,22,24)])), "cream", f"{tag}_boot")
        add(f"{tag}_boot_sole", ellipsoid((side*56,-15,9),(43,57,8),2), "sole", f"{tag}_boot")
        for i in range(3):
            add(f"{tag}_toe_{i}", ellipsoid((side*(32+i*23),-58,24),
                (14,19,16),1.8), "cream", f"{tag}_boot")

    return p


def packages():
    p: list[Part] = []
    add = lambda n,sh,c,l: p.append(Part(n,sh,c,l,"package"))
    add("shared_Tattu_4S850_60x30x30p5", box((0,17,337),(60,30,30.5)), "pack", "torso")
    add("shared_converter_service_25x20x14", box((0,-22,365),(25,20,14)), "service", "torso")
    add("head_yaw_XL330", box((0,8,414),(20,34,29)), "orange", "torso")
    for side in (-1,1):
        tag = "L" if side < 0 else "R"
        for axis,center,link in [
            ("shoulder_yaw",(side*51,24,383),"torso"),
            ("shoulder_pitch",(side*83,-7,365),"torso"),
            ("shoulder_roll",(side*102,5,340),f"{tag}_upper"),
            ("elbow",(side*107,0,272),f"{tag}_upper"),
            ("paw",(side*109,-4,180),f"{tag}_fore"),
            ("hip_yaw",(side*38,24,294),"pelvis"),
        ]:
            add(f"{tag}_{axis}_XL330_20x34x29",box(center,(20,34,29)),"orange",link)
        for axis,center,link in [
            ("hip_roll",(side*53,-19,267),"pelvis"),
            ("hip_pitch",(side*61,21,238),"pelvis"),
            ("knee",(side*54,-3,199),f"{tag}_thigh"),
        ]:
            add(f"{tag}_{axis}_ST3250_45p22x24p72x35",
                box(center,(45.22,24.72,35)),"orange",link)
        add(f"{tag}_ankle_ST3215_45p2x24p7x35",
            box((side*55,8,100),(45.2,24.7,35)),"orange",f"{tag}_shin")
        # Opposed bearing stations are physically separate from each servo
        # output. Shafts and 2 mm seats show a radial reaction route.
        for joint,z,half,link in [("knee",146,22,f"{tag}_thigh"),
                                  ("ankle",59,19,f"{tag}_shin")]:
            c=(side*55,4,z)
            add(f"{tag}_{joint}_shaft_6mm",cylinder(c,3,2*half+8),"steel",link)
            for station in (-1,1):
                x=side*55+station*half
                add(f"{tag}_{joint}_bearing_{station}_W618_6",
                    cylinder((x,4,z),6.5,3.5),"steel",link)
                add(f"{tag}_{joint}_seat_{station}",
                    cylinder((x,4,z),8.5,5.5),"service",link)
            add(f"{tag}_{joint}_parent_bridge_reserve",
                box((side*55,25,z+17),(2*half+10,4,34)),"steel",link)
            add(f"{tag}_{joint}_moving_yoke_reserve",
                box((side*55,-17,z-13),(2*half+10,4,27)),"steel",
                    f"{tag}_shin" if joint=="knee" else f"{tag}_boot")
    return p


POSES = {
    "stand": {"torso":0,"hip":0,"knee":0,"ankle":0,"shoulder":0,"elbow":0},
    "bend": {"torso":22,"hip":-43,"knee":77,"ankle":-30,"shoulder":-30,"elbow":35},
    "crawl": {"torso":58,"hip":18,"knee":68,"ankle":-25,"shoulder":-67,"elbow":48},
}


def pose_matrix(link, pose):
    rot = trimesh.transformations.rotation_matrix
    rad = np.deg2rad
    spec = POSES[pose]
    if link=="world": return np.eye(4)
    torso = rot(rad(spec["torso"]),(1,0,0),(0,0,269))
    if link in ("torso","pelvis"):
        return torso if link=="torso" else np.eye(4)
    side = -1 if link.startswith("L") else 1
    if any(link.endswith(x) for x in ("_thigh","_shin","_boot")):
        hip = rot(rad(spec["hip"]),(1,0,0),(side*52,6,274))
        if link.endswith("_thigh"): return hip
        knee = hip @ rot(rad(spec["knee"]),(1,0,0),(side*55,4,146))
        if link.endswith("_shin"): return knee
        return knee @ rot(rad(spec["ankle"]),(1,0,0),(side*55,4,59))
    shoulder = torso @ rot(rad(spec["shoulder"]),(1,0,0),(side*96,2,364))
    if link.endswith("_upper"): return shoulder
    elbow = shoulder @ rot(rad(spec["elbow"]),(1,0,0),(side*108,1,246))
    return elbow


def bridge_shape(side, joint, pose):
    """Illustrative flexible envelope joining two moving shell hems.

    It is a pose-specific BRep, but no fabric/TPU strain or fold clearance is
    inferred from this loft. It makes the closure hypothesis visible.
    """
    tag="L" if side<0 else "R"
    if joint=="knee":
        ends=[((side*55,4,145),(28,29),f"{tag}_thigh"),
              ((side*55,4,156),(24,26),f"{tag}_shin")]
    else:
        ends=[((side*56,5,62),(24,26),f"{tag}_shin"),
              ((side*56,5,68),(24,26),f"{tag}_boot")]
    outer=[]; inner=[]
    for center,(rx,ry),link in ends:
        mat=pose_matrix(link,pose)
        world=mat@np.array([*center,1.])
        normal=mat[:3,:3]@np.array([0.,0.,1.])
        plane=Plane(origin=tuple(world[:3]),z_dir=tuple(normal))
        outer.append(Wire.make_ellipse(rx,ry,plane))
        inner.append(Wire.make_ellipse(rx-1.5,ry-1.5,plane))
    return Solid.make_loft(outer)-Solid.make_loft(inner)


def mesh(part):
    shape=part.shape if isinstance(part,Part) else part
    vertices, triangles = shape.tessellate(0.9, 0.2)
    return trimesh.Trimesh(vertices=np.asarray([tuple(v) for v in vertices]),
                           faces=np.asarray(triangles), process=False)


def rgb(hexcolor):
    return tuple(int(hexcolor[i:i+2],16) for i in (1,3,5))


def render(meshes, pose, direction, layer="skin", size=(470,520)):
    """Fixed orthographic software render; all views share scale and target."""
    w,h=size
    scene=np.full((h,w,3), (246,241,231), dtype=np.uint8)
    depth=np.full((h,w), np.inf, dtype=np.float32)
    camera={
        "front":np.array([0.,-1.,0.]), "side":np.array([1.,0.,0.]),
        "rear":np.array([0.,1.,0.]), "three-quarter":np.array([0.65,-0.76,0.]),
    }[direction]
    right=np.cross(np.array([0.,0.,1.]), camera)
    right/=np.linalg.norm(right)
    up=np.array([0.,0.,1.])
    scale=0.87
    for part, base in meshes:
        if part.layer!=layer: continue
        m=base.copy()
        m.apply_transform(pose_matrix(part.link, pose))
        verts=np.asarray(m.vertices)
        pix=np.stack([w/2 + verts@right*scale, h-35-(verts[:,2]*scale)],axis=1)
        toward=verts@camera
        normals=m.face_normals
        color=np.array(rgb(PALETTE[part.color]),dtype=float)
        for face,n in zip(m.faces,normals):
            if n@camera > -0.02: continue
            xy=pix[face]
            xmin=max(0,int(np.floor(xy[:,0].min()))); xmax=min(w-1,int(np.ceil(xy[:,0].max())))
            ymin=max(0,int(np.floor(xy[:,1].min()))); ymax=min(h-1,int(np.ceil(xy[:,1].max())))
            if xmin>=xmax or ymin>=ymax: continue
            v0,v1,v2=xy
            den=(v1[1]-v2[1])*(v0[0]-v2[0])+(v2[0]-v1[0])*(v0[1]-v2[1])
            if abs(den)<1e-6: continue
            yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1]
            a=((v1[1]-v2[1])*(xx-v2[0])+(v2[0]-v1[0])*(yy-v2[1]))/den
            b=((v2[1]-v0[1])*(xx-v2[0])+(v0[0]-v2[0])*(yy-v2[1]))/den
            c=1-a-b
            z=a*toward[face[0]]+b*toward[face[1]]+c*toward[face[2]]
            field=depth[ymin:ymax+1,xmin:xmax+1]
            mask=(a>=0)&(b>=0)&(c>=0)&(-z<field)
            if not np.any(mask): continue
            light=max(0.63,min(1.05,0.84+0.17*np.dot(n,[-0.4,-0.6,0.7])))
            scene[ymin:ymax+1,xmin:xmax+1][mask]=np.clip(color*light,0,255).astype(np.uint8)
            field[mask]=-z[mask]
    return Image.fromarray(scene)


def contact_sheet(meshes, pose, hardware=False):
    dirs=["front","side","rear","three-quarter"]
    sheet=Image.new("RGB",(4*470,560),(246,241,231))
    d=ImageDraw.Draw(sheet)
    posed=[]
    for p,m in meshes:
        if p.name.endswith("_membrane"):
            side=-1 if p.name.startswith("L") else 1
            joint="knee" if "knee" in p.name else "ankle"
            posed.append((p,mesh(bridge_shape(side,joint,pose))))
        else: posed.append((p,m))
    bg=np.array([246,241,231])
    for i,direction in enumerate(dirs):
        im=render(posed,pose,direction)
        if hardware:
            hw=np.asarray(render(posed,pose,direction,"package"))
            pix=np.asarray(im).copy()
            mask=np.any(hw!=bg,axis=2)
            pix[mask]=(0.38*pix[mask]+0.62*hw[mask]).astype(np.uint8)
            im=Image.fromarray(pix)
        sheet.paste(im,(i*470,40))
        d.text((i*470+18,14),f"{pose.upper()}  ·  {direction.upper()}",fill=(67,60,51))
    d.text((16,535),"500 mm concept · fixed camera/scale · diagnostic packages" if hardware
           else "500 mm concept · fixed camera/scale · opaque outer skin",fill=(83,76,67))
    return sheet


def export(parts, hardware, meshes):
    OUT.mkdir(parents=True,exist_ok=True)
    export_step(Compound(children=[p.shape for p in parts]), OUT/"outer_shells.step")
    export_step(Compound(children=[p.shape for p in hardware]), OUT/"package_reserves.step")
    entries=[]
    for p in parts+hardware:
        bb=p.shape.bounding_box()
        entries.append({"name":p.name,"layer":p.layer,"link":p.link,
            "volume_mm3":round(p.shape.volume,2),
            "bbox_mm":[[round(v,2) for v in bb.min],[round(v,2) for v in bb.max]],
            "valid":p.shape.is_valid})
    (OUT/"parts.json").write_text(json.dumps(entries,indent=2)+"\n")
    for pose in POSES:
        contact_sheet(meshes,pose).save(OUT/f"{pose}-opaque.png")
        contact_sheet(meshes,pose,True).save(OUT/f"{pose}-packages.png")
    # One inspectable neutral mesh file per layer; pose is rebuilt by JS from
    # source-linked part meshes and transforms, rather than duplicating GLBs.
    records=[]
    for part,tri in meshes:
        record={"name":part.name,"layer":part.layer,"link":part.link,
            "color":PALETTE[part.color],"vertices":np.round(tri.vertices,2).tolist(),
            "faces":tri.faces.tolist()}
        if part.name.endswith("_membrane"):
            side=-1 if part.name.startswith("L") else 1
            joint="knee" if "knee" in part.name else "ankle"
            record["posed"]={}
            for pose in ("bend","crawl"):
                posed=mesh(bridge_shape(side,joint,pose))
                record["posed"][pose]={"vertices":np.round(posed.vertices,2).tolist(),
                                       "faces":posed.faces.tolist()}
        records.append(record)
    (OUT/"meshes.json").write_text(json.dumps(records,separators=(',',':')))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args()
    global OUT
    if args.output: OUT=args.output
    parts=model(); hardware=packages()
    meshes=[(p,mesh(p)) for p in parts+hardware]
    export(parts,hardware,meshes)
    print(f"{len(parts)} shell parts, {len(hardware)} package/support reserves -> {OUT}")


if __name__=="__main__": main()
