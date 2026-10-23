"""Two whole-character 500 mm sketch-ape shell studies.

Run: .venv/bin/python experiments/compact_morphology/ape_design/swarm_shell_v2/build.py

Millimetres; X is robot right, Y is rear, Z is up. This is an outer-skin and
anatomical-axis study. Source-sized component cases are diagnostic reservations,
not selected drives or a supported assembly. STEP is emitted for the selected
standing shell; the GLBs and pose matrices allow inspecting the other variant.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import build123d as b
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
IVORY = (228, 221, 207, 255)
WARM = (237, 227, 209, 255)
FACE = (207, 185, 157, 255)
SEAM = (75, 70, 65, 255)
EYE = (29, 29, 27, 255)
ORANGE = (203, 105, 65, 255)
PACKAGE = (52, 111, 160, 165)
CORRIDOR = (71, 153, 123, 140)
WALL = 2.0


@dataclass(frozen=True)
class Variant:
    name: str
    head_rx: float
    torso_rx: float
    torso_ry: float
    shoulder_x: float
    hip_x: float
    thigh_rx: float
    cheek_depth: float
    boot_forward: float
    v_center: float
    v_curve: float


VARIANTS = (
    Variant("cove", 53, 88, 47, 119, 59, 34, 32, 68, 232, .0083),
    Variant("moss", 61, 96, 45, 130, 65, 37, 35, 77, 248, .0040),
)


@dataclass
class Part:
    name: str
    group: str
    shape: b.Shape
    color: tuple[int, int, int, int]
    moving: str = "fixed"


def ellipsoid(center, radii):
    return b.Solid.make_sphere(1).scale(radii).translate(center)


def loft(stations):
    """Each station: (Z, X centre, Y centre, X radius, Y radius)."""
    return b.Solid.make_loft([
        b.Wire.make_ellipse(rx, ry, b.Plane(origin=(x, y, z)))
        for z, x, y, rx, ry in stations
    ])


def skin(stations, wall=WALL):
    outer = loft(stations)
    inner = loft([(z+(-1 if i==0 else 1 if i==len(stations)-1 else 0),
                   x, y, rx-wall, ry-wall)
                  for i,(z,x,y,rx,ry) in enumerate(stations)])
    return outer-inner, inner


def clip(shape, x0=-500, x1=500, y0=-500, y1=500, z0=-500, z1=600):
    box = b.Solid.make_box(x1-x0, y1-y0, z1-z0, b.Plane(origin=(x0, y0, z0)))
    return shape & box


def box(center, size):
    return b.Pos(*center) * b.Box(*size)


def above_v(v):
    samples=[(x,-120,v.v_center+v.v_curve*x*x) for x in np.linspace(-160,160,17)]
    edges=[b.Edge.make_spline(samples),b.Edge.make_line(samples[-1],(160,-120,600)),
           b.Edge.make_line((160,-120,600),(-160,-120,600)),
           b.Edge.make_line((-160,-120,600),samples[0])]
    return b.Solid.extrude(b.Face(b.Wire(edges)),(0,240,0))


def chest_v_inlay(v, stations):
    points=[]
    for x in np.linspace(-53,53,15):
        z=max(236.,v.v_center+v.v_curve*x*x+2.2)
        rx=float(np.interp(z,[s[0] for s in stations],[s[3] for s in stations]))
        ry=float(np.interp(z,[s[0] for s in stations],[s[4] for s in stations]))
        cy=float(np.interp(z,[s[0] for s in stations],[s[2] for s in stations]))
        y=cy-ry*math.sqrt(max(.02,1-(x/rx)**2))-1.1
        points.append((float(x),y,z))
    path=b.Edge.make_spline(points)
    return b.Solid.sweep(b.Wire.make_circle(.9,b.Plane(origin=points[0],z_dir=(1,0,0))),path)


def add(parts, name, group, shape, color=IVORY, moving="fixed"):
    if not shape.is_valid or len(shape.solids()) != 1 or shape.volume <= 0:
        raise ValueError(f"{name}: invalid / disconnected CAD shell")
    parts.append(Part(name, group, shape, color, moving))


def build(v: Variant):
    parts = []
    cavities = {}
    # One rounded face, a low muzzle, and a small visual neck seam.
    head = ellipsoid((0, -1, 447), (v.head_rx, 47, 53))
    inner = ellipsoid((0, -1, 447), (v.head_rx-WALL, 45, 51))
    add(parts, "head", "head", head-inner)
    cavities["head"] = inner
    face = clip(ellipsoid((0, -39, 442), (37, 13, 35)), y1=-38)
    add(parts, "face", "head", face, FACE)
    muzzle = ellipsoid((0, -51, 419), (27, 16, 18))
    add(parts, "muzzle", "head", muzzle, FACE)
    for side in (-1, 1):
        add(parts, f"eye_{side}", "head", ellipsoid((side*18, -52, 452), (6.8, 3.8, 7.5)), EYE)
    add(parts, "neck_shadow", "head", ellipsoid((0, 1, 397), (25, 24, 6)), SEAM)

    # The broad chest slopes into an intentionally short fixed pelvis V.
    torso_st = [(234,0,2,44,34),(252,0,3,59,39),
                (282,0,4,v.torso_rx-9,v.torso_ry+2),
                (324,0,4,v.torso_rx+2,v.torso_ry+5),
                (355,0,4,v.shoulder_x-12,v.torso_ry),
                (375,0,3,v.shoulder_x-30,39),(397,0,2,34,26)]
    torso, torso_inner = skin(torso_st)
    lower_edge=above_v(v)
    torso=torso & lower_edge
    torso_inner=torso_inner & lower_edge
    for side in (-1,1):
        torso=torso-ellipsoid((side*v.shoulder_x,2,345),(19,23,19))
    add(parts, "torso", "torso", torso)
    cavities["torso"] = torso_inner
    add(parts,"chest_v_seam","detail",chest_v_inlay(v,torso_st),SEAM)
    pelvis_st = [(201,0,3,20,24),(214,0,4,39,32),
                 (232,0,4,59,38),(251,0,4,69+(v.hip_x-59)*.7,41),
                 (266,0,4,64+(v.hip_x-59)*.7,40)]
    pelvis, pelvis_inner = skin(pelvis_st)
    # The lateral bowl is open for each *moving* thigh-dominant hip lobe.
    for side in (-1, 1):
        pelvis = pelvis - ellipsoid((side*v.hip_x, 2, 231), (33, 35, 29))
    add(parts, "pelvis_v", "pelvis", pelvis, WARM)
    cavities["pelvis"] = pelvis_inner
    add(parts, "chest_light", "detail", ellipsoid((0,-(v.torso_ry+1),354),(2.4,2.4,6.8)), ORANGE)

    for side in (-1, 1):
        tag = "R" if side > 0 else "L"
        sx = side*v.shoulder_x
        hx = side*v.hip_x
        kx = side*(v.hip_x+2)
        ax = side*(v.hip_x+4)
        # The torso loft itself is one fixed sloping roof over the root.
        upper, upper_inner = skin([
            (272,sx+side*9,0,17,20),(281,sx+side*10,0,23,25),
            (307,sx+side*5,1,27,28),(329,sx+side*2,2,25,27),
            (342,sx,2,19,23),(352,sx-side*2,2,16,19)])
        add(parts, f"{tag}_upper_arm", "arm", upper, IVORY, f"{tag}_upper")
        cavities[f"{tag}_upper_arm"] = upper_inner
        fore, fore_inner = skin([
            (176,sx+side*12,-2,17,19),(185,sx+side*13,-2,23,25),
            (219,sx+side*13,-1,27,28),(252,sx+side*11,0,25,26),
            (273,sx+side*9,0,18,20)])
        add(parts, f"{tag}_forearm", "arm", fore, WARM, f"{tag}_lower")
        cavities[f"{tag}_forearm"] = fore_inner
        hand, _ = skin([(119,sx+side*15,-4,14,17),(128,sx+side*15,-5,23,24),
                        (151,sx+side*14,-4,26,27),(174,sx+side*12,-3,17,20)])
        add(parts, f"{tag}_paw", "hand", hand, IVORY, f"{tag}_lower")
        # Moving hip owns the round side/rear lobe; its narrow upper neck is
        # concealed by the shortened V. These are separate outer CAD pieces.
        thigh, thigh_inner = skin([
            (132,kx,3,23,26),(146,kx,3,v.thigh_rx-3,32),
            (173,hx,3,v.thigh_rx+2,36),(202,hx,4,v.thigh_rx+3,38),
            (226,hx,4,v.thigh_rx+1,37),(244,hx-side*2,4,30,34),
            (255,hx-side*4,4,24,28),(264,hx-side*6,4,15,20)])
        # A nearly round moving side/back volume closes the fixed aperture
        # through large pitch changes. It is integral to the thigh outer part,
        # not a separate exposed cheek or tall cup.
        hip_outer=ellipsoid((hx,2,231),(31,33,27))
        hip_inner=ellipsoid((hx,2,231),(29,31,25))
        thigh=thigh.fuse(hip_outer-hip_inner)
        add(parts, f"{tag}_thigh", "leg", thigh, IVORY, f"{tag}_thigh")
        cavities[f"{tag}_thigh"] = thigh_inner
        # The cheek is outboard of a recessed moving knee crown, avoiding a
        # big exposed ball at the neutral seam. Its distal lip is rounded.
        cheek_c = (kx, 3, 139)
        cheek = ellipsoid(cheek_c, (31, v.cheek_depth, 34)) - \
                ellipsoid(cheek_c, (29, v.cheek_depth-2, 32))
        sign = side
        x_cut = kx+sign*22
        cheek = clip(cheek, x0=x_cut if sign>0 else -500,
                     x1=500 if sign>0 else x_cut)
        add(parts, f"{tag}_knee_cheek", "knee", cheek, IVORY, f"{tag}_thigh")
        crown = ellipsoid((kx, 3, 132), (20.5, 25.5, 24)) - \
                ellipsoid((kx, 3, 132), (18.5, 23.5, 22))
        add(parts, f"{tag}_knee_crown", "knee", crown, WARM, f"{tag}_shin")
        shin, shin_inner = skin([
            (44,ax,3,19,19),(52,ax,3,25,24),
            (76,ax-side*1,3,29,28),(104,kx,3,28,27),
            (123,kx,3,24,24),(139,kx,3,18,20)])
        add(parts, f"{tag}_shin", "leg", shin, WARM, f"{tag}_shin")
        cavities[f"{tag}_shin"] = shin_inner
        # A low boot carries the complete foot. The fixed rounded shin skirt
        # covers a shortened moving throat; dark bellows are shown as an
        # envelope only and have no strain / fold validation.
        skirt, skirt_inner = skin([(39,ax,3,29,27),(45,ax,3,31,29),
                                   (56,ax,3,30,29),(64,ax,3,27,27)])
        add(parts, f"{tag}_shin_skirt", "ankle", skirt, WARM, f"{tag}_shin")
        cavities[f"{tag}_shin_skirt"] = skirt_inner
        throat = ellipsoid((ax, 1, 40), (21, 21, 10)) - \
                 ellipsoid((ax, 1, 40), (19, 19, 8))
        add(parts, f"{tag}_boot_throat", "ankle", throat, IVORY, f"{tag}_boot")
        bellows = ellipsoid((ax, 2, 42), (25, 24, 7)) - \
                  ellipsoid((ax, 2, 42), (23.5, 22.5, 5.5))
        add(parts, f"{tag}_bellows_envelope", "ankle", bellows, SEAM, f"{tag}_boot")
        foot, foot_inner = skin([
            (1,ax,-10,33,v.boot_forward-10),(4,ax,-10,35,v.boot_forward),
            (13,ax,-10,35,v.boot_forward-1),(25,ax,-8,32,47),
            (35,ax,-2,25,30),(44,ax,1,19,21)])
        # Two short shallow toe divisions in the forward cap.
        for dx in (-11, 11):
            foot = foot - box((ax+dx,-59,9),(1.2,19,12))
        add(parts, f"{tag}_boot", "foot", foot, IVORY, f"{tag}_boot")
        cavities[f"{tag}_boot"] = foot_inner

    return parts, cavities


def mesh(shape, color):
    vertices, faces = shape.tessellate(1.15, .18)
    result = trimesh.Trimesh(vertices=np.array([[q.X,q.Y,q.Z] for q in vertices]),
                             faces=np.asarray(faces), process=False)
    result.visual.vertex_colors = color
    return result


def rotation(axis, angle_deg, pivot):
    from scipy.spatial.transform import Rotation
    mat = np.eye(4)
    mat[:3,:3] = Rotation.from_rotvec(np.radians(angle_deg)*np.asarray(axis)).as_matrix()
    q = np.asarray(pivot)
    mat[:3,3] = q-mat[:3,:3]@q
    return mat


def translation(z=0):
    m=np.eye(4);m[2,3]=z;return m


POSES = {
    "stand": dict(torso=0, hip=0, knee=0, ankle=0, shoulder=0, elbow=0),
    "bend": dict(torso=27, hip=-27, knee=53, ankle=-26, shoulder=18, elbow=-19),
    "squat": dict(torso=9, hip=-42, knee=85, ankle=-43, shoulder=30, elbow=-32),
    "crawl": dict(torso=68, hip=-45, knee=45, ankle=0, shoulder=-68, elbow=0),
}
POSE_GROUND_SHIFT_MM={"stand":0,"bend":-19,"squat":-48,"crawl":-23}


def matrices(v, pose):
    q=POSES[pose]
    t=rotation((1,0,0),q["torso"],(0,0,190 if pose=="crawl" else 220))
    result={"fixed":np.eye(4),"upper":t}
    for side,tag in ((-1,"L"),(1,"R")):
        sx=side*v.shoulder_x;hx=side*v.hip_x;kx=side*(v.hip_x+2);ax=side*(v.hip_x+4)
        h=rotation((1,0,0),q["hip"],(hx,2,231))
        k=h@rotation((1,0,0),q["knee"],(kx,3,139))
        a=k@rotation((1,0,0),q["ankle"],(ax,2,43))
        sh=t@rotation((1,0,0),q["shoulder"],(sx,2,347))
        el=sh@rotation((1,0,0),q["elbow"],(sx+side*9,0,273))
        result.update({f"{tag}_thigh":h,f"{tag}_shin":k,f"{tag}_boot":a,
                       f"{tag}_upper":sh,f"{tag}_lower":el})
    # Fixed pelvis, torso/head/shoulder crown and movable links have distinct
    # groups; this transform is a pose sketch, not a closed-loop robot solve.
    floor=translation(POSE_GROUND_SHIFT_MM[pose])
    return {name:floor@mat for name,mat in result.items()}


def posed_transform(part, mats):
    if part.moving != "fixed":return mats[part.moving]
    if part.group in ("head","torso","shoulder","detail"):return mats["upper"]
    return mats["fixed"]


def packages(v):
    """Valid conservative envelopes at source sizes, with explicit guesses."""
    out=[("4S_pack_60x30x30p5",box((0,13,290),(60,30,30.5)),"upper"),
         ("converter_service_25x20x14",box((0,-18,369),(25,20,14)),"upper")]
    for side,tag in ((-1,"L"),(1,"R")):
        hx=side*v.hip_x;kx=side*(v.hip_x+2);ax=side*(v.hip_x+4)
        # 25 x 35.8 mm RSF-5B-50-IDT housing, based on earlier sourced coupon.
        out.append((f"{tag}_hip_RSF25x35p8",
            b.Solid.make_cylinder(12.5,35.8,b.Plane(origin=(side*14,0,211))),"fixed"))
        out.append((f"{tag}_shoulder_RSF25x35p8",
            b.Solid.make_cylinder(12.5,35.8,b.Plane(origin=(side*44,0,318))),"upper"))
        # Official XL330 bounding envelope 20 x 34 x 29 mm, not raw STEP.
        out.append((f"{tag}_knee_XL330_20x34x29",box((kx,0,180),(20,34,29)),f"{tag}_thigh"))
        out.append((f"{tag}_ankle_XL330_20x34x29",box((ax,3,78),(20,34,29)),f"{tag}_shin"))
    return out


def corridors(v):
    """Named exact reserve solids, not a claim of swept nonmating clearance."""
    out=[]
    for side,tag in ((-1,"L"),(1,"R")):
        sx=side*v.shoulder_x;hx=side*v.hip_x;kx=side*(v.hip_x+2);ax=side*(v.hip_x+4)
        out.extend([
            (f"{tag}_shoulder_roof_opening",ellipsoid((sx,2,345),(20,24,19)),"upper",["torso",f"{tag}_upper_arm"]),
            (f"{tag}_hip_lobe_reserve",ellipsoid((hx,2,231),(33,35,29)),"fixed",["pelvis",f"{tag}_thigh"]),
            (f"{tag}_knee_nested_space",ellipsoid((kx,3,139),(23,29,29)),f"{tag}_thigh",[f"{tag}_thigh",f"{tag}_shin"]),
            (f"{tag}_ankle_skirt_space",ellipsoid((ax,2,43),(26,26,13)),f"{tag}_shin",[f"{tag}_shin_skirt",f"{tag}_boot"]),
        ])
    return out


VIEWS = {"front":(0,-1,.12),"side":(1,0,.08),
         "rear":(0,1,.12),"three-quarter":(1,-1,.22)}


def render(parts_mesh, mats, view, size=(290,340)):
    forward=np.asarray(VIEWS[view],float);forward/=np.linalg.norm(forward)
    up=np.asarray((0.,0.,1.));right=np.cross(up,forward);right/=np.linalg.norm(right)
    up=np.cross(forward,right)
    img=Image.new("RGB",size,(242,238,229));draw=ImageDraw.Draw(img)
    center=np.asarray((0.,-25.,248.));scale=.57
    light=np.asarray((.35,-.4,.85));light/=np.linalg.norm(light)
    polygons=[]
    for part,m in parts_mesh:
        tf=posed_transform(part,mats)
        verts=(m.vertices@tf[:3,:3].T)+tf[:3,3]
        rel=verts-center
        q=np.stack((rel@right,rel@up),axis=1)*scale+np.array((size[0]/2,size[1]/2))
        q[:,1]=size[1]-q[:,1]
        depth=rel@forward
        col=np.array(part.color[:3])
        for i,f in enumerate(m.faces):
            polygons.append((float(np.mean(depth[f])),tuple(map(tuple,q[f])),
                             tuple(np.clip(col*.93,0,255).astype(int))))
    for _,poly,color in sorted(polygons,key=lambda p:p[0]):draw.polygon(poly,fill=color)
    return img


def make_sheet(v, parts_mesh):
    w,h=290,340
    sheet=Image.new("RGB",(4*w,4*(h+24)),"white")
    pen=ImageDraw.Draw(sheet)
    for row,pose in enumerate(POSES):
        mats=matrices(v,pose)
        for col,view in enumerate(VIEWS):
            image=render(parts_mesh,mats,view,(w,h))
            sheet.paste(image,(col*w,row*(h+24)+24))
            pen.text((col*w+8,row*(h+24)+5),f"{v.name} · {pose} · {view}",fill=(45,42,39))
    path=OUT/f"{v.name}-silhouettes.png"
    sheet.save(path,optimize=True)


def shape_bounds(shape):
    q=shape.bounding_box()
    return [[round(float(x),2) for x in q.min],[round(float(x),2) for x in q.max]]


def measure(v, parts, cavities, pkgs, reserves):
    shell={p.name:p.shape for p in parts}
    records=[]
    for name,shape,moving in pkgs:
        intersections=[]
        for p in parts:
            # A fast AABB guard avoids unrelated Boolean work.
            a=shape.bounding_box();c=p.shape.bounding_box()
            if any(getattr(a.min,q)>=getattr(c.max,q) or getattr(c.min,q)>=getattr(a.max,q) for q in "XYZ"):
                continue
            try:
                volume=sum(s.volume for s in (shape&p.shape).solids())
            except Exception:
                volume=None
            if volume is None or volume>.05:intersections.append(dict(shell=p.name,intersection_mm3=None if volume is None else round(volume,1)))
        records.append(dict(name=name,moving=moving,volume_mm3=round(shape.volume,1),bounds_mm=shape_bounds(shape),
                            neutral_shell_intersections=intersections,
                            placement="provisional envelope; output datum/support/drive path not solved"))
    joint_openings={}
    for name in ("torso","pelvis","R_upper_arm","R_thigh","R_shin","R_shin_skirt","R_boot"):
        q=cavities[name];joint_openings[name]=dict(interior_volume_mm3=round(q.volume,1),bounds_mm=shape_bounds(q))
    corridor_records=[]
    for name,shape,moving,near in reserves:
        interior={}
        for key in near:
            try:interior[key]=round(sum(s.volume for s in (shape & cavities[key]).solids()),1)
            except Exception:interior[key]=None
        corridor_records.append(dict(name=name,moving=moving,bounds_mm=shape_bounds(shape),
                                     gross_reserve_volume_mm3=round(shape.volume,1),
                                     overlap_with_named_open_interiors_mm3=interior,
                                     interpretation="gross space proxy; interior overlaps must not be added"))
    return dict(height_target_mm=500,wall_mm=WALL,variant=v.__dict__,
                axes_mm={"shoulder_R":[v.shoulder_x,2,347],
                         "hip_R":[v.hip_x,2,231],"knee_R":[v.hip_x+2,3,139],
                         "ankle_R":[v.hip_x+4,2,43],
                         "pitch_axis":[1,0,0],"shoulder_yaw_axis":[0,0,1],
                         "hip_yaw_axis":[0,0,1]},
                interior_volumes=joint_openings,
                joint_corridors=corridor_records,
                package_envelopes=records,
                parts=[dict(name=p.name,group=p.group,moving=p.moving,
                            volume_mm3=round(p.shape.volume,1),bounds_mm=shape_bounds(p.shape)) for p in parts],
                pose_angles_deg=POSES,
                pose_ground_shift_mm=POSE_GROUND_SHIFT_MM,
                limitations=["Pose images use anatomical axes and rigid transforms, not a closed-loop joint solve.",
                             "No dynamic BRep clearance or nonmating support pass is claimed.",
                             "Bellows fold, material strain, attachments, shell fastening and heat paths remain open."])


def export_variant(v):
    print(f"building {v.name}",flush=True)
    parts,cavities=build(v)
    part_mesh=[(p,mesh(p.shape,p.color)) for p in parts]
    scene=trimesh.Scene()
    for p,m in part_mesh:scene.add_geometry(m,geom_name=p.name,node_name=p.name)
    scene.export(OUT/f"{v.name}-shell.glb")
    pkg=packages(v)
    pkg_scene=trimesh.Scene()
    for name,shape,moving in pkg:
        pkg_scene.add_geometry(mesh(shape,PACKAGE),geom_name=name,node_name=name)
    pkg_scene.export(OUT/f"{v.name}-packages.glb")
    reserves=corridors(v)
    reserve_scene=trimesh.Scene()
    for name,shape,moving,near in reserves:
        reserve_scene.add_geometry(mesh(shape,CORRIDOR),geom_name=name,node_name=name)
    reserve_scene.export(OUT/f"{v.name}-corridors.glb")
    (OUT/f"{v.name}-report.json").write_text(json.dumps(measure(v,parts,cavities,pkg,reserves),indent=2)+"\n")
    make_sheet(v,part_mesh)
    if v.name=="cove":
        assembly=b.Compound(label="500 mm Cove sketch-ape shell",children=[p.shape for p in parts])
        step=OUT/"cove-shell.step"
        b.export_step(assembly,str(step))
        # OCCT writes trailing spaces on multiline STEP entity declarations.
        step.write_text("\n".join(line.rstrip() for line in step.read_text().splitlines())+"\n")
    return parts


def make_comparison():
    """Paired, scale-matched front and three-quarter views of both forms."""
    sheets={v.name:Image.open(OUT/f"{v.name}-silhouettes.png") for v in VARIANTS}
    w,h=290,364
    canvas=Image.new("RGB",(4*w,3*h),(255,255,255))
    for row,pose_index in enumerate((0,1,3)):
        for column,(name,view_index) in enumerate((("cove",0),("moss",0),("cove",3),("moss",3))):
            tile=sheets[name].crop((view_index*w,pose_index*h,(view_index+1)*w,(pose_index+1)*h))
            canvas.paste(tile,(column*w,row*h))
    canvas.save(OUT/"paired-comparison.png",optimize=True)


def make_reference_comparison():
    """Stand at one displayed figure height beside the exact-sketch CAD preview."""
    source=HERE.parents[2]/"sketch_ape_shell"/"previews"
    panels={v.name:Image.open(OUT/f"{v.name}-silhouettes.png").convert("RGB") for v in VARIANTS}
    w,h=290,364
    canvas=Image.new("RGB",(3*w,2*h),(242,238,229))
    pen=ImageDraw.Draw(canvas)
    for row,(view,source_name,crop) in enumerate((("front","front.png",(103,70,597,805)),
                                                   ("side","side.png",(255,70,445,805)))):
        src=Image.open(source/source_name).convert("RGB").crop(crop)
        scaled=src.resize((round(src.width*288/src.height),288),Image.Resampling.LANCZOS)
        canvas.paste(scaled,((w-scaled.width)//2,row*h+24+340-288))
        pen.text((8,row*h+5),f"source 500 mm · {view}",fill=(45,42,39))
        view_index=0 if view=="front" else 1
        for col,name in enumerate(("cove","moss"),1):
            tile=panels[name].crop((view_index*w,0,(view_index+1)*w,h))
            canvas.paste(tile,(col*w,row*h))
    canvas.save(OUT/"source-scale-comparison.png",optimize=True)


def main():
    OUT.mkdir(exist_ok=True)
    for v in VARIANTS:export_variant(v)
    make_comparison()
    make_reference_comparison()
    pose_data={}
    for v in VARIANTS:
        pose_data[v.name]={}
        for pose in POSES:
            mats=matrices(v,pose)
            pose_data[v.name][pose]={k:np.round(m,8).tolist() for k,m in mats.items()}
    (OUT/"poses.json").write_text(json.dumps(pose_data,separators=(',',':')))
    print("done",flush=True)


if __name__=="__main__":main()
