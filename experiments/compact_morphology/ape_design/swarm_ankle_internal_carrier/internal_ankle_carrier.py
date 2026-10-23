"""Source-sized 50 cm internal-link ankle carrier against the exact ape shell.

This bounded topology replaces the rejected 20T ankle belt with a 10 mm
parallel-crank link.  All dimensions are mm.  The source cuff is locally
recessed only where a physical shaft, bearing carrier, rod or foot horn needs
to move; its original outer silhouette is not enlarged.  The exact BRep gate
is evaluated before any ankle-pitch pose screen.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import build123d as b
import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/"artifacts/internal-ankle-carrier-20261005"
sys.path.insert(0,str(HERE.parent/"swarm_rated_leg_skin_v2"))
from skin_v2_case import integration_shapes
sys.path.insert(0,str(HERE.parent/"swarm_rated_hip_ankle"))
from rated_hip_ankle_case import ankle_pitch, cylinder_x, maybe_intersection, mesh, shell_shapes, vol

K=500/350
OUTPUT=(46*K,4*K,24*K)
MOTOR_OUTPUT=(75.0,OUTPUT[1],OUTPUT[2]+31.5)
CUFF_CENTRE=(46*K,3*K,24*K)
CUFF_RADII=(19*K,18*K,5*K)
NTN_BC5_9={"bore_mm":5,"od_mm":9,"width_mm":2.5,
           "catalog_static_N":211,"catalog_lbf":40,"conservative_static_N":40*4.4482216153,
           "source":"https://bearingfinder.ntnamericas.com/ja/item/deep-groove-ball-bearings/single-row-radial-ball-bearings/bc5-9"}
PHASES=(-20,-15,-10,-5,0,5,10,15,20)
CUFF_SHIFT_Z=-1.0  # source cuff/foot seam gains 0.602 mm at neutral
SHIN_FRONT_RIM_Z=37.0  # hidden bottom edge; retains one 2 mm-wall shell solid
SHIN_BACK_RIM_Z=37.0


def contact_volume(a,c):
    # OCCT intersection is asymmetric for this exact source shell: the raw
    # back-shin/foot cover is 312.457 mm3 one way and 0 the other way.
    return max(maybe_intersection(a,c),maybe_intersection(c,a))


def rod(theta):
    a=math.radians(theta)
    y=OUTPUT[1]+10*math.cos(a)
    z0=OUTPUT[2]+10*math.sin(a)
    return b.Solid.make_cylinder(1.5,31.5,
        b.Plane(origin=(78,y,z0),z_dir=(0,0,1)))


def crank(output,theta):
    # A compact 2 mm-thick web in the x plane, 3 mm tall at neutral.
    z=output[2]
    arm=b.Pos(78,output[1]+5,z)*b.Box(2,10,3)
    pin=b.Solid.make_cylinder(2,2,b.Plane(origin=(77,output[1]+10,z),z_dir=(1,0,0)))
    return arm.fuse(pin).rotate(b.Axis(output,(1,0,0)),theta)


def fixed_parts():
    motor=ankle_pitch(6)["parts"]["motor_case"]
    parts={"motor_case":motor}
    for label,x in (("A",55),("B",70)):
        parts["bearing_"+label]=cylinder_x(4.5,x,x+2.5,OUTPUT[1],OUTPUT[2])\
            -cylinder_x(2.5,x-.1,x+2.6,OUTPUT[1],OUTPUT[2])
        parts["seat_"+label]=cylinder_x(6,x-.25,x+2.75,OUTPUT[1],OUTPUT[2])\
            -cylinder_x(4.5,x-.35,x+2.85,OUTPUT[1],OUTPUT[2])
        parts["web_"+label]=b.Pos(x+1.25,OUTPUT[1]+6,OUTPUT[2]+9.5)*b.Box(2.5,4,19)
    parts["back_rail"]=b.Pos(63.75,22,52)*b.Box(20,24,3)
    parts["motor_saddle"]=b.Pos(58,OUTPUT[1],54.5)*b.Box(28,14,3)
    return parts


def moving_parts(theta):
    turn=b.Axis(OUTPUT,(1,0,0))
    parts={"output_shaft":cylinder_x(2.5,47,82,OUTPUT[1],OUTPUT[2]),
           "output_crank":crank(OUTPUT,theta),
           "motor_crank":crank(MOTOR_OUTPUT,theta),
           "motor_spline_adapter_reserve":cylinder_x(3,74.5,78,MOTOR_OUTPUT[1],MOTOR_OUTPUT[2]),
           "rod":rod(theta)}
    for label,x in (("A",49),("B",74)):
        horn=b.Pos(x+1.25,OUTPUT[1],OUTPUT[2]-12)*b.Box(2.5,7,24)
        parts["foot_horn_"+label]=horn.rotate(turn,theta)
    foot_frame=b.Pos(63.75,OUTPUT[1],9)*b.Box(29,8,12)
    parts["foot_frame"]=foot_frame.rotate(turn,theta)
    for label,x,ymax in (("A",50.25,34),("B",75.25,36)):
        tab=b.Pos(x,(8+ymax)/2,9)*b.Box(2.5,ymax-8,8)
        parts["foot_tab_"+label]=tab.rotate(turn,theta)
    return parts


def cuff_outer():
    return b.Solid.make_sphere(1).scale(CUFF_RADII).translate(CUFF_CENTRE)


def clearance_cutters(fixed,poses):
    """Dimensioned recess keep-outs, not Boolean subtraction of the reported parts."""
    cutters=[]
    for label,x in (("A",55),("B",70)):
        cutters.append(cylinder_x(6.5,x-.75,x+3.25,OUTPUT[1],OUTPUT[2]))
        web=fixed["web_"+label]
        bb=web.bounding_box()
        cutters.append(b.Pos((bb.min.X+bb.max.X)/2,(bb.min.Y+bb.max.Y)/2,(bb.min.Z+bb.max.Z)/2)
            *b.Box(bb.size.X+1,bb.size.Y+1,bb.size.Z+1))
    cutters.append(cylinder_x(3,46.5,82.5,OUTPUT[1],OUTPUT[2]))
    for angle in poses:
        phase=moving_parts(angle)
        for name in ("rod","output_crank","foot_horn_A","foot_horn_B"):
            shape=phase[name]
            bb=shape.bounding_box()
            cutters.append(b.Pos((bb.min.X+bb.max.X)/2,(bb.min.Y+bb.max.Y)/2,(bb.min.Z+bb.max.Z)/2)
                *b.Box(bb.size.X+1,bb.size.Y+1,bb.size.Z+1))
    return cutters


def revised_cuff(source_cuff,fixed,poses=(0,)):
    modified=source_cuff
    for cutter in clearance_cutters(fixed,poses):
        if maybe_intersection(modified,cutter)>.05:
            modified=modified.cut(cutter)
    return modified


def trim_shin_rim(source_shell,edge_z):
    # A planar end trim removes only the pre-existing boot/front-shin overlap.
    # Intersect returns a ShapeList, so preserve the single resulting shell.
    keep=b.Pos(0,0,(edge_z+500)/2)*b.Box(1000,1000,500-edge_z)
    pieces=source_shell.intersect(keep)
    if len(pieces)!=1 or len(pieces[0].solids())!=1:
        raise RuntimeError("front-shin hidden rim trim disconnected the shell")
    return pieces[0]


def adjacent_outer():
    outer,_=shell_shapes(500,2)
    outer["ankle_cuff_R"]=(cuff_outer(),None)
    return outer


def outside(shape,outer):
    left=shape
    for key in ("shin_R","ankle_cuff_R","boot_R"):
        if contact_volume(left,outer[key][0])>.05:
            left=left.cut(outer[key][0])
    return round(vol(left),3)


def screen_neutral(case,fixed,moving,cuff,outer,raw_source_skin):
    skin={**case["revised_skin"],"R_ankle_cuff":cuff}
    local={n:s for n,s in skin.items() if n.startswith(("R_shin_","R_ankle_cuff","R_foot_","R_sole","R_toe_"))}
    allowed_contact={"back_rail":{"R_shin_back"},
                     "foot_tab_A":{"R_foot_dorsal"},"foot_tab_B":{"R_foot_dorsal"}}
    rows=[]
    for group,parts in (("fixed",fixed),("moving",moving)):
        for name,shape in parts.items():
            h={n:round(v,3) for n,s in local.items()
               if (v:=contact_volume(shape,s))>.05}
            accepted={n:v for n,v in h.items() if n in allowed_contact.get(name,set())}
            blocked={n:v for n,v in h.items() if n not in allowed_contact.get(name,set())}
            gaps={n:shape.distance(w) for n,w in local.items() if n not in allowed_contact.get(name,set())}
            rows.append(dict(group=group,part=name,skin_blockers_mm3=blocked,
                             intended_mount_contacts_mm3=accepted,
                             min_other_skin_gap_mm=round(min(gaps.values()),3),
                             outside_source_outer_mm3=outside(shape,outer)))
    motor=fixed["motor_case"]
    self_clashes={}
    for n in ("back_rail","web_A","web_B","seat_A","seat_B"):
        v=contact_volume(motor,fixed[n])
        if v>.05:self_clashes["motor_case:"+n]=round(v,3)
    combined={**fixed,**moving}
    contacts=(("seat_A","web_A"),("seat_B","web_B"),
              ("web_A","back_rail"),("web_B","back_rail"),
              ("back_rail","motor_saddle"),("motor_saddle","motor_case"),
              ("motor_case","motor_spline_adapter_reserve"),
              ("motor_spline_adapter_reserve","motor_crank"),
              ("motor_crank","rod"),("rod","output_crank"),
              ("output_crank","output_shaft"),
              ("output_shaft","foot_horn_A"),("output_shaft","foot_horn_B"),
              ("foot_horn_A","foot_frame"),("foot_horn_B","foot_frame"),
              ("foot_frame","foot_tab_A"),("foot_frame","foot_tab_B"))
    connected={a+":"+c:round(contact_volume(combined[a],combined[c]),3) for a,c in contacts}
    fits={name:round(a.distance(c),4) for name,a,c in (
        ("bearing_A:seat_A",fixed["bearing_A"],fixed["seat_A"]),
        ("bearing_B:seat_B",fixed["bearing_B"],fixed["seat_B"]),
        ("shaft:bearing_A",moving["output_shaft"],fixed["bearing_A"]),
        ("shaft:bearing_B",moving["output_shaft"],fixed["bearing_B"]))}
    foot_skin={n:s for n,s in skin.items() if n.startswith(("R_foot_","R_sole","R_toe_"))}
    raw_back=raw_source_skin["R_shin_back"]
    raw_dorsal=raw_source_skin["R_foot_dorsal"]
    boolean_order_example=dict(pair="raw R_shin_back / R_foot_dorsal",
        shin_intersect_foot_mm3=round(maybe_intersection(raw_back,raw_dorsal),3),
        foot_intersect_shin_mm3=round(maybe_intersection(raw_dorsal,raw_back),3))
    cuff_to_foot={n:dict(overlap_mm3=round(contact_volume(cuff,s),3),
                         gap_mm=round(cuff.distance(s),3)) for n,s in foot_skin.items()}
    source_front=raw_source_skin["R_shin_front"]
    revised_front=skin["R_shin_front"]
    raw_source_pairs={n:dict(overlap_mm3=round(contact_volume(source_front,s),3),
                             gap_mm=round(source_front.distance(s),3)) for n,s in foot_skin.items()}
    revised_source_pairs={n:dict(overlap_mm3=round(contact_volume(revised_front,s),3),
                                 gap_mm=round(revised_front.distance(s),3)) for n,s in foot_skin.items()}
    raw_fixed_skin_to_foot={}
    revised_fixed_skin_to_foot={}
    for fixed_name in ("R_shin_front","R_shin_back","R_ankle_cuff"):
        original=raw_source_skin[fixed_name]
        revised=skin[fixed_name]
        for foot_name,foot_shape in foot_skin.items():
            key=fixed_name+":"+foot_name
            raw_fixed_skin_to_foot[key]=dict(overlap_mm3=round(contact_volume(original,foot_shape),3),
                                             gap_mm=round(original.distance(foot_shape),3))
            revised_fixed_skin_to_foot[key]=dict(overlap_mm3=round(contact_volume(revised,foot_shape),3),
                                                 gap_mm=round(revised.distance(foot_shape),3))
    local_clear=all(not r["skin_blockers_mm3"] and r["min_other_skin_gap_mm"]>=.5
                    and r["outside_source_outer_mm3"]<=.05 for r in rows) and not self_clashes \
                and all(v>.05 for v in connected.values()) and all(v<=.05 for v in fits.values()) \
                and all(any(r["part"]==part and r["intended_mount_contacts_mm3"] for r in rows)
                        for part in ("back_rail","foot_tab_A","foot_tab_B")) \
                and all(v["overlap_mm3"]<=.05 and v["gap_mm"]>=.5 for v in cuff_to_foot.values()) \
                and all(v["overlap_mm3"]<=.05 and v["gap_mm"]>=.5 for v in revised_fixed_skin_to_foot.values())
    return dict(parts=rows,motor_support_clashes_mm3=self_clashes,
                connected_contacts_mm3=connected,nominal_bearing_fit_gaps_mm=fits,
                cuff_to_moving_foot_skin=cuff_to_foot,
                raw_source_shin_front_to_foot_skin=raw_source_pairs,
                revised_shin_front_to_foot_skin=revised_source_pairs,
                raw_source_fixed_skin_to_foot_skin=raw_fixed_skin_to_foot,
                revised_fixed_skin_to_foot_skin=revised_fixed_skin_to_foot,
                boolean_operand_order_example=boolean_order_example,
                hidden_front_rim_trim_removed_mm3=round(vol(source_front)-vol(revised_front),3),
                hidden_back_rim_trim_removed_mm3=round(vol(raw_source_skin["R_shin_back"])-vol(skin["R_shin_back"]),3),
                revised_front_solids_after=len(revised_front.solids()),
                revised_back_solids_after=len(skin["R_shin_back"].solids()),
                cuff_outside_approved_outer_mm3=outside(cuff,outer),
                cuff_solids_after=len(cuff.solids()),
                removed_cuff_mm3=round(vol(case["revised_skin"]["R_ankle_cuff"])-vol(cuff),3),
                static_clear=local_clear)


def screen_poses(case,fixed,cuff):
    stationary_skin={n:s for n,s in case["revised_skin"].items() if n.startswith("R_shin_")}
    stationary_skin["R_ankle_cuff"]=cuff
    structure={n:s for n,s in fixed.items() if n!="motor_case"}
    stationary={**stationary_skin,**structure,"motor_case":fixed["motor_case"]}
    foot_skin={n:s for n,s in case["revised_skin"].items()
               if n.startswith(("R_foot_","R_toe_","R_sole"))}
    turn=b.Axis(OUTPUT,(1,0,0))
    rows=[]
    for angle in PHASES:
        hardware=moving_parts(angle)
        foot={n:s.rotate(turn,angle) for n,s in foot_skin.items()}
        parts={**hardware,**foot}
        violations=[]
        for name,shape in parts.items():
            for target,solid in stationary.items():
                # Nominal bearing/seat capture, motor spline engagement and
                # the fixed foot mounts are assembled contacts, not sweep hits.
                if name=="output_shaft" and target.startswith(("bearing_","seat_")):continue
                if name=="motor_spline_adapter_reserve" and target=="motor_case":continue
                if name=="motor_crank" and target=="motor_case":continue
                value=contact_volume(shape,solid)
                if value>.05:
                    violations.append(dict(moving=name,fixed=target,overlap_mm3=round(value,3)))
        rows.append(dict(pitch_deg=angle,collisions=violations,
                         covered_pose_clear=not violations))
        print("pose",angle,"collisions",len(violations),flush=True)
    return rows


def scene_for(skin,cuff,fixed=None,moving=None,angle=0,opaque=True):
    scene=trimesh.Scene()
    turn=b.Axis(OUTPUT,(1,0,0))
    for name,shape in skin.items():
        if not name.startswith(("R_shin_","R_foot_","R_sole","R_toe_","R_ankle_cuff")):
            continue
        if name=="R_ankle_cuff":shape=cuff
        elif angle and name.startswith(("R_foot_","R_sole","R_toe_")):
            shape=shape.rotate(turn,angle)
        color=(102,110,102,255) if name=="R_ankle_cuff" else (225,218,204,255)
        if not opaque:color=(*color[:3],90)
        mesh(scene,name,shape,color)
    if fixed:
        for name,shape in fixed.items():mesh(scene,"fixed_"+name,shape,(60,96,121,255))
    if moving:
        for name,shape in moving.items():mesh(scene,"moving_"+name,shape,(184,118,67,255))
    return scene


def panel(scene,view):
    geo=scene.to_geometry()
    vertices=np.asarray(geo.vertices)
    faces=np.asarray(geo.faces)
    normals=np.asarray(geo.face_normals)
    tris=vertices[faces]
    if view=="front":
        u=tris[:,:,0]-65;depth=tris[:,:,1].mean(axis=1)
        facing=-normals[:,1];order=np.argsort(-depth)
    else:
        u=-tris[:,:,1];depth=tris[:,:,0].mean(axis=1)
        facing=normals[:,0];order=np.argsort(depth)
    v=tris[:,:,2]-59
    size=620
    xy=np.stack((size/2+u*4.5,size/2-v*4.5),axis=2)
    colors=np.asarray(geo.visual.face_colors)[:,:3]
    canvas=Image.new("RGB",(size,size),(220,217,207))
    draw=ImageDraw.Draw(canvas)
    light=np.array([.35,-.7,.63]);light/=np.linalg.norm(light)
    for i in order:
        if facing[i]<=.01:continue
        shade=.68+.32*max(0,float(normals[i]@light))
        draw.polygon([tuple(p) for p in xy[i]],fill=tuple(int(np.clip(c*shade,0,255)) for c in colors[i]))
    return canvas


def export_views(case,raw_source_skin,fixed,cuff):
    revised=case["revised_skin"]
    source=scene_for(raw_source_skin,raw_source_skin["R_ankle_cuff"])
    neutral=scene_for(revised,cuff)
    flex10=scene_for(revised,cuff,angle=10)
    engineering=scene_for(revised,cuff,fixed,moving_parts(0),opaque=False)
    variants=(("Original 50 cm shell · raw contact",source),
              ("Hidden rim + internal carrier · neutral",neutral),
              ("10° pitch · cuff collision",flex10))
    sheet=Image.new("RGB",(620*3,620*2+64),(248,246,240))
    draw=ImageDraw.Draw(sheet)
    for col,(label,scene) in enumerate(variants):
        scene.export(OUT/("source.glb" if col==0 else "neutral.glb" if col==1 else "pitch-10-collision.glb"))
        for row,view in enumerate(("front","side")):
            draw.text((col*620+14,row*(620+32)+8),label+" · "+view,fill=(35,50,43))
            sheet.paste(panel(scene,view),(col*620,row*(620+32)+32))
    sheet.save(OUT/"matched-opaque.png")
    engineering.export(OUT/"carrier-neutral-engineering.glb")
    b.export_step(b.Compound(children=[*fixed.values(),*moving_parts(0).values(),
                                       revised["R_shin_front"],cuff,revised["R_foot_dorsal"]]),
                  str(OUT/"internal-ankle-carrier-500.step"))
    shutil.copyfile(HERE/"viewer.html",OUT/"index.html")


def candidate_shapes():
    """Return separated source skin, locally revised skin, and load-path parts.

    Consumers must retain the neutral/motion verdict; this is not an accepted
    articulated ankle or a rated transmission assembly.
    """
    case=integration_shapes(500)
    raw_source_skin=case["revised_skin"]
    case["revised_skin"]={**raw_source_skin,
        "R_shin_front":trim_shin_rim(raw_source_skin["R_shin_front"],SHIN_FRONT_RIM_Z),
        "R_shin_back":trim_shin_rim(raw_source_skin["R_shin_back"],SHIN_BACK_RIM_Z)}
    fixed=fixed_parts()
    cuff=revised_cuff(raw_source_skin["R_ankle_cuff"].translate((0,0,CUFF_SHIFT_Z)),fixed,(0,))
    return dict(case=case,raw_source_skin=raw_source_skin,fixed=fixed,
                moving_neutral=moving_parts(0),cuff=cuff,
                output_axis_mm=OUTPUT,motor_output_axis_mm=MOTOR_OUTPUT)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selected=candidate_shapes()
    case=selected["case"]
    raw_source_skin=selected["raw_source_skin"]
    fixed=selected["fixed"]
    moving=selected["moving_neutral"]
    cuff=selected["cuff"]
    outer=adjacent_outer()
    neutral=screen_neutral(case,fixed,moving,cuff,outer,raw_source_skin)
    poses=screen_poses(case,fixed,cuff) if neutral["static_clear"] else []
    covered_motion_clear=bool(poses) and all(row["covered_pose_clear"] for row in poses)
    report=dict(height_mm=500,wall_mm=2,datum_mm=OUTPUT,
                hidden_front_rim_trim_z_mm=SHIN_FRONT_RIM_Z,
                hidden_back_rim_trim_z_mm=SHIN_BACK_RIM_Z,
                cuff_relocation_z_mm=CUFF_SHIFT_Z,
                motor_output_mm=MOTOR_OUTPUT,source_cuff_volume_mm3=round(vol(case["revised_skin"]["R_ankle_cuff"]),3),
                motor="Feetech ST3215-C018; source-sized 45.2 x 24.7 x 35 mm, 55 g, rated 0.981 Nm at 12 V",
                transmission="two 10 mm cranks and 31.5 mm center rod, geometric 1:1 parallelogram; full assembly rating unknown",
                bearing=NTN_BC5_9,neutral=neutral,
                motion=poses if poses else "not_run_static_gate_failed",
                covered_motion_clear=covered_motion_clear,
                sources=dict(ST3215="https://www.feetechrc.com/525603.html",NTN_BC5_9=NTN_BC5_9["source"]),
                verdict=("Neutral source-sized internal carrier clears after hidden front- and back-shin rim trims; the raw source shell intersects its foot cover at neutral. Covered motion fails at 5 degrees from cuff/foot-horn contact and at other sampled phases from foot/shin/cuff contacts. This is a measured packaging result, not an accepted moving ankle." if neutral["static_clear"] and not covered_motion_clear else
                         "Static gate failed; moving poses were not screened." if not neutral["static_clear"] else
                         "Sampled motion clear; cyclic transmission rating and thermal path remain unproven."))
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    export_views(case,raw_source_skin,fixed,cuff)
    print("neutral clear",neutral["static_clear"],"cuff solids",neutral["cuff_solids_after"],
          "removed cuff",neutral["removed_cuff_mm3"])
    for row in neutral["parts"]:
        if row["skin_blockers_mm3"] or row["min_other_skin_gap_mm"]<.5 or row["outside_source_outer_mm3"]>.05:
            print(row["part"],row["skin_blockers_mm3"],row["min_other_skin_gap_mm"],row["outside_source_outer_mm3"])
    print("motor_support_clashes",neutral["motor_support_clashes_mm3"])


if __name__=="__main__":main()
