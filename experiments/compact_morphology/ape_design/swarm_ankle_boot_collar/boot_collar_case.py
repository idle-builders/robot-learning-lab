"""Bounded moving upper boot collar for the source-sized 50 cm ankle carrier.

The ST3215, two BC5-9 output bearings, shaft and parallel-crank link come from
the committed neutral carrier.  A 2 mm upper lip is fused to its cuff and
rotates with the foot.  Two swept support-web slots and a rod slot are cut in
collar coordinates.  Exact BRep checks take the larger Boolean operand order.
All dimensions are mm.  This is a clearance study, not a cyclic-load proof.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import build123d as b
import trimesh
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/"artifacts/ankle-boot-collar-20261005"
sys.path.insert(0,str(HERE.parent/"swarm_ankle_internal_carrier"))
from internal_ankle_carrier import (CUFF_CENTRE, CUFF_RADII, OUTPUT,
    adjacent_outer, candidate_shapes, contact_volume, cylinder_x, mesh, moving_parts,
    panel, trim_shin_rim, vol)

PHASES=(-20,-10,-5,0,5,10,20)
COLLAR_TOP_RADIUS_Z=12.0
FRONT_HEM_Z=48.0
BACK_HEM_Z=47.0
SLOT_GAP=0.6


def swept_box(parts,axis,phases=PHASES,margin=SLOT_GAP):
    """Conservative local frame AABB of a fixed or differently moving part."""
    boxes=[shape.rotate(axis,-angle).bounding_box() for angle,shape in parts]
    lo=[min(getattr(box.min,k) for box in boxes)-margin for k in "XYZ"]
    hi=[max(getattr(box.max,k) for box in boxes)+margin for k in "XYZ"]
    box=b.Pos(*[(a+c)/2 for a,c in zip(lo,hi)])*b.Box(*[c-a for a,c in zip(lo,hi)])
    return box,dict(min_mm=[round(a,3) for a in lo],max_mm=[round(c,3) for c in hi])


def upper_lip():
    outer=b.Solid.make_sphere(1).scale((CUFF_RADII[0],CUFF_RADII[1],COLLAR_TOP_RADIUS_Z)).translate(CUFF_CENTRE)
    inner=b.Solid.make_sphere(1).scale((CUFF_RADII[0]-2,CUFF_RADII[1]-2,COLLAR_TOP_RADIUS_Z-2)).translate(CUFF_CENTRE)
    keep=b.Pos(0,0,(CUFF_CENTRE[2]+500)/2)*b.Box(1000,1000,500-CUFF_CENTRE[2])
    pieces=(outer-inner).intersect(keep)
    if len(pieces)!=1:return b.Compound(children=list(pieces))
    return pieces[0]


def collar_with_swept_slots(base,fixed):
    axis=b.Axis(OUTPUT,(1,0,0))
    collar=base.fuse(upper_lip())
    raw_volume=vol(collar)
    slots={}
    for name in ("web_A","web_B","rod"):
        parts=[(angle,fixed[name] if name!="rod" else moving_parts(angle)["rod"])
               for angle in PHASES]
        cutter,bounds=swept_box(parts,axis)
        before=vol(collar)
        collar=collar.cut(cutter)
        slots[name]=dict(**bounds,removed_material_mm3=round(before-vol(collar),3))
    shaft_clearance=cylinder_x(3.1,46.5,82.5,OUTPUT[1],OUTPUT[2])
    before=vol(collar)
    collar=collar.cut(shaft_clearance)
    slots["output_shaft"]=dict(radius_mm=3.1,
        removed_material_mm3=round(before-vol(collar),3))
    return collar,dict(upper_lip_added_before_slots_mm3=round(raw_volume-vol(base),3),
                       swept_slot_bounds=slots,
                       removed_by_slots_mm3=round(raw_volume-vol(collar),3),
                       collar_solids=len(collar.solids()),collar_valid=collar.is_valid)


def collar_mounts():
    # Interior bridges from each foot horn to the moving collar shell.
    return {"collar_tab_A":b.Pos(50.25,16,30)*b.Box(2.5,16,2),
            "collar_tab_B":b.Pos(75.25,18,30)*b.Box(2.5,20,2)}


def case_shapes():
    parent=candidate_shapes()
    raw=parent["raw_source_skin"]
    skin={**parent["case"]["revised_skin"],
          "R_shin_front":trim_shin_rim(raw["R_shin_front"],FRONT_HEM_Z),
          "R_shin_back":trim_shin_rim(raw["R_shin_back"],BACK_HEM_Z)}
    collar,construction=collar_with_swept_slots(parent["cuff"],parent["fixed"])
    moving_boot=collar.fuse(skin["R_foot_dorsal"])
    construction["moving_boot_solids"]=len(moving_boot.solids())
    construction["moving_boot_valid"]=moving_boot.is_valid
    construction["collar_to_foot_dorsal_fusion_mm3"]=round(contact_volume(collar,skin["R_foot_dorsal"]),3)
    return dict(raw_skin=raw,skin=skin,collar=moving_boot,collar_core=collar,fixed=parent["fixed"],
                mounts=collar_mounts(),construction=construction)


def pose_outer(angle):
    axis=b.Axis(OUTPUT,(1,0,0))
    outer=adjacent_outer()
    for key in ("boot_R","ankle_cuff_R"):
        outer[key]=(outer[key][0].rotate(axis,angle),None)
    return outer


def exterior_residual(shape,outer):
    left=shape
    for key in ("shin_R","ankle_cuff_R","boot_R"):
        if contact_volume(left,outer[key][0])>.05:
            left=left.cut(outer[key][0])
    return left


def pose_geometry(case,angle):
    axis=b.Axis(OUTPUT,(1,0,0))
    foot={n:s.rotate(axis,angle) for n,s in case["skin"].items()
          if n.startswith(("R_toe_","R_sole"))}
    hardware=moving_parts(angle)
    mounts={n:s.rotate(axis,angle) for n,s in case["mounts"].items()}
    return dict(foot=foot,hardware=hardware,mounts=mounts,
                collar=case["collar"].rotate(axis,angle))


def screen_pose(case,angle):
    moving=pose_geometry(case,angle)
    skin_fixed={n:case["skin"][n] for n in ("R_shin_front","R_shin_back")}
    stationary={**skin_fixed,**case["fixed"]}
    moving_all={**moving["foot"],**moving["hardware"],**moving["mounts"],
                "moving_collar":moving["collar"]}
    intentional={("output_shaft","bearing_A"),("output_shaft","bearing_B"),
                 ("output_shaft","seat_A"),("output_shaft","seat_B"),
                 ("motor_spline_adapter_reserve","motor_case"),
                 ("motor_crank","motor_case")}
    blockers=[]
    closest=[]
    for name,shape in moving_all.items():
        for fixed_name,fixed_shape in stationary.items():
            if (name,fixed_name) in intentional:continue
            hit=contact_volume(shape,fixed_shape)
            gap=shape.distance(fixed_shape)
            if hit>.05 or gap<.5-1e-5:
                blockers.append(dict(moving=name,fixed=fixed_name,
                                     overlap_mm3=round(hit,3),gap_mm=round(gap,3)))
            if gap<1:closest.append(dict(moving=name,fixed=fixed_name,gap_mm=round(gap,3)))
    collar=moving["collar"]
    collar_internal=[]
    for name,shape in {**moving["foot"],**moving["hardware"]}.items():
        if name in ("foot_tab_A","foot_tab_B"):continue
        hit=contact_volume(collar,shape)
        gap=collar.distance(shape)
        if hit>.05 or gap<.5-1e-5:
            collar_internal.append(dict(part=name,overlap_mm3=round(hit,3),gap_mm=round(gap,3)))
    mounts={}
    for side in ("A","B"):
        tab=moving["mounts"]["collar_tab_"+side]
        mounts[side]=dict(tab_to_collar_mm3=round(contact_volume(tab,collar),3),
                          tab_to_foot_horn_mm3=round(contact_volume(tab,moving["hardware"]["foot_horn_"+side]),3))
    outer=pose_outer(angle)
    axis=b.Axis(OUTPUT,(1,0,0))
    core_exterior=vol(exterior_residual(case["collar_core"].rotate(axis,angle),outer))
    dorsal_exterior=vol(exterior_residual(case["skin"]["R_foot_dorsal"].rotate(axis,angle),outer))
    # A fused boot BRep returns a false 3595 mm3 remainder at 0/+5 degrees,
    # though each constituent has zero remainder.  Sum the independently
    # screened constituents as a conservative union upper bound.
    exterior=core_exterior+dorsal_exterior
    joined_brep_exterior=(round(vol(exterior_residual(collar,outer)),3)
                          if angle in (0,5) else None)
    return dict(pitch_deg=angle,blockers=blockers,near_gaps_under_1mm=closest,
                collar_to_moving_part_blockers=collar_internal,
                mount_contacts=mounts,
                collar_outside_rotated_source_outer_mm3=round(exterior,3),
                exterior_components_mm3=dict(collar_core=round(core_exterior,3),
                    dorsal_foot=round(dorsal_exterior,3),
                    joined_brep_diagnostic=joined_brep_exterior),
                hard_clear=not blockers and not collar_internal and
                    all(min(m.values())>.05 for m in mounts.values()),
                silhouette_contained=exterior<=.05)


def make_scene(case,angle,engineering=False,source=False):
    scene=trimesh.Scene()
    axis=b.Axis(OUTPUT,(1,0,0))
    skin=case["raw_skin"] if source else case["skin"]
    for name,shape in skin.items():
        if not name.startswith(("R_shin_","R_foot_","R_toe_","R_sole")):continue
        if not source and name=="R_foot_dorsal":continue
        if angle and name.startswith(("R_foot_","R_toe_","R_sole")):
            shape=shape.rotate(axis,angle)
        mesh(scene,name,shape,(225,218,204,110 if engineering else 255))
    collar=(case["raw_skin"]["R_ankle_cuff"].rotate(axis,angle)
            if source else case["collar"].rotate(axis,angle))
    mesh(scene,"collar",collar,
         ((100,108,99,100) if engineering else (225,218,204,255) if not source else (100,108,99,255)))
    if engineering and not source:
        for name,shape in case["fixed"].items():mesh(scene,"fixed_"+name,shape,(60,96,121,255))
        for name,shape in moving_parts(angle).items():mesh(scene,"moving_"+name,shape,(184,118,67,255))
        for name,shape in case["mounts"].items():mesh(scene,name,shape.rotate(axis,angle),(202,137,70,255))
    return scene


def export_views(case,report):
    variants=(("source",0,"Original source shell"),
              ("neutral",0,"Moving upper collar · neutral"),
              ("source-minus-20",-20,"Original source shell · −20°"),
              ("pitch-minus-20",-20,"Moving collar · −20°"),
              ("source-plus-20",20,"Original source shell · +20°"),
              ("pitch-plus-20",20,"Moving collar · +20°"))
    sheet=Image.new("RGB",(620*len(variants),620*2+64),(248,246,240))
    draw=ImageDraw.Draw(sheet)
    for col,(name,angle,label) in enumerate(variants):
        scene=make_scene(case,angle,source=name.startswith("source"))
        scene.export(OUT/(name+".glb"))
        for row,view in enumerate(("front","side")):
            draw.text((col*620+14,row*(620+32)+8),label+" · "+view,fill=(35,50,43))
            sheet.paste(panel(scene,view),(col*620,row*(620+32)+32))
    sheet.save(OUT/"matched-opaque.png")
    make_scene(case,0,engineering=True).export(OUT/"engineering-neutral.glb")
    b.export_step(b.Compound(children=[case["skin"]["R_shin_front"],
        case["skin"]["R_shin_back"],case["collar"],*case["fixed"].values(),
        *case["mounts"].values(),*moving_parts(0).values()]),
        str(OUT/"boot-collar-500.step"))
    shutil.copyfile(HERE/"viewer.html",OUT/"index.html")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    case=case_shapes()
    skin=case["skin"]
    source=case["raw_skin"]
    neutral=screen_pose(case,0)
    poses=[neutral]
    if neutral["hard_clear"]:
        for angle in PHASES:
            if angle==0:continue
            row=screen_pose(case,angle)
            poses.append(row)
            print("pose",angle,"hard blockers",len(row["blockers"])+len(row["collar_to_moving_part_blockers"]),
                  "exterior",row["collar_outside_rotated_source_outer_mm3"],flush=True)
    poses.sort(key=lambda row:row["pitch_deg"])
    report=dict(height_mm=500,source_shell_wall_mm=2,output_axis_mm=OUTPUT,
        collar_top_radius_z_mm=COLLAR_TOP_RADIUS_Z,
        fixed_hem_bottom_z_mm=dict(front=FRONT_HEM_Z,back=BACK_HEM_Z),
        geometry=dict(**case["construction"],
            front_shin_removed_from_raw_mm3=round(vol(source["R_shin_front"])-vol(skin["R_shin_front"]),3),
            back_shin_removed_from_raw_mm3=round(vol(source["R_shin_back"])-vol(skin["R_shin_back"]),3),
            front_shin_solids=len(skin["R_shin_front"].solids()),
            back_shin_solids=len(skin["R_shin_back"].solids()),
            collar_volume_mm3=round(vol(case["collar"]),3)),
        poses=poses,
        appearance_assessment="Matched opaque neutral and ±20 degree views show an open shin-to-boot air gap; provisional appearance rejection pending independent review.",
        verdict=("Neutral hard gate failed; representative moving sweep was not run." if not neutral["hard_clear"] else
                 "The sampled hard-contact gate clears, but the moving collar protrudes beyond the rotated source exterior at ±20 degrees and leaves a visible open ankle seam; this is not a covered silhouette pass." if any(not p["silhouette_contained"] for p in poses) else
                 "Sampled hard contacts and source exterior clear; appearance and cyclic link/thermal proof remain independent gates."),
        sources=dict(ST3215="https://www.feetechrc.com/525603.html",
                     NTN_BC5_9="https://bearingfinder.ntnamericas.com/ja/item/deep-groove-ball-bearings/single-row-radial-ball-bearings/bc5-9"))
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    export_views(case,report)
    print("neutral",neutral["hard_clear"],"collar solids",report["geometry"]["collar_solids"],flush=True)
    print("verdict",report["verdict"],flush=True)


if __name__=="__main__":main()
