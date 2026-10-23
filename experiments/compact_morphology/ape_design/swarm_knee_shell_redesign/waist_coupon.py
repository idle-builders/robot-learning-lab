"""Two local 50 cm knee-shell contour trials around the supported offset pivot.

Only the lower thigh, cuff and proximal shin profiles change. Existing motor,
belt, bearings, shaft, yoke, bridge and shin pad are reused unchanged. Exact
0/60/115 B-rep cuts and contacts are the workflow check; no broad test suite.
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
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments.compact_morphology.ape_design.swarm_knee_flex_closure.closure_coupon import (
    AXIS, hard_parts, overlap,
)
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import turn
from experiments.compact_morphology.ape_design.swarm_rated_leg_drive.rated_leg_coupon import mesh
from experiments.sketch_ape_shell.build import Study

ANGLES=(0,60,115)
WALL=2.0
MARGIN=.5
GAP=.65

# Output millimetres in the saved 500 mm sketch coordinate frame. Rows above
# the knee (thigh) and below it (shin) preserve the source loft stations.
THIGH_UPPER=[
    (177.143,60.0,4.286,38.571,35.714),
    (207.143,51.5,5.714,32.929,32.857),
    (224.286,47.143,5.714,22.857,25.714),
    (232.857,44.286,5.714,11.429,14.286),
]
SHIN_LOWER=[
    (32.857,65.714,4.286,30.0,28.571),
    (51.429,65.714,4.286,32.857,32.857),
    (81.429,64.286,5.714,31.429,32.857),
    (107.143,61.429,5.714,27.143,28.571),
    (118.071,60.0,4.286,22.857,24.286),
]
CASES={
    "waist_low":dict(
        thigh=[(137,60,4.3,23,23),(150,61,5,26,27)],
        cuff=((60,5,134.5),(23,23,7.5)),
        shin=[(124,60,5,19.5,20.5),(130,60,6,17.5,18.5)]),
    "waist_high":dict(
        thigh=[(147,60,8,24,22),(159,61,8,27,27)],
        cuff=((60,8,143.5),(22,21,7.5)),
        shin=[(127,60,6,19,20),(137,60,8,16,17)]),
}


def loft(rows):
    return b.Solid.make_loft([b.Wire.make_ellipse(rx,ry,b.Plane(origin=(x,y,z)))
                              for z,x,y,rx,ry in rows])


def hollow(rows):
    inner=[(z,x,y,rx-WALL,ry-WALL) for z,x,y,rx,ry in rows]
    return loft(rows)-loft(inner)


def split_shell(shape,front_name,back_name,split_y):
    front=b.Pos(0,split_y-100,150)*b.Box(400,200-GAP,400)
    back=b.Pos(0,split_y+100,150)*b.Box(400,200-GAP,400)
    return {front_name:shape & front,back_name:shape & back}


def proposed_shell(case):
    thigh=hollow(case["thigh"]+THIGH_UPPER)
    shin=hollow(SHIN_LOWER+case["shin"])
    study=Study(500,wall=WALL,gap=GAP)
    cuff=study.hollow_ellipsoid(*case["cuff"])
    fixed={**split_shell(thigh,"R_thigh_front","R_thigh_back",4.3),
           "R_knee_cuff":cuff}
    moving=split_shell(shin,"R_shin_front","R_shin_back",4.3)
    return fixed,moving


def volume(shape):
    return sum(s.volume for s in shape.solids()) if shape else 0.0


def cut_fixed(fixed,moving,hardware,moving_hw):
    # Clear every real fixed package envelope, including the belt and bearing
    # rings that the earlier skin coupon did not subtract.
    fixed_cutters={n:b.offset(hardware[n],amount=MARGIN)
                   for n in hardware}
    moving_cutters={n:b.offset(moving_hw[n],amount=MARGIN)
                    for n in ("yoke","bridge")}
    shin_cutters={n:b.offset(shape,amount=MARGIN) for n,shape in moving.items()}
    result={}
    removed={}
    for name,source in fixed.items():
        cut=source
        for cutter in fixed_cutters.values():
            if overlap(cut,cutter)>.05:cut=cut-cutter
        for angle in ANGLES:
            for cutter in (*moving_cutters.values(),*shin_cutters.values()):
                posed=turn(cutter,tuple(AXIS),angle)
                if overlap(cut,posed)>.05:cut=cut-posed
        result[name]=cut
        removed[name]=round(volume(source)-volume(cut),2)
    return result,removed


def force_mesh(shape):
    """Use absolute deflection: OCCT relative meshing misses tiny split faces."""
    BRepTools.Clean_s(shape.wrapped)
    BRepMesh_IncrementalMesh(shape.wrapped,.55,False,.1,True)


def projected_mask(shapes,coordinates):
    """0.5 mm/pixel opaque silhouette comparison of exact BRep-derived meshes."""
    image=Image.new("1",(720,760),0)
    draw=ImageDraw.Draw(image)
    for shape in shapes:
        if not shape or not shape.solids():continue
        force_mesh(shape)
        vv,ff=shape.tessellate(.55,.1)
        xy=np.asarray([[v.X,v.Y,v.Z] for v in vv])[:,coordinates]
        xy[:,0]=(xy[:,0]+150)*2
        xy[:,1]=(xy[:,1]+50)*2
        for face in ff:
            draw.polygon([tuple(p) for p in xy[np.asarray(face)]],fill=1)
    return np.asarray(image,dtype=bool)


def bbox_gap(a,c):
    aa,cc=a.bounding_box(optimal=False),c.bounding_box(optimal=False)
    return math.sqrt(sum(max(getattr(aa.min,q)-getattr(cc.max,q),
                             getattr(cc.min,q)-getattr(aa.max,q),0)**2
                         for q in "XYZ"))


def exact_contact(a,c):
    if bbox_gap(a,c)>0:return 0.0,0.0
    va=volume(a&c)
    vb=volume(c&a)
    return max(va,vb),abs(va-vb)


def pose_checks(fixed,moving,hardware,moving_hw,reference_fixed,reference_shin):
    rows=[]
    fixed_hits=[]
    order_disagreements=[]
    for name,shape in hardware.items():
        for wall_name,wall in fixed.items():
            v,disagreement=exact_contact(shape,wall)
            if v>.05:fixed_hits.append((name,wall_name,round(v,3)))
            if disagreement>.1:order_disagreements.append(("fixed",name,wall_name,round(disagreement,3)))
    for angle in ANGLES:
        print(f"  exact pose {angle}",flush=True)
        posed={n:turn(s,tuple(AXIS),angle) for n,s in moving.items()}
        posed_hw={n:turn(s,tuple(AXIS),angle) for n,s in moving_hw.items()}
        hits=[]
        gaps=[]
        for moving_name,shape in {**posed,**posed_hw}.items():
            for fixed_name,wall in fixed.items():
                v,disagreement=exact_contact(shape,wall)
                if v>.05:hits.append((moving_name,fixed_name,round(v,3)))
                elif bbox_gap(shape,wall)<MARGIN:
                    d=shape.distance_to(wall)
                    if d<MARGIN-.01:gaps.append((moving_name,fixed_name,round(d,3)))
                if disagreement>.1:order_disagreements.append((angle,moving_name,fixed_name,round(disagreement,3)))
        moving_hw_hits=[]
        for name,shape in posed.items():
            for hw_name,hw in hardware.items():
                v,disagreement=exact_contact(shape,hw)
                if v>.05:moving_hw_hits.append((name,hw_name,round(v,3)))
                if disagreement>.1:order_disagreements.append((angle,name,hw_name,round(disagreement,3)))
        source_shapes=[*reference_fixed.values(),
                       *(turn(s,tuple(AXIS),angle) for s in reference_shin.values())]
        new_shapes=[*fixed.values(),*posed.values()]
        print(f"  silhouette pose {angle}",flush=True)
        source_side=projected_mask(source_shapes,(1,2))
        new_side=projected_mask(new_shapes,(1,2))
        source_front=projected_mask(source_shapes,(0,2))
        new_front=projected_mask(new_shapes,(0,2))
        rows.append(dict(angle_deg=angle,moving_fixed_hits=hits,
                         moving_fixed_short_gaps=gaps,fixed_hardware_hits=fixed_hits,
                         moving_skin_fixed_hardware_hits=moving_hw_hits,
                         side_new_area_mm2=round(np.count_nonzero(new_side&~source_side)/4,2),
                         side_lost_area_mm2=round(np.count_nonzero(source_side&~new_side)/4,2),
                         front_new_area_mm2=round(np.count_nonzero(new_front&~source_front)/4,2)))
    return rows,order_disagreements


def export_poses(label,fixed,moving,hardware,moving_hw):
    for angle in ANGLES:
        scene=trimesh.Scene()
        for n,s in fixed.items():
            force_mesh(s)
            mesh(scene,n,s,(230,222,207,255))
        for n,s in moving.items():
            posed=turn(s,tuple(AXIS),angle)
            force_mesh(posed)
            mesh(scene,n,posed,(230,222,207,255))
        scene.export(HERE/f"{label}-{angle}.glb")
        diagnostic=trimesh.Scene()
        for n,s in fixed.items():mesh(diagnostic,n,s,(230,222,207,90))
        for n,s in moving.items():
            posed=turn(s,tuple(AXIS),angle)
            force_mesh(posed)
            mesh(diagnostic,n,posed,(230,222,207,90))
        for n,s in hardware.items():
            force_mesh(s)
            mesh(diagnostic,n,s,(55,100,148,255))
        for n,s in moving_hw.items():
            posed=turn(s,tuple(AXIS),angle)
            force_mesh(posed)
            mesh(diagnostic,n,posed,(196,129,74,255))
        diagnostic.export(HERE/f"{label}-mechanism-{angle}.glb")


def main():
    source_fixed,source_moving,hardware=hard_parts()
    reference_shin={n:source_moving[n] for n in ("R_shin_front","R_shin_back")}
    moving_hw={n:source_moving[n] for n in ("yoke","bridge","shin_cross_pad")}
    report={"source_commits":["c48e952","ac7d6aa"],"wall_mm":WALL,
            "cut_margin_mm":MARGIN,"pivot_axis_mm":list(map(float,AXIS)),
            "variants":{}}
    for label,case in CASES.items():
        print(f"Building {label}",flush=True)
        raw_fixed,moving=proposed_shell(case)
        fixed,removed=cut_fixed(raw_fixed,moving,hardware,moving_hw)
        rows,order_disagreements=pose_checks(fixed,moving,hardware,moving_hw,source_fixed,reference_shin)
        export_poses(label,fixed,moving,hardware,moving_hw)
        report["variants"][label]={
            "profiles":case,
            "fixed_removed_mm3":removed,
            "raw_fixed_solids":{n:len(s.solids()) for n,s in raw_fixed.items()},
            "cut_fixed_solids":{n:len(s.solids()) for n,s in fixed.items()},
            "moving_shin_solids":{n:len(s.solids()) for n,s in moving.items()},
            "cuff_thigh_front_overlap_mm3":round(overlap(raw_fixed["R_knee_cuff"],raw_fixed["R_thigh_front"]),2),
            "cuff_thigh_back_overlap_mm3":round(overlap(raw_fixed["R_knee_cuff"],raw_fixed["R_thigh_back"]),2),
            "cut_cuff_thigh_front_overlap_mm3":round(exact_contact(fixed["R_knee_cuff"],fixed["R_thigh_front"])[0],2),
            "cut_cuff_thigh_back_overlap_mm3":round(exact_contact(fixed["R_knee_cuff"],fixed["R_thigh_back"])[0],2),
            "support_clearance_mm":{n:round(min(s.distance_to(hw) for s in fixed.values()),3)
                                    for n,hw in hardware.items() if n.startswith("support_")},
            "boolean_order_disagreements_mm3":order_disagreements,
            "rows":rows,
        }
    (HERE/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
