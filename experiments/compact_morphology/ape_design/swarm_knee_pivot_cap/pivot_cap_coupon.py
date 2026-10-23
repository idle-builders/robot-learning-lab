"""Exact 50 cm offset-knee pivot-cap cup trial, with opaque pose exports.

The right lateral cap rotates with the output shaft. Its 2 mm plate and lip are
real B-rep solids, so cap contact is checked by exact common volume. This is
only a local coupon: broad opaque appearance and whole-body clearance are
separate gates.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as b
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments.compact_morphology.ape_design.swarm_knee_flex_closure.closure_coupon import (
    AXIS, hard_parts, overlap,
)
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import turn
from experiments.compact_morphology.ape_design.swarm_rated_leg_drive.rated_leg_coupon import mesh

ANGLES = (0,60,115)
R_OUT = 52.0
R_IN = 50.0
CAP_FACE_X = (93.0,95.0)
RIM_X = (80.0,93.0)
WALL = 2.0
MARGIN = 0.5


def cylinder_x(radius,x0,x1):
    return b.Pos((x0+x1)/2,AXIS[1],AXIS[2])*b.Cylinder(
        radius,x1-x0,rotation=(0,90,0))


def cap_parts():
    # A shallow side cup: 2 mm plate, 2 mm annular return lip. The return
    # reaches only the lateral edge of the existing cuff. R_IN=50 leaves
    # about 0.77 mm radial clearance to its full ~49.23 mm pivot extent.
    plate=cylinder_x(R_OUT,*CAP_FACE_X)
    lip=cylinder_x(R_OUT,*RIM_X)-cylinder_x(R_IN,RIM_X[0]-.1,RIM_X[1]+.1)
    cap=plate+lip
    # Integral short hub meets the moving output shaft for a geometric load
    # route; it is not a rated joint or fastener specification.
    hub=cylinder_x(4.5,91.5,94.5)
    return cap+hub,dict(plate=plate,lip=lip,hub=hub)


def volume(shape):
    return sum(s.volume for s in shape.solids()) if shape else 0.0


def projected(shape,coordinates):
    vertices,faces=shape.tessellate(.65,.65)
    xyz=np.asarray([[v.X,v.Y,v.Z] for v in vertices])[:,coordinates]
    polygons=[]
    for face in faces:
        poly=Polygon(xyz[np.asarray(face)])
        if poly.area>1e-5:
            polygons.append(poly)
    return unary_union(polygons)


def clearance_rows(cap, fixed, moving, hardware, shaft):
    rows=[]
    for angle in ANGLES:
        posed={n:turn(s,tuple(AXIS),angle) for n,s in moving.items()}
        checks={**fixed,**hardware,**posed}
        hits={}
        distances={}
        for name,shape in checks.items():
            v=overlap(cap,shape)
            if v>.05:
                hits[name]=round(v,3)
            else:
                distances[name]=round(cap.distance_to(shape),3)
        source_side=unary_union([projected(s,(1,2)) for s in
                                 (*fixed.values(),posed["R_shin_front"],
                                  posed["R_shin_back"])])
        cap_side=projected(cap,(1,2))
        source_front=unary_union([projected(s,(0,2)) for s in
                                  (*fixed.values(),posed["R_shin_front"],
                                   posed["R_shin_back"])])
        cap_front=projected(cap,(0,2))
        rows.append(dict(angle_deg=angle,cap_hard_overlap_mm3=hits,
                         disjoint_clearance_mm=distances,
                         side_projected_new_area_mm2=round(cap_side.difference(source_side).area,2),
                         front_projected_new_area_mm2=round(cap_front.difference(source_front).area,2)))
    return rows


def export_poses(cap,fixed,moving,hardware,shaft):
    for angle in ANGLES:
        posed={n:turn(s,tuple(AXIS),angle) for n,s in moving.items()}
        opaque=trimesh.Scene()
        for name,shape in fixed.items():
            mesh(opaque,name,shape,(230,222,207,255))
        for name in ("R_shin_front","R_shin_back"):
            mesh(opaque,name,posed[name],(230,222,207,255))
        mesh(opaque,"moving_pivot_cap",cap,(214,206,190,255))
        opaque.export(HERE/f"cap-{angle}.glb")
        diagnostic=trimesh.Scene()
        for name,shape in fixed.items():
            mesh(diagnostic,name,shape,(230,222,207,80))
        for name in ("R_shin_front","R_shin_back"):
            mesh(diagnostic,name,posed[name],(230,222,207,80))
        for name,shape in hardware.items():
            mesh(diagnostic,name,shape,(55,100,148,255))
        for name in ("yoke","bridge","shin_cross_pad"):
            mesh(diagnostic,name,posed[name],(196,129,74,255))
        mesh(diagnostic,"moving_shaft",shaft,(196,129,74,255))
        mesh(diagnostic,"moving_pivot_cap",cap,(54,119,104,155))
        diagnostic.export(HERE/f"mechanism-{angle}.glb")


def main():
    fixed,moving,hardware=hard_parts()
    shaft=cylinder_x(4,36,92)
    cap,parts=cap_parts()
    rows=clearance_rows(cap,fixed,moving,hardware,shaft)
    component_hits={}
    for part_name,part in parts.items():
        contacts={}
        for name,shape in {**fixed,**moving}.items():
            v=overlap(part,shape)
            if v>.05:
                contacts[name]=round(v,3)
        component_hits[part_name]=contacts
    export_poses(cap,fixed,moving,hardware,shaft)
    # Max pivot radius of the cuff mesh bounds a full radial nesting claim;
    # the side return still does not cover the full axial width of the cuff.
    cuff=fixed["R_knee_cuff"]
    vv,_=cuff.tessellate(.35,.35)
    cuff_r=max(math.hypot(v.Y-AXIS[1],v.Z-AXIS[2]) for v in vv)
    report=dict(source_commits=["c48e952","00453e7"],
                target_height_mm=500,pivot_axis_mm=list(map(float,AXIS)),
                cap_parent="moving output shaft; axisymmetric cup rotates without silhouette change",
                support="4.5 mm radius integral hub x91.5–94.5 overlaps 4 mm radius moving shaft ending x92",
                cap_outer_radius_mm=R_OUT,cap_inner_lip_radius_mm=R_IN,
                cap_face_x_mm=CAP_FACE_X,return_lip_x_mm=RIM_X,
                cap_wall_mm=WALL,source_hard_margin_mm=MARGIN,
                cap_volume_mm3=round(volume(cap),2),
                cap_solids=len(cap.solids()),
                cuff_max_pivot_radius_mm=round(cuff_r,2),
                cuff_radial_gap_to_lip_mm=round(R_IN-cuff_r,2),
                shaft_to_cap_overlap_mm3=round(overlap(cap,shaft),3),
                source_cuff_max_x_mm=round(cuff.bounding_box().max.X,2),
                cap_max_x_mm=round(cap.bounding_box().max.X,2),
                cap_rear_y_mm=round(cap.bounding_box().max.Y,2),
                source_fixed_rear_y_mm=round(max(s.bounding_box().max.Y for s in fixed.values()),2),
                fixed_skin_solids={n:len(s.solids()) for n,s in fixed.items()},
                component_overlap_mm3_at_zero=component_hits,
                rows=rows,
                verdict="REJECT: exact cap-to-thigh and cap-to-shin intersections, a proud circular side puck, and an unclosed 115-degree rear opening",
                qualification="exact sampled-pose B-rep overlap and projection only; no continuous-angle sweep, fastener rating, or whole-robot motion proof")
    (HERE/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
