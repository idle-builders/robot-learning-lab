"""Bounded source-shell, right-leg four-bar knee geometry screen (millimetres).

X is lateral, Y rear, Z up.  A/B are thigh pivots, C/D are shin pivots.
The pivot search was planar; all reported overlaps use the exact imported STEP
solids and newly constructed B-reps.  This is a geometry coupon, not hardware.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/lightweight-ape-20261004/source-shell-1mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/lightweight-ape-fourbar-knee-20261005"

# (Y,Z) from one compact planar search.  Rounded to 0.1 mm so the coupon is
# editable; no load, fatigue, or manufacturing optimization is implied.
A = np.array((1.6, 114.4))
B = np.array((11.5, 100.0))
C = np.array((1.9, 85.4))
D = np.array((13.9, 93.1))
SHIN_TOP = np.array((3.0, 83.0))
OUT_AXIS = np.array((-6.45, 132.5))
COLORS = {
    "thigh": (222, 216, 202, 255), "shin": (223, 207, 181, 255),
    "fixed_cover": (78, 84, 84, 255), "moving_cover": (58, 68, 69, 255),
    "link": (175, 112, 76, 255), "seat": (81, 105, 111, 255),
    "motor": (76, 124, 158, 200), "drive": (161, 85, 55, 255),
}


def box(x0, x1, y0, y1, z0, z1):
    return bd.Solid.make_box(x1-x0, y1-y0, z1-z0, bd.Plane(origin=(x0,y0,z0)))


def cylinder_x(x0, x1, yz, radius):
    return bd.Solid.make_cylinder(radius, x1-x0,
                                  bd.Plane(origin=(x0,*yz), z_dir=(1,0,0)))


def ring_x(x0, x1, yz, ro=3.0, ri=1.05):
    return cylinder_x(x0,x1,yz,ro)-cylinder_x(x0-.1,x1+.1,yz,ri)


def bar(x0, x1, start, end, radius=1.8):
    """Planar capsule, with a true constant lateral thickness."""
    d=np.asarray(end)-np.asarray(start)
    n=np.array((-d[1],d[0]))/np.linalg.norm(d)*radius
    yz=[np.asarray(start)+n,np.asarray(end)+n,np.asarray(end)-n,np.asarray(start)-n]
    wire=bd.Wire.make_polygon([(x0,*p) for p in yz],close=True)
    return (bd.Solid.extrude(bd.Face(wire),(x1-x0,0,0))
            + cylinder_x(x0,x1,start,radius)
            + cylinder_x(x0,x1,end,radius))


def rod(x, start, end, radius=.8):
    p=np.array((x,*start)); q=np.array((x,*end)); d=q-p
    return bd.Solid.make_cylinder(radius,float(np.linalg.norm(d)),
                                  bd.Plane(origin=tuple(p),z_dir=tuple(d/np.linalg.norm(d))))


def volume(shape):
    return sum(s.volume for s in shape.solids()) if shape else 0.0


def measured(a,b):
    # Keep exact B-rep intersection and gap.  A touch has zero volume and zero
    # gap; neither is a printable clearance.
    try:
        return {"overlap_mm3":round(volume(a & b),3),
                "gap_mm":round(a.distance_to(b),3)}
    except Exception as exc:
        return {"overlap_mm3":None,"gap_mm":None,"error":str(exc)}


def crossing(p,q):
    return float(p[0]*q[1]-p[1]*q[0])


def closure(input_deg):
    """Analytic circle intersection, continuous standing assembly branch."""
    front_len=np.linalg.norm(C-A); rear_len=np.linalg.norm(D-B)
    coupler_len=np.linalg.norm(D-C)
    t=math.atan2(*(C-A)[::-1])+math.radians(input_deg)
    cp=A+front_len*np.array((math.cos(t),math.sin(t)))
    v=B-cp; dist=np.linalg.norm(v)
    if dist>coupler_len+rear_len or dist<abs(coupler_len-rear_len):
        raise ValueError(f"four-bar cannot close at input {input_deg:g} deg")
    e=v/dist; along=(coupler_len**2-rear_len**2+dist**2)/(2*dist)
    height2=coupler_len**2-along**2
    if height2<0:raise ValueError("four-bar toggle")
    branch=np.sign(crossing(B-C,D-C))
    dp=cp+along*e+branch*math.sqrt(height2)*np.array((-e[1],e[0]))
    delta=dp-cp; base=D-C
    flex=math.degrees(math.atan2(crossing(base,delta),float(base@delta)))
    return cp,dp,flex,math.sqrt(height2)


def input_for_flex(flex_deg):
    lo,hi=0.,66.
    if flex_deg==0:return 0.
    if closure(hi)[2]<flex_deg:raise ValueError("target beyond sampled branch")
    for _ in range(48):
        mid=(lo+hi)/2
        if closure(mid)[2]<flex_deg:lo=mid
        else:hi=mid
    return (lo+hi)/2


def transform(shape, flex_deg, cp):
    # Rotate in the exact source frame then translate the shin C pivot to its
    # solved four-bar position.  Positive X-axis rotation is knee flexion.
    turned=shape.rotate(bd.Axis((0,*C),(1,0,0)),flex_deg)
    return turned.translate((0,float(cp[0]-C[0]),float(cp[1]-C[1])))


def transformed_yz(point,flex_deg,cp):
    d=np.asarray(point)-C;t=math.radians(flex_deg)
    return cp+np.array((math.cos(t)*d[0]-math.sin(t)*d[1],
                        math.sin(t)*d[0]+math.cos(t)*d[1]))


def mesh(shape,color):
    vertices,faces=shape.tessellate(.45,.3)
    m=trimesh.Trimesh(np.array([tuple(v) for v in vertices]),
                      np.asarray(faces),process=False)
    m.visual.vertex_colors=color
    return m


def source_geometry():
    if not SOURCE.exists():raise FileNotFoundError(SOURCE)
    src={part.label:part for part in bd.import_step(SOURCE).children}
    required=("R_thigh_front","R_thigh_back","R_shin_front","R_shin_back","R_knee_cuff")
    if any(label not in src for label in required):
        raise ValueError("source STEP has no exact right-leg B-reps")
    thigh=bd.Compound(children=[src["R_thigh_front"],src["R_thigh_back"]])
    shin=bd.Compound(children=[src["R_shin_front"],src["R_shin_back"]])
    cuff=src["R_knee_cuff"]
    # Opposing pieces of the *source* cuff make the first front/rear lap.
    # The planar split keeps a narrow standing fold and exposes any opening
    # and interference in flexion.  This is not a developed fabric pattern.
    front=cuff & box(20,64,-25,4.5,84.5,94)
    rear=cuff & box(20,64,4.8,27,78,89)
    return thigh,shin,cuff,front,rear


def fixed_geometry():
    motor=box(20,56.5,-15,19,125,140) # Feetech 36.5 x 34 x 15 mm, rotated about X
    plug=box(13,20,-11,-1,126,136)  # assumed connector/bend clearance
    spline=cylinder_x(56.5,62.6,OUT_AXIS,2.975)
    # Two opposed side mechanisms.  The pin seats are only 3 mm radius and
    # 2.5 mm axial allocation; bearings, retention, fit, and ribs are absent.
    seats=[ring_x(x0,x1,p) for x0,x1 in ((27,29.5),(54.5,57)) for p in (A,B)]
    return motor,plug,spline,seats


def moving_geometry():
    # Coupler bars and opposed moving seats form a rigid shin-side pivot
    # allocation.  No source shell is cut to pretend a printable attachment.
    couplers=[bar(x0,x1,C,D,1.45) for x0,x1 in ((27,29.5),(54.5,57))]
    seats=[ring_x(x0,x1,p) for x0,x1 in ((27,29.5),(54.5,57)) for p in (C,D)]
    # Ties couple left/right plates.  Root bar is an envelope, not a solved
    # fastener or a demonstrated load-bearing connection to the 1 mm skin.
    ties=[cylinder_x(29.5,54.5,C,1.05),cylinder_x(29.5,54.5,D,1.05)]
    root=bar(38,40,D,SHIN_TOP,1.6)
    return couplers,seats,ties,root


def drive(input_deg,cp):
    # Sourced spline size, but generic 6 mm crank and rod envelope.  The rod
    # length is set at standing; circle closure chooses the same branch at
    # subsequent poses.  No rated load or shaft adapter is assumed.
    front_point=A+.63*(cp-A)
    neutral_cp=closure(0)[0]
    neutral_q=A+.63*(neutral_cp-A)
    crank_radius=6.; initial=math.radians(-45.)
    neutral_p=OUT_AXIS+crank_radius*np.array((math.cos(initial),math.sin(initial)))
    rod_length=float(np.linalg.norm(neutral_q-neutral_p))
    v=front_point-OUT_AXIS; dist=float(np.linalg.norm(v))
    if dist>rod_length+crank_radius or dist<abs(rod_length-crank_radius):
        return None,rod_length
    e=v/dist; along=(crank_radius**2-rod_length**2+dist**2)/(2*dist)
    h=math.sqrt(max(crank_radius**2-along**2,0))
    candidates=[OUT_AXIS+along*e+s*h*np.array((-e[1],e[0])) for s in (-1,1)]
    # Follow the branch nearest the standing crank without switching.
    p=min(candidates,key=lambda item:np.linalg.norm(item-neutral_p))
    return (p,front_point),rod_length


def shapes_for(flex, src, fixed, moving):
    thigh,shin,cuff,front,rear=src
    motor,plug,spline,fixed_seats=fixed
    couplers,moving_seats,ties,root=moving
    q=input_for_flex(flex);cp,dp,actual,toggle_height=closure(q)
    side_links=[]
    for x0,x1 in ((27,29.5),(54.5,57)):
        side_links.extend((bar(x0,x1,A,cp),bar(x0,x1,B,dp)))
    moved=lambda shape:transform(shape,actual,cp)
    ds=drive(q,cp)
    crank=None; drive_rod=None
    if ds[0] is not None:
        p,attach=ds[0]
        crank=bar(59,61,OUT_AXIS,p,1.1)
        drive_rod=rod(61,p,attach,.85)
    parts={
        "source_thigh":thigh,"source_shin":moved(shin),
        "front_lap":front,"rear_lap":moved(rear),
        "FT3332_case":motor,"connector_assumed":plug,"spline":spline,
        "fixed_seats":bd.Compound(children=fixed_seats),
        "moving_seats":bd.Compound(children=[moved(s) for s in moving_seats]),
        "front_rear_links":bd.Compound(children=side_links),
        "coupler":bd.Compound(children=[moved(s) for s in couplers]),
        "tie_pins":bd.Compound(children=[moved(s) for s in ties]),
        "root_envelope":moved(root),
    }
    if crank is not None:parts["crank_envelope"]=crank
    if drive_rod is not None:parts["rod_envelope"]=drive_rod
    root_pos=transformed_yz(SHIN_TOP,actual,cp)
    return parts,{"flex_deg":flex,"input_deg":round(q,4),
                  "actual_flex_deg":round(actual,5),
                  "C_yz_mm":np.round(cp,3).tolist(),"D_yz_mm":np.round(dp,3).tolist(),
                  "shin_top_yz_mm":np.round(root_pos,3).tolist(),
                  "circle_height_mm":round(toggle_height,3),
                  "drive_rod_length_mm":round(ds[1],3),
                  "drive_closes":ds[0] is not None}


def export_scene(flex,parts):
    scene=trimesh.Scene()
    kind={"source_thigh":"thigh","source_shin":"shin","front_lap":"fixed_cover",
          "rear_lap":"moving_cover","FT3332_case":"motor","connector_assumed":"motor",
          "spline":"drive","fixed_seats":"seat","moving_seats":"seat",
          "front_rear_links":"link","coupler":"link","tie_pins":"seat",
          "root_envelope":"link","crank_envelope":"drive","rod_envelope":"drive"}
    for name,part in parts.items():
        color=COLORS[kind[name]]
        for i,solid in enumerate(part.solids()):
            if solid.volume>.001:
                node=f"{name}_{i}"
                scene.add_geometry(mesh(solid,color),node_name=node,geom_name=node)
    scene.export(OUT/f"fourbar-{flex:03d}.glb")
    named=[]
    for name,part in parts.items():
        part.label=name
        named.append(part)
    bd.export_step(bd.Compound(children=named),OUT/f"fourbar-{flex:03d}.step")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    src=source_geometry();fixed=fixed_geometry();moving=moving_geometry()
    thigh,shin,cuff,front,rear=src
    # Kept small and explicit: all hard pairs relevant to this first coupon.
    pairs=(("source_thigh","source_shin"),("source_thigh","rear_lap"),
           ("front_lap","source_shin"),("front_lap","rear_lap"),
           ("FT3332_case","source_thigh"),("FT3332_case","front_rear_links"),
           ("FT3332_case","coupler"),("FT3332_case","source_shin"),
           ("fixed_seats","source_shin"),("fixed_seats","coupler"),
           ("source_thigh","front_rear_links"),("source_thigh","coupler"),
           ("source_thigh","moving_seats"),("rod_envelope","source_thigh"),
           ("rod_envelope","front_rear_links"))
    report={"status":"geometry screen; no hardware or control selection",
            "source_step":str(SOURCE.relative_to(ROOT)),
            "frame":"X lateral, Y rear, Z up; millimetres",
            "fixed_pivots_yz_mm":{"A":A.tolist(),"B":B.tolist()},
            "shin_pivots_at_extension_yz_mm":{"C":C.tolist(),"D":D.tolist()},
            "links_mm":{"front_AC":round(float(np.linalg.norm(C-A)),3),
                        "rear_BD":round(float(np.linalg.norm(D-B)),3),
                        "coupler_CD":round(float(np.linalg.norm(D-C)),3),
                        "ground_AB":round(float(np.linalg.norm(B-A)),3)},
            "case_envelope_mm":[36.5,34,15],
            "case_orientation":"FT3332 34 x 15 x 36.5 mm body rotated about X; output +X",
            "case_bounds_mm":[[20,-15,125],[56.5,19,140]],
            "connector":"assumed 7 x 10 x 10 mm envelope; vendor bend not validated",
            "pin_seats":"two opposed 2.5 mm side allocations at X 27-29.5 and 54.5-57; 3 mm outer radius, 1.05 mm nominal bore",
            "cover":"source cuff divided into fixed front lap and moving rear lap; no developed compliant fold or retention",
            "source_cuff_volume_mm3":round(volume(cuff),3),
            "front_lap_volume_mm3":round(volume(front),3),
            "rear_lap_volume_mm3":round(volume(rear),3),
            "source_thigh_shell_removed_mm3":0,
            "source_shin_shell_removed_mm3":0,
            "sweep_step_deg":5,"poses":[]}
    for flex in range(0,121,5):
        parts,row=shapes_for(flex,src,fixed,moving)
        if flex in (0,60,90,115,120):
            row["brep_pairs"]={f"{a}__{b}":measured(parts[a],parts[b])
                                 for a,b in pairs if a in parts and b in parts}
            # The closest reported separation across the specified non-mating
            # cover/shell pairs excludes deliberate pin/link and mount contact.
            nonmate=("source_thigh__source_shin","source_thigh__rear_lap",
                     "front_lap__source_shin","front_lap__rear_lap")
            row["minimum_cover_shell_gap_mm"]=min(row["brep_pairs"][p]["gap_mm"]
                                                   for p in nonmate)
            export_scene(flex,parts)
            print(f"exported {flex} degrees",flush=True)
        h=.05
        row["d_shin_d_input"] = round((closure(row["input_deg"]+h)[2]-
                                       closure(max(0,row["input_deg"]-h))[2])/
                                      (2*h if row["input_deg"]>=h else h),4)
        report["poses"].append(row)
    report["any_fourbar_toggle_in_sample"]=any(p["circle_height_mm"]<1 for p in report["poses"])
    report["drive_closes_all_sampled"]=all(p["drive_closes"] for p in report["poses"])
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"artifact":str(OUT),"drive_closes_all_sampled":report["drive_closes_all_sampled"],
                      "key_poses":[p for p in report["poses"] if p["flex_deg"] in (0,60,90,115,120)]},indent=2))


if __name__=="__main__":main()
