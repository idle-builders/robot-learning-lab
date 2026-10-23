"""One 500 mm Cove hip root candidate; execute the CAD coupon, not a robot pass.

Coordinates are mm: X robot right, Y rear, Z up. The Cove shell is imported
from its committed parametric source. All motors and bearings stay source size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import build123d as b
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_shell_v2 import build as cove

OUT = HERE / "output"
WALL = 2.0
GAP = 0.5
PIVOT = (59.0, 2.0, 247.0)
PITCH = (59.0, 2.0, 219.0)
POSES = {
    "stand": (0, 0, 0),
    "yaw_out": (-20, 0, 0),
    "yaw_in": (20, 0, 0),
    "stride": (20, 0, -35),
    "squat": (0, 0, -42),
    "crawl": (0, 0, -45),
}


def cyl(radius, length, origin, axis=(0, 0, 1)):
    return b.Solid.make_cylinder(radius, length, b.Plane(origin=origin, z_dir=axis))


def annulus(ro, ri, length, origin, axis=(0, 0, 1)):
    return cyl(ro, length, origin, axis) - cyl(ri, length + .2,
        tuple(origin[i] - .1 * axis[i] for i in range(3)), axis)


def box(center, size):
    return b.Pos(*center) * b.Box(*size)


def volume(shape):
    if shape is None:
        return 0.0
    return sum(x.volume for x in shape.solids())


def overlap(a, other):
    aa, bb = a.bounding_box(), other.bounding_box()
    if any(getattr(aa.min, k) >= getattr(bb.max, k) or
           getattr(bb.min, k) >= getattr(aa.max, k) for k in "XYZ"):
        return [0.0, 0.0], None
    try:
        return [volume(a & other), volume(other & a)], None
    except Exception as exc:
        return [None, None], type(exc).__name__ + ": " + str(exc)[:120]


def bbox_gap_lower_bound(a, other):
    aa, bb = a.bounding_box(), other.bounding_box()
    gaps=[max(0.0, float(getattr(bb.min,k)-getattr(aa.max,k)),
                  float(getattr(aa.min,k)-getattr(bb.max,k))) for k in "XYZ"]
    return sum(g*g for g in gaps)**.5


def transform(shape, yaw, roll, pitch, stage, side):
    pivot = (side*PIVOT[0],PIVOT[1],PIVOT[2])
    pitch_pivot = (side*PITCH[0],PITCH[1],PITCH[2])
    p = b.Pos(*pivot)
    inv = b.Pos(*(-x for x in pivot))
    t = p * b.Rot(0, 0, yaw) * inv
    if stage in ("roll", "pitch", "thigh"):
        t = t * p * b.Rot(0, roll, 0) * inv
    if stage == "thigh":
        q = b.Pos(*pitch_pivot)
        t = t * q * b.Rot(pitch, 0, 0) * b.Pos(*(-x for x in pitch_pivot))
    return t * shape


def candidate_shell():
    variant = cove.VARIANTS[0]
    parts, _ = cove.build(variant)
    by_name = {p.name:p for p in parts}
    pelvis_st = [(201,0,3,20,24),(214,0,4,39,32),
                 (232,0,4,59,38),(251,0,4,69,41),(266,0,4,64,40)]
    pelvis, _ = cove.skin(pelvis_st, WALL)
    for side in (-1, 1):
        pelvis = pelvis - cove.ellipsoid((side*59, 2, 247), (33,35,29))
    by_name["pelvis_v"].shape = pelvis
    for side, tag in ((-1, "L"),(1, "R")):
        hx = side*59; kx = side*61
        thigh, _ = cove.skin([
            (132,kx,3,23,26),(146,kx,3,31,32),
            (173,hx,3,36,36),(204,hx,4,37,38),
            (232,hx,4,35,37),(256,hx-side*2,4,30,34),
            (271,hx-side*4,4,24,28),(280,hx-side*6,4,15,20)])
        lobe = cove.ellipsoid((hx,2,247),(31,33,27)) - \
               cove.ellipsoid((hx,2,247),(29,31,25))
        by_name[f"{tag}_thigh"].shape = thigh.fuse(lobe)
    return parts, by_name


def hardware_right():
    """Serial axes with real case/bearing envelopes; transfer teeth are not modeled."""
    fixed = {
        # RSF-5B-50-IDT: housing diameter 25, length 35.8. Inboard yaw
        # output is routed to the supported vertical axis by a belt reserve.
        "yaw_RSF_case": cyl(12.5, 35.8, (14,0,227)),
        "yaw_bearing_low_W618_6": annulus(6.5,3.1,3.5,(59,2,228)),
        "yaw_bearing_high_W618_6": annulus(6.5,3.1,3.5,(59,2,241)),
        "yaw_seat_low": annulus(9,6.6,5.0,(59,2,227.25)),
        "yaw_seat_high": annulus(9,6.6,5.0,(59,2,240.25)),
        "yaw_bridge_front": box((38,-15,229.5),(41,4,5)),
        "yaw_bridge_rear": box((38,19,229.5),(41,4,5)),
        "yaw_belt_reserve": box((36.5,0,263.5),(49,13,2)),
    }
    yaw = {
        "yaw_shaft": cyl(3,39,(59,2,224)),
        "roll_RSF_case": cyl(12.5,35.8,(59,-19,247),(0,1,0)),
        "roll_bearing_a_W618_6": annulus(6.5,3.1,3.5,(59,19,247),(0,1,0)),
        "roll_bearing_b_W618_6": annulus(6.5,3.1,3.5,(59,29,247),(0,1,0)),
        "roll_seat_a": annulus(9,6.6,5,(59,18.25,247),(0,1,0)),
        "roll_seat_b": annulus(9,6.6,5,(59,28.25,247),(0,1,0)),
    }
    roll = {
        "roll_shaft": cyl(3,15.2,(59,16.8,247),(0,1,0)),
        "pitch_RSF_case": cyl(12.5,35.8,(29,2,219),(1,0,0)),
        "pitch_bearing_a_W618_6": annulus(6.5,3.1,3.5,(70,2,219),(1,0,0)),
        "pitch_bearing_b_W618_6": annulus(6.5,3.1,3.5,(82,2,219),(1,0,0)),
        "pitch_seat_a": annulus(9,6.6,5,(69.25,2,219),(1,0,0)),
        "pitch_seat_b": annulus(9,6.6,5,(81.25,2,219),(1,0,0)),
    }
    pitch = {"pitch_shaft":cyl(3,31.2,(64.8,2,219),(1,0,0))}
    return fixed,yaw,roll,pitch


def checked_shape(name, shape):
    return {"valid": bool(shape.is_valid), "solid_count":len(shape.solids()),
            "volume_mm3":round(volume(shape),3),
            "bounds_mm":[[round(float(v),2) for v in shape.bounding_box().min],
                         [round(float(v),2) for v in shape.bounding_box().max]]}


def report_pair(rows, pose, a_name, a, b_name, other, kind):
    if bbox_gap_lower_bound(a,other) >= GAP:
        return
    vals, err = overlap(a, other)
    if err:
        rows.append({"pose":pose,"pair":[a_name,b_name],"kind":kind,
                     "status":"unknown","error":err})
        return
    if max(vals) > .05:
        rows.append({"pose":pose,"pair":[a_name,b_name],"kind":kind,
                     "status":"penetration", "both_orders_mm3":[round(v,3) for v in vals]})
    elif max(vals) == 0 and any(v is None for v in vals):
        raise AssertionError("unreachable")
    else:
        try:
            d = a.distance_to(other)
        except Exception as exc:
            rows.append({"pose":pose,"pair":[a_name,b_name],"kind":kind,
                         "status":"unknown_gap","error":type(exc).__name__})
            return
        if d < GAP:
            rows.append({"pose":pose,"pair":[a_name,b_name],"kind":kind,
                         "status":"tight_gap","gap_mm":round(d,4)})


def screen(parts, by_name):
    fixed_r,yaw_r,roll_r,pitch_r = hardware_right()
    all_hardware={}
    for side,tag in ((-1,"L"),(1,"R")):
        reflect=lambda s:s if side==1 else s.mirror(b.Plane.YZ)
        all_hardware[tag]={stage:{name:reflect(s) for name,s in group.items()}
                           for stage,group in (("fixed",fixed_r),("yaw",yaw_r),
                                               ("roll",roll_r),("thigh",pitch_r))}
    pack = box((0,27,290),(60,30,30.5)) # prior 14 mm rear shift from Y=13
    service = box((0,-18,369),(25,20,14))
    fixed_skin={n:by_name[n].shape for n in ("pelvis_v","torso")}
    rows=[]
    for tag in ("L","R"):
        hw=all_hardware[tag]
        for name in ("yaw_RSF_case","yaw_bridge_front","yaw_bridge_rear","yaw_belt_reserve"):
            shape=hw["fixed"][name]
            for other_name,other in {"shared_pack":pack,"service":service,**fixed_skin}.items():
                report_pair(rows,"stand",tag+"_"+name,shape,other_name,other,"fixed")
        report_pair(rows,"stand",tag+"_roll_RSF_case",hw["yaw"]["roll_RSF_case"],
                    tag+"_yaw_RSF_case",hw["fixed"]["yaw_RSF_case"],"case_case")
    for pose,(yaw_angle,roll_angle,pitch_angle) in POSES.items():
        print("screen",pose,flush=True)
        for side,tag in ((-1,"L"),(1,"R")):
            hw=all_hardware[tag]
            # yaw signs mirror between sides; pitch is anatomical same direction.
            yaw=yaw_angle*side
            moving={}
            for stage in ("yaw","roll","thigh"):
                for name,shape in hw[stage].items():
                    moving[name]=transform(shape,yaw,roll_angle,pitch_angle,stage,side)
            moving["thigh_shell"]=transform(by_name[f"{tag}_thigh"].shape,
                                             yaw,roll_angle,pitch_angle,"thigh",side)
            screen_names=("roll_RSF_case","pitch_RSF_case","thigh_shell")
            fixed_targets={**fixed_skin,"shared_pack":pack,
                           **{n:hw["fixed"][n] for n in
                              ("yaw_RSF_case","yaw_bridge_front","yaw_bridge_rear",
                               "yaw_belt_reserve","yaw_seat_low","yaw_seat_high")}}
            for name in screen_names:
                shape=moving[name]
                for fixed_name,fixed in fixed_targets.items():
                    report_pair(rows,pose,tag+"_"+name,shape,
                                fixed_name if fixed_name in fixed_skin or fixed_name=="shared_pack"
                                else tag+"_"+fixed_name,fixed,"moving_fixed")
            # Cases of serial adjacent stages cannot share volume.
            for name in ("roll_RSF_case","pitch_RSF_case"):
                if name=="roll_RSF_case":
                    other=moving["pitch_RSF_case"]
                    report_pair(rows,pose,tag+"_roll_RSF_case",moving[name],
                                tag+"_pitch_RSF_case",other,"case_case")
    stats={}
    for pose in POSES:
        subset=[r for r in rows if r["pose"]==pose]
        stats[pose]={"penetrations":sum(r["status"]=="penetration" for r in subset),
                     "unknown":sum(r["status"].startswith("unknown") for r in subset),
                     "tight_gaps":sum(r["status"]=="tight_gap" for r in subset)}
    return rows,stats,all_hardware,pack,service


def render_sheet(parts, by_name):
    # The hip coupon keeps the complete opaque body for silhouette context.
    # Distal groups follow the source pose sketch; they are not a constrained
    # closure and are not part of the exact hip screen.
    w,h=290,340
    views=list(cove.VIEWS)
    sheet=Image.new("RGB",(w*4,(h+24)*len(POSES)),"white")
    pen=ImageDraw.Draw(sheet)
    meshes=[(p,cove.mesh(p.shape,p.color)) for p in parts]
    for row,(pose,angles) in enumerate(POSES.items()):
        source_pose="crawl" if pose=="crawl" else "squat" if pose=="squat" else "stand"
        mats=cove.matrices(cove.VARIANTS[0],source_pose)
        for side,tag in ((-1,"L"),(1,"R")):
            # The source renderer accepts 4x4 matrices; its thigh transform is
            # replaced by the candidate's serial-axis rotation.
            yaw=angles[0]*side;roll=angles[1];pitch=angles[2]
            p=np.asarray(PIVOT)*np.array([side,1,1]);q=np.asarray(PITCH)*np.array([side,1,1])
            mat=(cove.rotation((0,0,1),yaw,p) @
                 cove.rotation((0,1,0),roll,p) @
                 cove.rotation((1,0,0),pitch,q))
            source_hip=cove.rotation((1,0,0),cove.POSES[source_pose]["hip"],(side*59,2,231))
            floor=mats["fixed"]
            for distal in (f"{tag}_shin",f"{tag}_boot"):
                mats[distal]=(floor @ mat @ np.linalg.inv(source_hip) @
                              np.linalg.inv(floor) @ mats[distal])
            mats[f"{tag}_thigh"]=floor @ mat
        for col,view in enumerate(views):
            tile=cove.render(meshes,mats,view,(w,h))
            sheet.paste(tile,(col*w,row*(h+24)+24))
            pen.text((col*w+8,row*(h+24)+5),f"Cove hip · {pose} · {view}",fill=(45,42,39))
    sheet.save(OUT/"opaque-poses.png",optimize=True)
    detail=Image.new("RGB",(4*380,len(POSES)*364),"white")
    detail_pen=ImageDraw.Draw(detail)
    for row,pose in enumerate(POSES):
        for col,view in enumerate(views):
            tile=sheet.crop((col*w+50,row*(h+24)+125,
                             col*w+240,row*(h+24)+307))
            detail.paste(tile.resize((380,364),Image.Resampling.LANCZOS),
                         (col*380,row*364))
            detail_pen.text((col*380+8,row*364+5),
                            f"{pose} · {view}",fill=(45,42,39))
    detail.save(OUT/"opaque-hip-detail.png",optimize=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parts,by_name=candidate_shell()
    rows,stats,hardware,pack,service=screen(parts,by_name)
    component_shapes={tag+"_"+name:checked_shape(name,s) for tag,stages in hardware.items()
                      for stage,group in stages.items() for name,s in group.items()}
    report={"height_mm":500,"wall_mm":WALL,"clearance_gate_mm":GAP,
            "candidate":"raised yaw pivot and thigh crown, 14 mm rear pack, serial yaw/roll/pitch round drives",
            "axes_right_mm":{"yaw_z":PIVOT,"roll_y":PIVOT,"pitch_x":PITCH},
            "cases":"RSF-5B-50-IDT Ø25 x 35.8 mm, three per side; official housing envelope",
            "bearings":"SKF W618/6 Ø13 x 3.5 mm, two per axis per side; seats are proposals",
            "pack":"Tattu 4S850 60 x 30 x 30.5 mm at (0,27,290); shared with both hips",
            "shell_shapes":{n:checked_shape(n,by_name[n].shape)
                            for n in ("pelvis_v","R_thigh","L_thigh")},
            "hardware_shapes":component_shapes,
            "pack_shape":checked_shape("pack",pack),
            "pose_angles_deg":POSES,"summary":stats,"contacts":rows,
            "screen_scope":"Exact both-order intersections for the two fixed pelvis/torso skins, shared pack, yaw case/bridge/belt/seats, moving thigh and moving roll/pitch cases at sampled poses. Bearing journals and unselected support-component pairs are not a complete assembly clearance proof.",
            "interpretation":"Any nonmating penetration or unresolved Boolean rejects this candidate. Sampled poses do not prove the intervening path. Belt, flange, fastening, duty and heat remain unqualified."}
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("render opaque views",flush=True)
    render_sheet(parts,by_name)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=="__main__":main()
