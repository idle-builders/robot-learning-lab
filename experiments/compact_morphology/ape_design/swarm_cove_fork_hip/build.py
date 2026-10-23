"""One bounded, source-size 500 mm Cove fork-hip coupon.

Coordinates are mm: +X right, +Y rear, +Z up.  The parent of each joint
carries its motor; radial/bending load goes through distinct shafts, seats and
frame plates.  This coupon screens exact rigid BReps, not belt teeth or fatigue.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as b
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_shell_v2 import build as cove
from experiments.compact_morphology.ape_design.swarm_cove_hip import build as old

OUT = HERE / "output"
GAP = .5
PIVOT = (59., 2., 247.)
PITCH = (59., 2., 219.)
POSES = old.POSES
R = 9.55  # 20 teeth x 3 mm pitch / 2pi, Gates 3MGT pitch radius
BELT_WIDTH = 15.
MOTOR_G = 74.5


def box(center, size):
    return b.Pos(*center) * b.Box(*size)


def cyl(radius, length, origin, axis=(0, 0, 1)):
    return b.Solid.make_cylinder(radius, length, b.Plane(origin=origin, z_dir=axis))


def ring(ro, ri, length, origin, axis=(0, 0, 1)):
    return cyl(ro, length, origin, axis) - cyl(ri, length+.2,
        tuple(origin[i]-.1*axis[i] for i in range(3)), axis)


def belt(c1, c2, axis, width=BELT_WIDTH):
    """Closed 3MGT pitch-loop material envelope; no tooth/cord rating implied.

    Axis-aligned capsule is the outer filled pulley sweep less its inner
    sweep.  Both 20T pulleys are 31.5 mm apart, so pitch length is 123 mm.
    """
    if axis == "z":
        x1,y,z=c1; x2=c2[0]; lo=min(x1,x2); hi=max(x1,x2)
        outer=cyl(R+.8,width,(x1,y,z))+cyl(R+.8,width,(x2,y,z))
        outer=outer.fuse(box(((lo+hi)/2,y,z+width/2),(hi-lo,2*(R+.8),width)))
        inner=cyl(R,width+.2,(x1,y,z-.1))+cyl(R,width+.2,(x2,y,z-.1))
        inner=inner.fuse(box(((lo+hi)/2,y,z+width/2),(hi-lo,2*R,width+.2)))
    elif axis == "y":
        x,y,z1=c1; z2=c2[2]; lo=min(z1,z2); hi=max(z1,z2)
        outer=cyl(R+.8,width,(x,y,z1),(0,1,0))+cyl(R+.8,width,(x,y,z2),(0,1,0))
        outer=outer.fuse(box((x,y+width/2,(lo+hi)/2),(2*(R+.8),width,hi-lo)))
        inner=cyl(R,width+.2,(x,y-.1,z1),(0,1,0))+cyl(R,width+.2,(x,y-.1,z2),(0,1,0))
        inner=inner.fuse(box((x,y+width/2,(lo+hi)/2),(2*R,width+.2,hi-lo)))
    elif axis == "x":
        x,y1,z=c1; y2=c2[1]; lo=min(y1,y2); hi=max(y1,y2)
        outer=cyl(R+.8,width,(x,y1,z),(1,0,0))+cyl(R+.8,width,(x,y2,z),(1,0,0))
        outer=outer.fuse(box((x+width/2,(lo+hi)/2,z),(width,hi-lo,2*(R+.8))))
        inner=cyl(R,width+.2,(x-.1,y1,z),(1,0,0))+cyl(R,width+.2,(x-.1,y2,z),(1,0,0))
        inner=inner.fuse(box((x+width/2,(lo+hi)/2,z),(width+.2,hi-lo,2*R)))
    else: raise ValueError(axis)
    return outer-inner


def right_hardware():
    # The actual drive is Feetech ST-3250-C001.  Its case, mass and rated
    # torque are sourced.  The 12.2 mm off-centre spline datum is retained.
    # Fixed yaw case at X=27.5, output Y=2 from case centre Y=14.2.
    fixed={
        "yaw_ST3250_case":box((27.5,14.2,223),(35,24.72,45.22)),
        "yaw_motor_output":cyl(3,25,(27.5,2,245)),
        "yaw_drive_20T":cyl(R,15,(27.5,2,255)),
        "yaw_belt_123_3MGT_15":belt((27.5,2,255),(59,2,255),"z"),
        "yaw_bearing_low_W618_6":ring(6.5,3,3.5,(59,2,228)),
        "yaw_bearing_high_W618_6":ring(6.5,3,3.5,(59,2,246)),
        "yaw_seat_low":ring(8.5,6.5,5.5,(59,2,227)),
        "yaw_seat_high":ring(8.5,6.5,5.5,(59,2,245)),
        "yaw_bulkhead_low":box((59,2,227.75),(27,39,2)),
        "yaw_bulkhead_high":box((59,2,245.75),(27,39,2)),
    }
    # Clear Ø13 bearing bores through the two bulkheads; remaining plate
    # reaches the inboard pelvis.  Seat-to-plate is an intentional fusion.
    for name,z in (("yaw_bulkhead_low",227.75),("yaw_bulkhead_high",245.75)):
        fixed[name]=fixed[name]-cyl(6.5,3,(59,2,z-1.5))
    fixed["yaw_bulkhead_low"]=fixed["yaw_bulkhead_low"].fuse(fixed["yaw_seat_low"])
    fixed["yaw_bulkhead_high"]=fixed["yaw_bulkhead_high"].fuse(fixed["yaw_seat_high"])

    # Yaw carrier has a separate spindle and fork.  Roll's motor sits on
    # this parent, 31.5 mm above roll output; its belt plane is ahead of fork.
    yaw={
        "yaw_spindle":cyl(3,46,(59,2,227)),
        "yaw_driven_20T":cyl(R,15,(59,2,255)),
        "yaw_fork_front":box((59,-20,253),(20,3,44)),
        "yaw_fork_rear":box((59,24,253),(20,3,44)),
        "yaw_fork_bridge":box((59,2,273.5),(20,47,3)),
        "roll_bearing_front_W618_6":ring(6.5,3,3.5,(59,-23,247),(0,1,0)),
        "roll_bearing_rear_W618_6":ring(6.5,3,3.5,(59,23,247),(0,1,0)),
        "roll_seat_front":ring(8.5,6.5,5.5,(59,-24,247),(0,1,0)),
        "roll_seat_rear":ring(8.5,6.5,5.5,(59,22,247),(0,1,0)),
        "roll_ST3250_case":box((59,-2,290.7),(35,45.22,24.72)),
        "roll_motor_output":cyl(3,32,(59,-43,278.5),(0,1,0)),
        "roll_drive_20T":cyl(R,15,(59,-43,278.5),(0,1,0)),
        "roll_belt_123_3MGT_15":belt((59,-43,247),(59,-43,278.5),"y"),
    }
    for nm in ("yaw_fork_front","yaw_fork_rear"):
        y=-21.5 if nm.endswith("front") else 22.5
        yaw[nm]=yaw[nm]-cyl(6.5,5,(59,y-2.5,247),(0,1,0))
    # The roll frame is driven at the output journal, not supported by the
    # motor bushing.  It carries pitch's case and two separated X bearings.
    roll={
        "roll_spindle":cyl(3,76,(59,-45,247),(0,1,0)),
        "roll_driven_20T":cyl(R,15,(59,-43,247),(0,1,0)),
        "pitch_fork_medial":box((43,2,230),(3,35,36)),
        "pitch_fork_lateral":box((75,2,230),(3,35,36)),
        "pitch_fork_bridge":box((59,2,247),(35,6,4)),
        "pitch_bearing_medial_W618_6":ring(6.5,3,3.5,(43,2,219),(1,0,0)),
        "pitch_bearing_lateral_W618_6":ring(6.5,3,3.5,(73,2,219),(1,0,0)),
        "pitch_seat_medial":ring(8.5,6.5,5.5,(42,2,219),(1,0,0)),
        "pitch_seat_lateral":ring(8.5,6.5,5.5,(72,2,219),(1,0,0)),
        # Output spline is 12.2 mm below case centre in this orientation.
        "pitch_ST3250_case":box((59,33.5,231.2),(35,45.22,24.72)),
        "pitch_motor_output":cyl(3,30,(76,33.5,219),(1,0,0)),
        "pitch_drive_20T":cyl(R,15,(87,33.5,219),(1,0,0)),
        "pitch_belt_123_3MGT_15":belt((87,2,219),(87,33.5,219),"x"),
    }
    for nm,x in (("pitch_fork_medial",43),("pitch_fork_lateral",75)):
        roll[nm]=roll[nm]-cyl(6.5,5,(x-2.5,2,219),(1,0,0))
    thigh={
        "pitch_spindle":cyl(3,68,(40,2,219),(1,0,0)),
        "pitch_driven_20T":cyl(R,15,(87,2,219),(1,0,0)),
        "thigh_root_pad":box((88,2,218),(9,25,14)),
    }
    return {"fixed":fixed,"yaw":yaw,"roll":roll,"thigh":thigh}


def volume(shape):
    if shape is None: return 0.
    return sum(s.volume for s in shape.solids())


def both(a, other):
    try: return [volume(a & other),volume(other & a)],None
    except Exception as exc: return [None,None],f"{type(exc).__name__}: {str(exc)[:120]}"


def bbox_gap(a,other):
    aa,bb=a.bounding_box(),other.bounding_box()
    v=[max(0.,float(getattr(bb.min,k)-getattr(aa.max,k)),
                 float(getattr(aa.min,k)-getattr(bb.max,k))) for k in "XYZ"]
    return math.sqrt(sum(q*q for q in v))


def screen_pair(rows,pose,n,a,m,other,kind):
    if bbox_gap(a,other)>=GAP: return
    vals,err=both(a,other)
    if err: rows.append(dict(pose=pose,pair=[n,m],kind=kind,status="unknown",error=err));return
    if max(vals)>.05:
        rows.append(dict(pose=pose,pair=[n,m],kind=kind,status="penetration",
                         both_orders_mm3=[round(q,3) for q in vals]))
    else:
        try: d=float(a.distance_to(other))
        except Exception as exc:
            rows.append(dict(pose=pose,pair=[n,m],kind=kind,status="unknown_gap",
                             error=type(exc).__name__));return
        if d<GAP: rows.append(dict(pose=pose,pair=[n,m],kind=kind,
                                  status="tight_gap",gap_mm=round(d,4)))


def props(s,name=""):
    bb=s.bounding_box()
    v=volume(s)
    if "ST3250_case" in name:
        material,mass,origin="aluminum-case servo assembly",MOTOR_G,"Feetech listed unit mass"
    elif "W618_6" in name:
        material,mass,origin="stainless bearing assembly",1.8,"SKF catalogue unit mass"
    elif "pack" in name:
        material,mass,origin="LiPo pack assembly",107,"Tattu listed unit mass"
    elif "belt" in name:
        material,mass,origin="elastomer belt envelope",v*.0011,"assumed 1.1 g/cm3"
    elif "skin" in name or name in ("torso","pelvis_v","R_thigh","L_thigh"):
        material,mass,origin="printed polymer shell",v*.00124,"assumed 1.24 g/cm3"
    elif "spindle" in name or "motor_output" in name:
        material,mass,origin="steel shaft proxy",v*.00785,"assumed 7.85 g/cm3"
    else:
        material,mass,origin="aluminum support/pulley proxy",v*.0027,"assumed 2.7 g/cm3"
    return dict(valid=bool(s.is_valid),solids=len(s.solids()),volume_mm3=round(v,3),
                material=material,mass_g=round(mass,3),mass_basis=origin,
                bounds_mm=[[round(float(v),2) for v in bb.min],
                           [round(float(v),2) for v in bb.max]])


def make_shell():
    # Keep the prior one-correction Cove shell and its full 2 mm thigh.  The
    # previously measured overlap is a constraint, not silently machined away.
    return old.candidate_shell()


def frame(hw,side,yaw,roll,pitch):
    transformed={}
    for stage,group in hw.items():
        transformed[stage]={}
        for name,s in group.items():
            s=s if side==1 else s.mirror(b.Plane.YZ)
            transformed[stage][name]=s if stage=="fixed" else old.transform(
                s,yaw*side,roll,pitch,"thigh" if stage=="thigh" else stage,side)
    return transformed


def cavity_outside(case,outer):
    try:
        left=case
        for shell in outer: left=left-shell
        return round(volume(left),3),None
    except Exception as exc: return None,type(exc).__name__


def screen(parts,by_name):
    hw=right_hardware()
    pack=box((0,27,290),(60,30,30.5))  # one shared Tattu 4S850
    service=box((0,-18,369),(25,20,14))
    skins={n:by_name[n].shape for n in ("torso","pelvis_v","R_thigh","L_thigh")}
    # Source interior plus its skin recovers the original outer BRep for
    # explicit motor-case protrusion, independent of mere skin intersection.
    original,cavities=cove.build(cove.VARIANTS[0])
    source_outer={p.name:p.shape.fuse(cavities[p.name.replace("pelvis_v","pelvis")])
                  for p in original if p.name in ("torso","pelvis_v")}
    # The candidate's raised pelvis aperture alters the outer boundary only
    # by subtracting material; original outer still gives a conservative gate.
    outer=list(source_outer.values())
    rows=[];containment=[];interface=[]
    cases=("yaw_ST3250_case","roll_ST3250_case","pitch_ST3250_case")
    for pose,(yaw,roll,pitch) in POSES.items():
        print("BRep",pose,flush=True)
        f={}
        for side,tag in ((-1,"L"),(1,"R")):
            f[tag]=frame(hw,side,yaw,roll,pitch)
            flat={name:s for stage in f[tag].values() for name,s in stage.items()}
            thigh=old.transform(skins[f"{tag}_thigh"],yaw*side,roll,pitch,"thigh",side)
            dynamic_names=set(cases)|{
                "yaw_belt_123_3MGT_15","roll_belt_123_3MGT_15",
                "pitch_belt_123_3MGT_15","yaw_fork_front","yaw_fork_rear",
                "yaw_fork_bridge","pitch_fork_medial","pitch_fork_lateral",
                "pitch_fork_bridge","yaw_bulkhead_low","yaw_bulkhead_high"}
            for stage,group in f[tag].items():
                for name,shape in group.items():
                    if name in cases:
                        outside,err=cavity_outside(shape,outer)
                        containment.append(dict(pose=pose,part=f"{tag}_{name}",
                                                outside_source_outer_mm3=outside,error=err))
                    if pose=="stand" or name in dynamic_names:
                        for sn in ("torso","pelvis_v"):
                            screen_pair(rows,pose,f"{tag}_{name}",shape,sn,skins[sn],"hardware_skin")
                        screen_pair(rows,pose,f"{tag}_{name}",shape,"shared_pack",pack,"hardware_pack")
            for name,shape in flat.items():
                if name in dynamic_names:
                    screen_pair(rows,pose,f"{tag}_{name}",shape,f"{tag}_thigh_skin",thigh,"hardware_thigh")
            for name,shape in (("torso",skins["torso"]),("pelvis_v",skins["pelvis_v"]),
                               ("shared_pack",pack)):
                screen_pair(rows,pose,f"{tag}_thigh_skin",thigh,name,shape,"thigh_fixed")
            # Cases and belts must be separate from other cases and belts.
            for a,bb in (("yaw_ST3250_case","roll_ST3250_case"),
                         ("yaw_ST3250_case","pitch_ST3250_case"),
                         ("roll_ST3250_case","pitch_ST3250_case"),
                         ("yaw_belt_123_3MGT_15","roll_ST3250_case"),
                         ("roll_belt_123_3MGT_15","pitch_ST3250_case"),
                         ("yaw_belt_123_3MGT_15","pitch_belt_123_3MGT_15"),
                         ("yaw_belt_123_3MGT_15","yaw_fork_bridge"),
                         ("pitch_ST3250_case","pitch_fork_medial"),
                         ("pitch_ST3250_case","pitch_fork_lateral")):
                screen_pair(rows,pose,f"{tag}_{a}",flat[a],f"{tag}_{bb}",flat[bb],"drive_drive")
            # Check a load-carrying root connection, rather than inferring it
            # from a rendering.  A zero/unknown interface rejects the path.
            for a,bb in (("yaw_bulkhead_low","yaw_seat_low"),
                         ("yaw_spindle","yaw_fork_bridge"),
                         ("yaw_fork_front","roll_seat_front"),
                         ("pitch_fork_medial","pitch_seat_medial"),
                         ("pitch_spindle","thigh_root_pad"),
                         ("thigh_root_pad","thigh_skin")):
                aa=flat[a];bshape=thigh if bb=="thigh_skin" else flat[bb]
                vals,err=both(aa,bshape)
                interface.append(dict(pose=pose,side=tag,path=[a,bb],
                                      both_orders_mm3=None if err else [round(v,3) for v in vals],error=err))
        # Bilateral package and thigh crossing, one shared pack.
        for a in cases:
            screen_pair(rows,pose,"L_"+a,f["L"][next(st for st in hw if a in hw[st])][a],
                        "R_"+a,f["R"][next(st for st in hw if a in hw[st])][a],"bilateral")
        lth=old.transform(skins["L_thigh"],-yaw,roll,pitch,"thigh",-1)
        rth=old.transform(skins["R_thigh"],yaw,roll,pitch,"thigh",1)
        screen_pair(rows,pose,"L_thigh_skin",lth,"R_thigh_skin",rth,"bilateral")
    stats={pose:{"penetrations":sum(r["pose"]==pose and r["status"]=="penetration" for r in rows),
                 "unknown":sum(r["pose"]==pose and r["status"].startswith("unknown") for r in rows),
                 "tight_gaps":sum(r["pose"]==pose and r["status"]=="tight_gap" for r in rows)}
           for pose in POSES}
    return hw,pack,service,rows,containment,interface,stats


def render(parts,by_name):
    # Source and candidate: same camera, solid ivory parts, same distal pose.
    source,_=cove.build(cove.VARIANTS[0])
    src_mesh=[(p,cove.mesh(p.shape,p.color)) for p in source]
    new_mesh=[(p,cove.mesh(p.shape,p.color)) for p in parts]
    views=("front","side","three-quarter")
    w,h=290,340
    sheet=Image.new("RGB",(w*6,(h+25)*len(POSES)),"white")
    pen=ImageDraw.Draw(sheet)
    for row,(pose,angles) in enumerate(POSES.items()):
        base="crawl" if pose=="crawl" else "squat" if pose=="squat" else "stand"
        sm=cove.matrices(cove.VARIANTS[0],base)
        cm=cove.matrices(cove.VARIANTS[0],base)
        for side,tag in ((-1,"L"),(1,"R")):
            yaw,roll,pitch=angles[0]*side,angles[1],angles[2]
            p=np.asarray((side*59,2,247));q=np.asarray((side*59,2,219))
            mat=(cove.rotation((0,0,1),yaw,p)@
                 cove.rotation((0,1,0),roll,p)@
                 cove.rotation((1,0,0),pitch,q))
            orig=cove.rotation((1,0,0),cove.POSES[base]["hip"],(side*59,2,231))
            floor=cm["fixed"]
            for distal in (f"{tag}_shin",f"{tag}_boot"):
                cm[distal]=floor@mat@np.linalg.inv(orig)@np.linalg.inv(floor)@cm[distal]
            cm[f"{tag}_thigh"]=floor@mat
        for col,view in enumerate(views):
            for offset,label,meshes,mats in ((0,"Cove source",src_mesh,sm),
                                             (3,"fork coupon",new_mesh,cm)):
                tile=cove.render(meshes,mats,view,(w,h))
                sheet.paste(tile,((col+offset)*w,row*(h+25)+25))
                pen.text(((col+offset)*w+6,row*(h+25)+5),
                         f"{label} · {pose} · {view}",fill=(47,43,40))
    sheet.save(OUT/"matched-opaque.png",optimize=True)
    detail=Image.new("RGB",(760,len(POSES)*340),"white")
    detail_pen=ImageDraw.Draw(detail)
    for row,pose in enumerate(POSES):
        for j,col in enumerate((2,5)):
            tile=sheet.crop((col*w+50,row*(h+25)+135,
                             col*w+240,row*(h+25)+305))
            detail.paste(tile.resize((380,340),Image.Resampling.LANCZOS),
                         (j*380,row*340))
            detail_pen.text((j*380+8,row*340+8),
                            f"{'Cove' if j==0 else 'fork'} · {pose} · three-quarter",
                            fill=(40,36,32))
    detail.save(OUT/"hip-detail-opaque.png",optimize=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parts,by_name=make_shell()
    hw,pack,service,rows,containment,interface,stats=screen(parts,by_name)
    report=dict(candidate="parent-driven fork, first and only placement",
       verdict="REJECT: source-sized hip drives and fork fail static case/skin and case/case fit before duty qualification.",
       height_mm=500,wall_mm=2,clearance_gate_mm=GAP,poses_deg=POSES,
       axes_right_mm=dict(yaw_z=PIVOT,roll_y=PIVOT,pitch_x=PITCH),
       component_sources=dict(
         motor="https://www.feetechrc.com/en/562636.html",
         bearing="https://cdn.skfmediahub.skf.com/api/public/094466a66aaba637/pdf_preview_medium/094466a66aaba637_pdf_preview_medium.pdf",
         belt="https://www.gates.com/content/dam/gates/home/knowledge-center/resource-library/catalogs/korea/power-transmission-2015-products-catalogue-en.pdf",
         pack="https://www.genstattu.com/tattu-850mah-14-8v-75c-4s1p-lipo-battery-pack-with-xt60-plug.html"),
       motor=dict(model="Feetech ST-3250-C001",case_mm=[45.22,24.72,35],
                  rated_Nm_12V=1.569,mass_g=MOTOR_G,
                  unloaded_sec_per_60deg_12V=.133,
                  count_both_hips=6,motor_mass_both_hips_g=6*MOTOR_G),
       transmission=dict(type="Gates 123-3MGT-15 size envelope, 20T:20T",
                         center_mm=31.5,pitch_radius_mm=R,ratio=1,
                         belt_width_mm=BELT_WIDTH,
                         pitch_length_formula_mm=round(2*31.5+2*math.pi*R,2),
                         torque_at_output_Nm="unknown; below nameplate after losses"),
       bearing=dict(model="SKF W618/6",bore_mm=6,od_mm=13,width_mm=3.5,
                    station_center_spacing_mm=dict(yaw=18,roll=46,pitch=30),
                    count_both_hips=12),
       shared_pack=dict(model="Tattu 4S850",nominal_V=14.8,
                        reserve_mm=[60,30,30.5],center_mm=[0,27,290],mass_g=107,
                        regulated_12V_rail_required=True),
       shapes={stage:{name:props(s,name) for name,s in group.items()} for stage,group in hw.items()},
       skin={n:props(by_name[n].shape,n) for n in ("torso","pelvis_v","R_thigh","L_thigh")},
       pack_shape=props(pack,"pack"),service_shape=props(service,"service"),
       case_containment=containment,load_path_interfaces=interface,
       summary=stats,contacts=rows,
       limits=["Both Boolean operand orders are computed for all reported pairs; +0.05 mm3 is a rigid penetration.",
               "Shaft journals, seat fits, pulley teeth and belt engagement are intentional mating interfaces, not clearance pairs.",
               "The full source and candidate thigh are retained. Source distal pose is a visual continuation, not closed-loop gait validation.",
               "The sampled angles do not bound swept intermediate motion or elastic belt motion.",
               "Feetech rated torque does not qualify loaded speed, continuous heat, spline adapter, belt capacity, bearing life, fasteners or gait duty."])
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("opaque views",flush=True)
    render(parts,by_name)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=="__main__": main()
