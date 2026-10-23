"""Exact 50 cm rated hip-roll/ankle packaging after the knee-skin coupon.

This is a bounded falsification coupon.  6 mm and 15 mm 20T:20T belt paths are
catalog-size candidates, not rated complete transmissions.  The two bearing
stations per output and cross-link skin are screened without silently cutting
the approved cosmetic shell.  No motion is run while the neutral gate fails.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import build123d as b
import trimesh

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/"artifacts/rated-hip-ankle-20261005"
sys.path.insert(0,str(HERE.parent/"swarm_rated_leg_skin_v2"))
from skin_v2_case import integration_shapes
sys.path.insert(0,str(HERE.parent/"swarm_rated_leg_drive"))
from rated_leg_coupon import MOTORS, BEARING, maybe_intersection, mesh, package, shell_shapes, vol
from front_belt_screen import PITCH_RADIUS, capsule, cylinder_x


def cylinder_y(radius,y0,y1,x,z):
    return b.Solid.make_cylinder(radius,y1-y0,b.Plane(origin=(x,y0,z),z_dir=(0,1,0)))


def capsule_y(radius,y0,y1,output,drive):
    dx,dz=drive[0]-output[0],drive[2]-output[2]
    span=math.hypot(dx,dz)
    angle=math.degrees(math.atan2(dx,dz))
    bar=b.Pos((output[0]+drive[0])/2,(y0+y1)/2,(output[2]+drive[2])/2)*b.Rot(0,angle,0)*b.Box(2*radius,y1-y0,span)
    return cylinder_y(radius,y0,y1,output[0],output[2]).fuse(
        cylinder_y(radius,y0,y1,drive[0],drive[2]),bar)


def hip_roll(width=6):
    scale=500/350
    output=(32*scale,0,158*scale)
    dx=-20.0
    drive=(output[0]+dx,-5,output[2]+math.sqrt(31.5**2-dx**2))
    # Radial long axis remains upright.  The case is 35 mm deep behind its
    # output face and does not cross the sagittal plane when mirrored.
    motor=b.Pos(drive[0],drive[1]+17.5,drive[2]+12.2)*b.Box(24.72,35,45.22)
    y0,y1=drive[1]-width,drive[1]
    belt=capsule_y(PITCH_RADIUS+1.5,y0,y1,output,drive)-capsule_y(PITCH_RADIUS-1,y0-.1,y1+.1,output,drive)
    parts={"motor_case":motor,"belt_envelope":belt,
           "drive_pulley":cylinder_y(9,y0,y1,drive[0],drive[2]),
           "output_pulley":cylinder_y(9,y0,y1,output[0],output[2]),
           "output_shaft":cylinder_y(3,-26,16,output[0],output[2])}
    for label,y in (("A",-23),("B",5)):
        parts["bearing_"+label]=cylinder_y(6.5,y,y+3.5,output[0],output[2])-cylinder_y(3.1,y-.1,y+3.6,output[0],output[2])
        parts["seat_"+label]=cylinder_y(8.5,y-.5,y+4,output[0],output[2])-cylinder_y(6.6,y-.6,y+4.1,output[0],output[2])
        parts["support_"+label]=b.Pos(output[0],y+2,output[2]+12)*b.Box(5,4,24)
    # Moving thigh root, intentionally distinct from fixed seat supports.
    parts["moving_yoke"]=b.Pos(output[0],12,output[2]-12)*b.Box(10,5,24)
    return dict(joint="hip_roll",model="ST3250",output=output,drive=drive,
                motor_center=(drive[0],drive[1]+17.5,drive[2]+12.2),
                belt_width_mm=width,parts=parts,
                fixed=tuple(n for n in parts if n not in ("output_pulley","output_shaft","moving_yoke")),
                moving=("output_pulley","output_shaft","moving_yoke"))


def ankle_pitch(width=6):
    scale=500/350
    output=(46*scale,4*scale,24*scale)
    drive=(75.0,output[1],output[2]+31.5)
    motor=b.Pos(drive[0]-17.5,drive[1],drive[2]+12.2)*b.Box(35,24.7,45.2)
    x0,x1=drive[0]+3,drive[0]+3+width
    belt=capsule(PITCH_RADIUS+1.5,x0,x1,output,drive)-capsule(PITCH_RADIUS-1,x0-.1,x1+.1,output,drive)
    parts={"motor_case":motor,"belt_envelope":belt,
           "drive_pulley":cylinder_x(9,x0,x1,drive[1],drive[2]),
           "output_pulley":cylinder_x(9,x0,x1,output[1],output[2]),
           "output_shaft":cylinder_x(3,35,95,output[1],output[2])}
    for label,x in (("A",40),("B",70)):
        parts["bearing_"+label]=cylinder_x(6.5,x,x+3.5,output[1],output[2])-cylinder_x(3.1,x-.1,x+3.6,output[1],output[2])
        parts["seat_"+label]=cylinder_x(8.5,x-.5,x+4,output[1],output[2])-cylinder_x(6.6,x-.6,x+4.1,output[1],output[2])
        parts["support_"+label]=b.Pos(x+1.75,output[1]+7,output[2]+12)*b.Box(3.5,5,24)
    parts["moving_yoke"]=b.Pos(output[0],output[1],output[2]-12)*b.Box(6,14,24)
    return dict(joint="ankle_pitch",model="ST3215",output=output,drive=drive,
                motor_center=(drive[0]-17.5,drive[1],drive[2]+12.2),
                belt_width_mm=width,parts=parts,fixed=tuple(n for n in parts if n not in ("output_pulley","output_shaft","moving_yoke")),
                moving=("output_pulley","output_shaft","moving_yoke"))


def hits(shape,skins):
    result={}
    for name,wall in skins.items():
        overlap=maybe_intersection(shape,wall)
        if overlap>.05:result[name]=round(overlap,3)
    return result


def outside(shape,outer,names):
    remaining=shape
    for name in names:
        if name in outer and maybe_intersection(remaining,outer[name][0])>.05:
            remaining=remaining.cut(outer[name][0])
    return round(vol(remaining),3)


def self_clashes(parts):
    # Mating shaft/bearing/seat interfaces and belt/pulley teeth are deliberate.
    # Everything below must have an independent free volume.
    incompatible=(("belt_envelope","bearing_A"),("belt_envelope","bearing_B"),
                  ("belt_envelope","seat_A"),("belt_envelope","seat_B"),
                  ("belt_envelope","support_A"),("belt_envelope","support_B"),
                  ("belt_envelope","moving_yoke"),("motor_case","moving_yoke"),
                  ("motor_case","bearing_A"),("motor_case","bearing_B"),
                  ("motor_case","seat_A"),("motor_case","seat_B"),
                  ("motor_case","support_A"),("motor_case","support_B"))
    return {f"{a}:{c}":round(v,3) for a,c in incompatible
            if (v:=maybe_intersection(parts[a],parts[c]))>.05}


def screen(module,skin,other_hardware,outer):
    parts=module["parts"]
    if module["joint"]=="hip_roll":
        local={n:s for n,s in skin.items() if n.startswith(("torso_","pelvis_","R_thigh_","R_hip_cuff"))}
    else:
        local={n:s for n,s in skin.items() if n.startswith(("R_shin_","R_ankle_cuff","R_foot_","R_sole","R_toe_"))}
    rows=[]
    for name,shape in parts.items():
        skin_hits=hits(shape,local)
        others={}
        for n,q in other_hardware.items():
            overlap=maybe_intersection(shape,q)
            if overlap>.05:others[n]=round(overlap,3)
        rows.append(dict(part=name,volume_mm3=round(vol(shape),3),
                         skin_hits_mm3=skin_hits,
                         min_skin_gap_mm=round(min(shape.distance(w) for w in local.values()),3),
                         knee_hardware_hits_mm3=others))
    module_report={k:v for k,v in module.items() if k!="parts"}
    allowed=("pelvis","torso","hip_cuff_R","thigh_R") if module["joint"]=="hip_roll" else ("shin_R","ankle_cuff_R","boot_R")
    containment={name:outside(parts[name],outer,allowed) for name in ("motor_case","belt_envelope")}
    clashes=self_clashes(parts)
    module_report.update(parts=rows,source_outer_residual_mm3=containment,self_clashes_mm3=clashes,
                         local_neutral_gate_clear=all(not r["skin_hits_mm3"] and r["min_skin_gap_mm"]>=.5
                                                      and not r["knee_hardware_hits_mm3"] for r in rows)
                                                  and not any(v>.05 for v in containment.values()) and not clashes,
                         pose_screen="not_run_static_gate_failed")
    return module_report


def direct_baseline(skin):
    direct={m["name"]:m for m in package(500/350) if m["name"] in ("hip_roll","ankle_pitch")}
    return {name:dict(motor_case_skin_hits_mm3=hits(m["parts"]["motor_case"],skin),
                      output_mm=m["joint"],motor_center_mm=m["motor_center"])
            for name,m in direct.items()}


def export_scene(case,hip,ankle):
    scene=trimesh.Scene()
    skin=case["revised_skin"]
    for n,s in skin.items():
        if n.startswith(("torso_","pelvis_","R_thigh_","L_thigh_","R_hip_cuff","L_hip_cuff","R_knee_cuff","R_shin_","R_ankle_cuff","R_foot_","R_sole","R_toe_")):
            mesh(scene,"skin_"+n,s,(226,219,205,130))
    for module in (hip,ankle):
        for n,s in module["parts"].items():
            col=(53,105,147,255) if n=="motor_case" else (187,78,64,255) if "belt" in n else (183,142,76,255)
            mesh(scene,module["joint"]+"_"+n,s,col)
            if module["joint"]=="hip_roll":
                mesh(scene,"left_hip_roll_"+n,s.mirror(b.Plane.YZ),col)
    scene.export(OUT/"rated-hip-ankle-500.glb")
    b.export_step(b.Compound(children=[*hip["parts"].values(),
                             *(s.mirror(b.Plane.YZ) for s in hip["parts"].values()),
                             *ankle["parts"].values()]),
                  str(OUT/"rated-hip-ankle-500.step"))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    case=integration_shapes(500)
    skin=case["revised_skin"]
    outer,_=shell_shapes(500,2)
    knee_hardware={**case["fixed_knee_hardware"],**case["moving_knee_hardware"]}
    selected_hip,selected_ankle=hip_roll(6),ankle_pitch(6)
    modules=[]
    for belt_width in (6,15):
        for builder in (hip_roll,ankle_pitch):
            module=builder(belt_width)
            print(module["joint"],belt_width,"mm belt",flush=True)
            modules.append(screen(module,skin,knee_hardware,outer))
    right=selected_hip["parts"]["motor_case"]
    left=right.mirror(b.Plane.YZ)
    left_parts={n:s.mirror(b.Plane.YZ) for n,s in selected_hip["parts"].items()}
    left_skin={n:s for n,s in skin.items() if n.startswith(("torso_","pelvis_","L_thigh_","L_hip_cuff"))}
    bilateral=dict(right_left_hip_motor_overlap_mm3=round(maybe_intersection(right,left),3),
                   right_left_hip_motor_gap_mm=round(right.distance(left),3),
                   right_left_hip_belt_overlap_mm3=round(maybe_intersection(selected_hip["parts"]["belt_envelope"],
                       selected_hip["parts"]["belt_envelope"].mirror(b.Plane.YZ)),3),
                   left_hip_skin_hits_mm3={part:row for part,shape in left_parts.items()
                       if (row:=hits(shape,left_skin))},
                   left_motor_outside_source_outer_mm3=outside(left_parts["motor_case"],outer,
                       ("pelvis","torso","hip_cuff_L","thigh_L")),
                   right_left_hip_hardware_clashes_mm3={f"{a}:{c}":round(v,3)
                       for a,shape in selected_hip["parts"].items() for c,other in left_parts.items()
                       if (v:=maybe_intersection(shape,other))>.05})
    bilateral["bilateral_hip_neutral_gate_clear"]=modules[0]["local_neutral_gate_clear"] and not bilateral["left_hip_skin_hits_mm3"] and not bilateral["right_left_hip_hardware_clashes_mm3"] and bilateral["left_motor_outside_source_outer_mm3"]<=.05
    cross={"hip_vs_ankle_hardware_overlap_mm3":round(sum(maybe_intersection(a,b)
                  for a in selected_hip["parts"].values() for b in selected_ankle["parts"].values()),3)}
    data=dict(target_height_mm=500,source_shell="exact saved 500 mm 2 mm wall STEP plus T-4cmz88 local knee contour",
              direct_baseline=direct_baseline(skin),modules=modules,bilateral_occupancy=bilateral,
              intermodule=cross,
              sources=dict(ST3250=MOTORS["ST3250"]["source"],ST3215=MOTORS["ST3215"]["source"],
                           W618_6=BEARING["source"],Gates_3MGT="https://www.gates.com/content/dam/gates/home/knowledge-center/resource-library/catalogs/korea/power-transmission-2015-products-catalogue-en.pdf"),
              torque=dict(hip_motor_rated_nm=MOTORS["ST3250"]["rated_nm"],
                          ankle_motor_rated_nm=MOTORS["ST3215"]["rated_nm"],
                          belt_output_rating="unknown: no qualified 25T adapter, pulley, tension/retention and cyclic 123 mm loop combination",
                          speed="ST3250 0.133 s/60° and ST3215 0.222 s/60° are unloaded at 12 V; loaded speed curves not used",
                          heat="ST3250 aluminum case can conduct to a yet-unbuilt frame; ST3215 PA+GF case and mounted temperature history unqualified"),
              verdict="Both relocated source-size motors fit inside adjacent source outer lobes, but 6/15 mm belt/output/support packages cut the exact source skin. Mirrored hip hardware has no inter-side collision but both sides cut thigh skin. The ankle support blocks also cut its motor case. Neither local static gate passes, so hip/ankle motion and aesthetic acceptance are withheld; no rated belt-output claim.")
    (OUT/"report.json").write_text(json.dumps(data,indent=2)+"\n")
    export_scene(case,selected_hip,selected_ankle)
    shutil.copyfile(HERE/"viewer.html",OUT/"index.html")
    print("wrote",OUT/"report.json",flush=True)


if __name__=="__main__":main()
