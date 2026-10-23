"""Bounded 500/600 mm front-cavity ST3250 knee transfer screen.

The 20T:20T Gates 123-3MGT-15 is a catalog-sized routing candidate, not a
qualified cyclic knee transmission.  No source gives the complete short-loop
pulley, adapter and mounting combination a 1.569 N m continuous rating.
"""
from __future__ import annotations

import json
import math
import build123d as b
import trimesh

from rated_leg_coupon import OUT, ROOT, maybe_intersection, mesh, normalize_step, package, shell_shapes, vol

SOURCE=ROOT/"artifacts/sketch-ape-shell/exports/500mm/ape_shell_assembly.step"
PITCH_RADIUS=60/(2*math.pi)  # 20 tooth, 3 mm-pitch GT geometry


def cylinder_x(radius,x0,x1,y,z):
    return b.Solid.make_cylinder(radius,x1-x0,b.Plane(origin=(x0,y,z),z_dir=(1,0,0)))


def capsule(radius,x0,x1,knee,drive):
    dy,dz=drive[1]-knee[1],drive[2]-knee[2]
    span=math.hypot(dy,dz)
    angle=math.degrees(math.atan2(-dy,dz))
    bar=b.Pos((x0+x1)/2,(knee[1]+drive[1])/2,(knee[2]+drive[2])/2)*b.Rot(angle,0,0)*b.Box(x1-x0,2*radius,span)
    return cylinder_x(radius,x0,x1,knee[1],knee[2]).fuse(cylinder_x(radius,x0,x1,drive[1],drive[2]),bar)


def remote_parts(height):
    s=height/500
    knee=(60*s,-2*s,140*s)
    drive=(knee[0],knee[1]-6,knee[2]+math.sqrt(31.5**2-6**2))
    # Output is on the +X face of a source-sized 35 mm axial ST3250 case.
    center=(drive[0]-17.5,drive[1],drive[2]+12.2)
    motor=b.Pos(*center)*b.Box(35,24.72,45.22)
    x0,x1=knee[0]+3,knee[0]+18
    drive_pulley=cylinder_x(9,x0,x1,drive[1],drive[2])
    output_pulley=cylinder_x(9,x0,x1,knee[1],knee[2])
    belt=capsule(PITCH_RADIUS+1.5,x0,x1,knee,drive)-capsule(PITCH_RADIUS-1.0,x0-.1,x1+.1,knee,drive)
    shaft=cylinder_x(4,knee[0]-30,knee[0]+26,knee[1],knee[2])
    bearings=[]; seats=[]; supports=[]
    for offset in (-28,20):
        x=knee[0]+offset
        bearings.append(cylinder_x(8,x,x+5,knee[1],knee[2])-cylinder_x(4.1,x-.1,x+5.1,knee[1],knee[2]))
        seats.append(cylinder_x(10,x-.5,x+5.5,knee[1],knee[2])-cylinder_x(8.1,x-.6,x+5.6,knee[1],knee[2]))
        supports.append(b.Pos(x+2.5,knee[1]+9,knee[2]+12)*b.Box(5,6,25))
    yoke=b.Pos(knee[0]-5,knee[1],knee[2]-12)*b.Box(8,14,30)
    parts={"motor_case":motor,"20T_drive_pulley":drive_pulley,"20T_output_pulley":output_pulley,
           "15mm_GT3_belt_envelope":belt,"8mm_output_shaft":shaft,
           "bearing_A":bearings[0],"bearing_B":bearings[1],"seat_A":seats[0],"seat_B":seats[1],
           "support_A":supports[0],"support_B":supports[1],"moving_yoke":yoke}
    return dict(knee=knee,drive=drive,motor_center=center,parts=parts)


def source_parts(height,base):
    s=height/500
    names=("torso_","pelvis_","R_thigh_","R_shin_","R_foot_","R_sole","R_toe_",
           "R_hip_cuff","R_knee_cuff","R_ankle_cuff")
    return {name:shape.scale(s) if s!=1 else shape for name,shape in base.items() if name.startswith(names)}


def screen(height,base):
    shell=source_parts(height,base)
    remote=remote_parts(height)
    neighbor={m["name"]:m for m in package(height/350) if m["name"] in ("hip_roll","ankle_pitch")}
    outer,_=shell_shapes(500,1)
    s=height/500
    lobes=[outer[k][0].scale(s) if s!=1 else outer[k][0] for k in ("thigh_R","knee_cuff_R","shin_R")]
    rows=[]
    for name,shape in remote["parts"].items():
        remain=shape
        for lobe in lobes:
            if maybe_intersection(remain,lobe)>.001:remain=remain.cut(lobe)
        hits=[dict(part=n,overlap_mm3=round(maybe_intersection(shape,wall),3))
              for n,wall in shell.items() if maybe_intersection(shape,wall)>.05]
        cross=[dict(neighbor_joint=j,neighbor_part=p,overlap_mm3=round(maybe_intersection(shape,sh),3))
               for j,m in neighbor.items() for p,sh in m["parts"].items()
               if maybe_intersection(shape,sh)>.05]
        rows.append(dict(part=name,volume_mm3=round(vol(shape),3),
                         outside_adjacent_station_lobes_mm3=round(vol(remain),3),
                         intact_source_skin_hits=hits,hip_ankle_reservation_hits=cross))
    hard_fail=any(r["outside_adjacent_station_lobes_mm3"]>.05 or r["intact_source_skin_hits"] or r["hip_ankle_reservation_hits"] for r in rows)
    neighbor_fail=any(maybe_intersection(shape,wall)>.05 for m in neighbor.values()
                      for shape in m["parts"].values() for wall in shell.values())
    return dict(height_mm=height,source_shell="exact 500 mm STEP" if height==500 else "exact 500 mm STEP uniformly scaled 1.2; wall scales from 1 to 1.2 mm",
                knee_output_mm=[round(v,3) for v in remote["knee"]],
                motor_output_mm=[round(v,3) for v in remote["drive"]],
                motor_center_mm=[round(v,3) for v in remote["motor_center"]],
                center_distance_mm=31.5,parts=rows,
                hip_ankle_direct_reservations_also_hit_source_skin=neighbor_fail,
                static_gate_clear=not hard_fail and not neighbor_fail,
                coupled_motion="not_run_static_gate_failed" if hard_fail or neighbor_fail else "not_run_transmission_rating_unresolved"),remote,shell,neighbor


def export_500(remote,shell,neighbor):
    scene=trimesh.Scene()
    for name,shape in shell.items():mesh(scene,"skin_"+name,shape,(226,219,204,80))
    for name,shape in remote["parts"].items():
        color=(48,101,153,255) if name=="motor_case" else (194,87,70,255) if "belt" in name else (188,139,68,255)
        mesh(scene,"remote_knee_"+name,shape,color)
    for joint,m in neighbor.items():
        for name,shape in m["parts"].items():
            if name in ("motor_case","seat_A","seat_B","parent_bridge"):
                mesh(scene,joint+"_"+name,shape,(53,106,160,255) if name=="motor_case" else (113,151,123,255))
    scene.export(OUT/"front-belt-500.glb")
    step=OUT/"front-belt-500.step"
    b.export_step(b.Compound(children=list(remote["parts"].values())),str(step))
    normalize_step(step)


def main():
    if not SOURCE.exists():raise SystemExit(f"source shell STEP missing: {SOURCE}")
    OUT.mkdir(parents=True,exist_ok=True)
    base={x.label:x for x in b.import_step(str(SOURCE)).children}
    cases=[]
    for h in (500,600):
        print(f"front-cavity belt knee with direct hip/ankle: {h} mm",flush=True)
        case,remote,shell,neighbor=screen(h,base)
        cases.append(case)
        print("  static gate",case["static_gate_clear"],flush=True)
        if h==500:export_500(remote,shell,neighbor)
    report=dict(cases=cases,belt_source="https://www.gates.com/content/dam/gates/home/knowledge-center/resource-library/catalogs/korea/power-transmission-2015-products-catalogue-en.pdf",
                belt_identity="Gates 123-3MGT-15, 41 teeth, 3 mm pitch, 15 mm width; 20T:20T nominal 1:1 on 31.5 mm centers",
                drive_source="https://www.feetechrc.com/en/562636.html",
                rated_output_nm="unknown: no validated cyclic short-loop belt, 25T spline-pulley adapter, pretension and power-off retention rating",
                motor_rated_nm=round(16*.0980665,3),
                output_reserve_at_old_ideal_limit_nm="less than zero for any nonzero belt loss if the old 1.569 N m clip level were required; old clip is not a new-body minimum",
                mass_sensitivity_60_cm="Uniform 500-to-600 mm shell scale multiplies material volume 1.728 and characteristic lever arms 1.2; shell-only gravity moment sensitivity 2.074 relative to 500 mm; no new-body load or COM solved",
                verdict="Front cavity reorientation avoids the prior broad posterior belt path geometrically, but static skin and shared-leg gates fail; 600 mm is a sensitivity only. No coupled motion or transmission-duty claim.")
    (OUT/"front-belt-report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
