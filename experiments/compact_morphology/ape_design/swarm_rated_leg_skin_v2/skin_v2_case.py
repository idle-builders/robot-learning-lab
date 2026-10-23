"""Local 2 mm shell contour trial for a source-sized, remote ST3250 knee.

The output bearing/seat path and source cosmetic skins are kept as separate
solids.  Positive hard-part clearance is required before any pose sweep.
This remains a geometry coupon, not a rated belt or printable leg.
"""
from __future__ import annotations

import contextlib
import argparse
import io
import json
import math
import shutil
import sys
from pathlib import Path

import build123d as b
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "artifacts/rated-leg-skin-v2-20261005"
SOURCE = ROOT / "artifacts/sketch-ape-shell/exports/500mm/ape_shell_assembly.step"
sys.path.insert(0, str(ROOT / "experiments/sketch_ape_shell"))
from build import Study
sys.path.insert(0, str(HERE.parent / "swarm_rated_leg_drive"))
from rated_leg_coupon import maybe_intersection, mesh, package, vol
from front_belt_screen import capsule, cylinder_x, PITCH_RADIUS


def derived_shell(height: int):
    if height == 500:
        return {p.label: p for p in b.import_step(str(SOURCE)).children}, "saved exact Study(500,2,.65) STEP"
    # Study bounds its public height to 500.  Build its source-coordinate
    # stations with wall and seam pre-divided by 1.2, then scale 1.2: the
    # resulting dimensions equal Study(600,2,.65) with a *2 mm* final wall.
    with contextlib.redirect_stdout(io.StringIO()):
        parts = Study(500, wall=5/3, gap=.65/1.2).build().parts
    return {p["name"]: p["shape"].scale(1.2) for p in parts}, \
        "source-derived Study(500,5/3,.65/1.2) x1.2; exact 600 mm, 2 mm wall, .65 mm seam"


def contoured_thigh(height: int, lateral_source_mm: float = 6.0):
    """Widen only the lateral lower thigh; preserve the medial silhouette.

    The delta is tapered over anatomical source loft stations, not a bolted-on
    circular cup.  Wall and front/back seam remain independent physical dims.
    """
    ratio = height/350
    study = Study(500, wall=2 if height == 500 else 5/3,
                  gap=.65 if height == 500 else .65/1.2)
    # At 600 output, source coordinates gain another factor 1.2.
    station = ((87,42,3,16,18,0), (98,43,3,20,22,.9),
               (124,39,3,24,25,1), (145,35,4,22,23,.35),
               (157,33,4,16,18,0), (163,31,4,8,10,0))
    rows = [(z,x+lateral_source_mm*w/2,y,rx+lateral_source_mm*w/2,ry)
            for z,x,y,rx,ry,w in station]
    shell,_ = study.hollow_loft(rows)
    g=study.gap/2
    front = shell & b.Pos(0,(-100+3-g)/2,150)*b.Box(360,103-g,350)
    back = shell & b.Pos(0,(3+g+100)/2,150)*b.Box(360,97-g,350)
    factor = 1 if height == 500 else 1.2
    return {"R_thigh_front":front.scale(500/350*factor),
            "R_thigh_back":back.scale(500/350*factor)}, rows


def knee_package(height: int):
    anatomy=height/500
    # Motor/belt stay manufactured size. The axis follows the anatomical
    # knee height and lateral offset; motor is moved proximally by 31.5 mm.
    knee=(66*anatomy,2*anatomy,146*anatomy)
    drive=(knee[0],knee[1],knee[2]+31.5)
    motor=b.Pos(drive[0]-17.5,drive[1],drive[2]+12.2)*b.Box(35,24.72,45.22)
    x0,x1=knee[0]+3,knee[0]+18
    belt=capsule(PITCH_RADIUS+1.5,x0,x1,knee,drive)-capsule(PITCH_RADIUS-1,x0-.1,x1+.1,knee,drive)
    fixed={"motor":motor,"belt":belt,
           "drive_pulley":cylinder_x(9,x0,x1,drive[1],drive[2])}
    moving={"output_pulley":cylinder_x(9,x0,x1,knee[1],knee[2]),
            "shaft":cylinder_x(4,knee[0]-30,knee[0]+26,knee[1],knee[2])}
    for label,offset in (("A",-28),("B",20)):
        x=knee[0]+offset
        fixed["bearing_"+label]=cylinder_x(8,x,x+5,knee[1],knee[2])-cylinder_x(4.1,x-.1,x+5.1,knee[1],knee[2])
        fixed["seat_"+label]=cylinder_x(10,x-.5,x+5.5,knee[1],knee[2])-cylinder_x(8.1,x-.6,x+5.6,knee[1],knee[2])
        fixed["support_"+label]=b.Pos(x+2.5,knee[1]+9,knee[2]+12)*b.Box(5,6,25)
    # Slender internal yoke crossing a narrow cuff aperture.
    moving["yoke"]=b.Pos(knee[0]-5,knee[1],knee[2]-18)*b.Box(6,10,40)
    # The internal pad intersects both original shin half-shells without
    # protruding beyond their source outer loft at 500 mm.  This is an
    # attachment envelope, not a strength or fastener proof.
    moving["shin_cross_pad"]=b.Pos(61*anatomy,5*anatomy,109*anatomy)*b.Box(12,51.5*anatomy,5)
    return knee,drive,fixed,moving


def row_hits(shape, skins):
    return {name:round(maybe_intersection(shape, wall),3) for name,wall in skins.items()
            if maybe_intersection(shape,wall)>.05}


def integration_shapes(height: int = 500):
    """Importable local knee interfaces for the shared-leg integration task.

    Returns exact skin solids and component envelopes, separated by parent
    frame.  The ankle direct-drive reservation is returned separately and is
    *not* a fitted ankle.  This function does not imply a passing gate.
    """
    source,provenance=derived_shell(height)
    thigh,stations=contoured_thigh(height)
    knee,drive,fixed,moving=knee_package(height)
    # One small, nonstructural cuff-edge revision: a recessed yoke slot with
    # 0.6 mm nominal margin on every box face. A gaiter must seal this slit.
    yoke_center=(knee[0]-5,knee[1],knee[2]-18)
    cuff_slot=b.Pos(*yoke_center)*b.Box(7.2,11.2,41.2)
    cuff=source["R_knee_cuff"]-cuff_slot
    # The source shin rim sits only 0.228 mm from the source cuff.  Removing
    # 0.5 output mm of its very top gives a measured >=0.7 mm neutral gap,
    # without changing the visible calf loft below the rim.
    shin_top=83*height/350-.5*height/500
    lower_clip=b.Pos(0,0,shin_top-200)*b.Box(500,400,400)
    shin={n:source[n]&lower_clip for n in ("R_shin_front","R_shin_back")}
    skin={**source,**thigh,**shin,"R_knee_cuff":cuff}
    neighbor={m["name"]:m for m in package(height/350) if m["name"] in ("hip_roll","ankle_pitch")}
    return dict(height_mm=height,knee_axis_mm=knee,motor_axis_mm=drive,
                source_skin=source,revised_skin=skin,cuff_slot=cuff_slot,
                fixed_knee_hardware=fixed,moving_knee_hardware=moving,
                fixed_knee_skin={n:skin[n] for n in skin if n.startswith(("R_thigh_","R_knee_cuff"))},
                moving_shin_skin={n:skin[n] for n in skin if n.startswith(("R_shin_","R_ankle_cuff","R_foot_","R_toe_","R_sole"))},
                hip_ankle_reservations=neighbor,source_provenance=provenance,
                contour_stations_source_mm=stations)


def turn(shape,axis,angle):
    return b.Pos(*axis)*b.Rot(angle,0,0)*b.Pos(*(-v for v in axis))*shape


def pose_screen(case,angles=(0,2,15,30,60,90,113,115)):
    fixed={**case["fixed_knee_skin"],**case["fixed_knee_hardware"]}
    # The output pulley/shaft deliberately run in the fixed belt/bearings.
    # Their mesh and bore interfaces need mechanical qualification, not a
    # nonmating air gap.  Screen only shin shell, yoke, and cross pad here.
    moving={**case["moving_shin_skin"],
            "yoke":case["moving_knee_hardware"]["yoke"],
            "shin_cross_pad":case["moving_knee_hardware"]["shin_cross_pad"]}
    fixed_compound=b.Compound(children=list(fixed.values()))
    rows=[]
    for angle in angles:
        hits=[]; minimum=math.inf
        for name,shape in moving.items():
            posed=turn(shape,case["knee_axis_mm"],angle)
            overlap=maybe_intersection(posed,fixed_compound)
            gap=posed.distance(fixed_compound)
            minimum=min(minimum,gap)
            if overlap>.05 or gap<.5:
                hits.append(dict(part=name,overlap_mm3=round(overlap,3),gap_mm=round(gap,3)))
        rows.append(dict(flex_deg=angle,minimum_gap_mm=round(minimum,3),hard_gap_pass=not hits,hits=hits))
    return rows


def screen(height):
    case=integration_shapes(height)
    source=case["source_skin"]
    skin=case["revised_skin"]
    fixed=case["fixed_knee_hardware"]
    moving=case["moving_knee_hardware"]
    knee=case["knee_axis_mm"]
    drive=case["motor_axis_mm"]
    neighbor=case["hip_ankle_reservations"]
    local_names=[n for n in skin if n.startswith(("R_thigh_","R_knee_cuff","R_shin_"))]
    local={n:skin[n] for n in local_names}
    rows=[]
    for name,shape in {**fixed,**moving}.items():
        check_skin=case["fixed_knee_skin"] if name=="shin_cross_pad" else local
        rows.append(dict(part=name,skin_hits_mm3=row_hits(shape,check_skin),
                         min_skin_gap_mm=round(min(shape.distance(x) for x in check_skin.values()),3)))
    yoke_cuff=maybe_intersection(moving["yoke"],skin["R_knee_cuff"])
    # Negative clearance is never silently converted to a pass by cutting a
    # source cover.  A future gaiter design must replace this cuff coherently.
    direct_hits={joint:{part:row_hits(shape,skin) for part,shape in m["parts"].items()
                        if row_hits(shape,skin)} for joint,m in neighbor.items()}
    cross_hits={joint:{part:round(sum(maybe_intersection(shape,q) for q in m["parts"].values()),3)
                       for part,shape in {**fixed,**moving}.items()
                       if sum(maybe_intersection(shape,q) for q in m["parts"].values())>.05}
                for joint,m in neighbor.items()}
    pad= moving["shin_cross_pad"]
    pad_contacts={n:round(maybe_intersection(pad,skin[n]),3) for n in ("R_shin_front","R_shin_back")}
    fixed_compound=b.Compound(children=[*case["fixed_knee_skin"].values(),*fixed.values()])
    moving_skin_neutral=[]
    for name,shape in case["moving_shin_skin"].items():
        moving_skin_neutral.append(dict(part=name,
            overlap_mm3=round(maybe_intersection(shape,fixed_compound),3),
            gap_mm=round(shape.distance(fixed_compound),3)))
    local_clear=all(not r["skin_hits_mm3"] and r["min_skin_gap_mm"]>=.5 for r in rows) \
        and len(skin["R_knee_cuff"].solids())==1 and all(not d for d in cross_hits.values()) \
        and all(v>1 for v in pad_contacts.values()) \
        and all(r["overlap_mm3"]<=.05 and r["gap_mm"]>=.5 for r in moving_skin_neutral)
    static_clear=local_clear and all(not d for d in direct_hits.values())
    poses=pose_screen(case) if local_clear and height==500 else []
    report=dict(height_mm=height,nominal_wall_mm=2,final_seam_gap_mm=.65,
                shell_provenance=case["source_provenance"],contour_delta_source_mm=6,
                unmodified_source_shell_volume_cm3=round(sum(vol(x) for x in source.values())/1000,3),
                contour_stations_source_mm=case["contour_stations_source_mm"],knee_axis_mm=knee,motor_axis_mm=drive,
                parts=rows,
                cuff_slot_mm=[7.2,11.2,41.2],
                cuff_material_removed_mm3=round(vol(source["R_knee_cuff"])-vol(skin["R_knee_cuff"]),3),
                cuff_solids_after_slot=len(skin["R_knee_cuff"].solids()),
                yoke_vs_intact_cuff_mm3=round(maybe_intersection(moving["yoke"],source["R_knee_cuff"]),3),
                yoke_vs_revised_cuff_mm3=round(yoke_cuff,3),
                cross_pad_to_shin_skin_mm3=pad_contacts,
                neutral_moving_skin_vs_fixed=moving_skin_neutral,
                source_shin_top_trim_mm=.5*height/500,
                other_joint_reservation_vs_skin_mm3=direct_hits,
                knee_vs_other_joint_reservation_mm3=cross_hits,
                local_knee_static_gate_clear=local_clear,
                static_gate_clear=static_clear,
                local_knee_pose_samples=poses,
                coupled_motion="not_run_local_static_gate_failed" if not local_clear else \
                    "local_knee_only; full_leg_static_gate_failed")
    return report,source,skin,fixed,moving


def export_case(height,source,skin,fixed,moving):
    scene=trimesh.Scene()
    selected={n:s for n,s in skin.items() if n.startswith(("R_thigh_","R_knee_cuff","R_shin_","R_foot_","R_sole","R_toe_"))}
    for n,s in selected.items():mesh(scene,"skin_"+n,s,(227,220,205,255))
    for n,s in fixed.items():mesh(scene,"fixed_"+n,s,(61,104,138,255))
    for n,s in moving.items():mesh(scene,"moving_"+n,s,(183,120,73,255))
    scene.export(OUT/f"leg-{height}.glb")
    for variant,parts in (("source",source),("revised",skin)):
        opaque=trimesh.Scene()
        for n,s in parts.items():
            if n.startswith(("R_thigh_","R_knee_cuff","R_shin_","R_ankle_cuff","R_foot_","R_sole","R_toe_")):
                mesh(opaque,"skin_"+n,s,(227,220,205,255) if "cuff" not in n else (105,106,98,255))
        opaque.export(OUT/f"{variant}-opaque-{height}.glb")
    b.export_step(b.Compound(children=[*selected.values(),*fixed.values(),*moving.values()]),
                  str(OUT/f"leg-{height}.step"))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--heights",type=int,nargs="+",choices=(500,600),default=(500,600))
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    reports=[]
    for h in args.heights:
        print("exact 2 mm shell",h,flush=True)
        report,source,skin,fixed,moving=screen(h)
        reports.append(report)
        export_case(h,source,skin,fixed,moving)
        print(" static clear",report["static_gate_clear"],"revised yoke cuff",report["yoke_vs_revised_cuff_mm3"],flush=True)
    shell_by_height={r["height_mm"]:r["unmodified_source_shell_volume_cm3"] for r in reports}
    mass=dict(shell_source_2mm_cm3_by_height=shell_by_height,
              solid_PLA_density_g_cm3=1.24,selected_bilateral_motors_g=408,
              shell_plus_six_selected_leg_motors_g_by_height={h:round(v*1.24+408,1) for h,v in shell_by_height.items()},
              missing="actual 688 bearing mass, other axes, pack, chassis, belt, supports, shafts, electronics, cables, fasteners")
    data=dict(cases=reports,mass=mass,torque=dict(ST3250_rated_Nm=1.569,ST3215_rated_Nm=.981,
              belt_output_rated_Nm="unqualified; pulley, adapter, pretension and cyclic belt rating missing",
              shell_only_gravity_moment_ratio_600_over_500=round(
                  shell_by_height.get(600,942.566)/shell_by_height.get(500,645.919)*1.2,3),
              old_body_warning="Old 2.265 kg ideal-servo body clipped hip roll at ±1.569 Nm in 73%/57% of forward frames; not a minimum for a new body."),
              verdict="Local knee neutral fit and pose evidence are per-case. Full static fit still fails because the direct hip/ankle reservations cut the shared shell; no whole-leg motion or buildable transmission claim.")
    (OUT/"report.json").write_text(json.dumps(data,indent=2)+"\n")
    shutil.copyfile(HERE/"viewer.html",OUT/"index.html")
    shutil.copyfile(HERE/"opaque_viewer.html",OUT/"opaque.html")


if __name__=="__main__":main()
