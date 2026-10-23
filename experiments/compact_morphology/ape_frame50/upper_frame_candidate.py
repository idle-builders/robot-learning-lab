"""Exposed 50 cm ape torso/arm coupon on the supported pelvis.

The Cove axes remain the anatomy datum. Cases are catalog envelopes, while
the rails, bearing stations and drive paths are candidate printed/stock parts.
This screens sampled rigid motion, not shaft fatigue or a print-ready assembly.
Coordinates are Cove millimetres: X right, Y rear, Z up.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import sys

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.pelvis_tube_candidate import (
    candidate as pelvis_candidate, intersection, mesh,
)
from experiments.compact_morphology.ape_frame50.shin_ankle_package import (
    package as lower_package, foot as lower_foot,
)
from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as hip

HERE = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/ape-frame50-upper-frame"
YAW = (82., 2., 347.)
ROLL = (105., 2., 347.)
PITCH = (119., 2., 347.)
ELBOW = (128., 0., 273.)
HAND = (134., -5., 132.)
NECK = (0., 1., 397.)
# Feetech ST3215 drawing, page 6: the 25T output is normal to the broad
# 45.23 x 24.73 mm face, 12.5 mm from one end. Keep the case and output
# axis together when changing a drive route; 12.5 mm is NOT the offset from
# the case centre (that offset is 10.115 mm).
ST_LONG, ST_SHORT, ST_OUTPUT_DEPTH = 45.23, 24.73, 36.5
ST_OUTPUT_OFFSET = ST_LONG / 2 - 12.5
# Gates 3MGT 20-groove equal-pulley centre-distance and torque tables are the
# routing datum. A 3 mm pitch, 20T pulley has a 9.55 mm pitch radius. The 9 mm
# width is a screen at 2x the ST3215 rated torque, not a loaded belt selection.
BELT_PITCH_RADIUS = 20 * 3 / (2 * math.pi)
BELT_PLANE_THICKNESS = 9
BELT_RADIAL_THICKNESS = 2.4


def rod(a, b, radius=3):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return hip.cyl(radius, float(np.linalg.norm(b-a)), tuple(a), tuple(b-a))


def belt_straight(a, b, axis):
    a, b = np.asarray(a,dtype=float), np.asarray(b,dtype=float)
    direction = b-a
    return bd.Plane(origin=tuple((a+b)/2),x_dir=tuple(direction/np.linalg.norm(direction)),
                    z_dir=tuple(axis)) * bd.Box(float(np.linalg.norm(direction)),
                                                BELT_RADIAL_THICKNESS,BELT_PLANE_THICKNESS)


def drive_route(group, name, motor, joint, axis):
    """Add the two coaxial pulley claims and straight equal-pulley tangents."""
    motor, joint = np.asarray(motor,dtype=float), np.asarray(joint,dtype=float)
    axis = np.asarray(axis,dtype=float)
    delta = joint-motor
    tangent = np.cross(axis,delta)
    tangent = tangent / np.linalg.norm(tangent) * BELT_PITCH_RADIUS
    group[f"{name}_motor_pulley_20T_claim"] = hip.cyl(
        BELT_PITCH_RADIUS,BELT_PLANE_THICKNESS,tuple(motor-axis*2),tuple(axis))
    group[f"{name}_belt_upper_claim"] = belt_straight(motor+tangent,joint+tangent,axis)
    group[f"{name}_belt_lower_claim"] = belt_straight(motor-tangent,joint-tangent,axis)
    return hip.cyl(BELT_PITCH_RADIUS,BELT_PLANE_THICKNESS,
                   tuple(joint-axis*2),tuple(axis))


def turn(shape, pivot, axis, degrees):
    p = tuple(float(v) for v in pivot)
    r = bd.Rot(*(degrees if i == axis else 0 for i in range(3)))
    return bd.Pos(*p) * r * bd.Pos(*(-v for v in p)) * shape


def pose(shape, stage, angles):
    yaw, roll, pitch, elbow = angles
    if stage == "forearm":
        shape = turn(shape, ELBOW, 0, elbow)
    if stage in ("arm", "forearm"):
        shape = turn(shape, PITCH, 0, pitch)
    if stage in ("pitch", "arm", "forearm"):
        shape = turn(shape, ROLL, 1, roll)
    if stage in ("yaw", "pitch", "arm", "forearm"):
        shape = turn(shape, YAW, 2, yaw)
    return shape


def build():
    # Torso structure joins the existing front-spine stock at Z=315-321.
    # Its front plane is forward of the battery and motor-case corridors.
    torso = {
        "chest_left_rail": rod((0,-32,313), (82,-32,341), 4),
        "chest_right_rail": rod((0,-32,313), (-82,-32,341), 4),
        "chest_crossbar": hip.box((0,-32,341), (168,8,8)),
        "neck_spine": rod((0,-32,315), (0,-32,400), 4),
        "neck_bridge": rod((0,-32,395), (0,1,397), 4),
        "neck_lower_seat_bridge": rod((0,-32,385), (0,1,385), 4),
        "neck_lower_seat": hip.ring(8.5,6.5,5,(0,1,385)),
        "neck_upper_seat": hip.ring(8.5,6.5,5,(0,1,405)),
        "head_yaw_XL330_case": hip.box((0,15,413), (20,34,26)),
        "head_motor_standoff": rod((0,-32,400),(0,0,413),3),
        "battery_4s850": hip.box((0,14,270), (30,60,30.5)),
        "battery_front_cradle": rod((-18,-20,265),(18,-20,265),3),
        "battery_rear_cradle": rod((-18,48,265),(18,48,265),3),
        "battery_left_side": rod((18,-20,265),(18,48,265),3),
        "battery_right_side": rod((-18,-20,265),(-18,48,265),3),
        "battery_front_left_support": rod((0,-32,265),(18,-20,265),3),
        "battery_front_right_support": rod((0,-32,265),(-18,-20,265),3),
        "imu_reserve": hip.box((0,0,296), (20,13,3)),
        "imu_bracket": rod((0,-32,296),(0,-11,296),3),
        "yaw_motor_standoff": rod((45,-32,335),(45,2,335),3),
    }
    # Both fixed camera brackets rotate with the one-DOF head. The skull is
    # only a sparse support ring here; the approved face/skin is not included.
    head = {
        "neck_spindle": hip.cyl(3,43,(0,1,384)),
        "head_lower_hoop": hip.ring(43,40,4,(0,0,423)),
        "head_upper_hoop": hip.ring(43,40,4,(0,0,477)),
        "head_spoke_left": rod((0,1,425),(40,1,425),3),
        "head_spoke_right": rod((0,1,425),(-40,1,425),3),
        "head_front_left_post": rod((-34,-27,424),(-34,-27,478),3),
        "head_front_right_post": rod((34,-27,424),(34,-27,478),3),
        "head_front_bridge": rod((-34,-27,450),(34,-27,450),3),
        "upper_camera_module": hip.box((15,-49,457),(24,25,12.4)),
        "lower_camera_module": hip.box((-15,-49,433),(24,25,12.4)),
        "upper_camera_bracket": rod((15,-27,450),(15,-34,457),3),
        "lower_camera_bracket": rod((-15,-27,450),(-15,-34,433),3),
    }
    fixed = {
        "yaw_ST3215_case": hip.box((45-ST_OUTPUT_OFFSET,2,339),
                                    (ST_LONG,ST_SHORT,ST_OUTPUT_DEPTH)),
        "yaw_motor_output_25T_envelope": hip.cyl(3,3.4,(45,2,339+ST_OUTPUT_DEPTH/2)),
        "yaw_bearing_lower_6x13x3p5": hip.ring(6.5,3,3.5,(82,2,328)),
        "yaw_bearing_upper_6x13x3p5": hip.ring(6.5,3,3.5,(82,2,349)),
        "yaw_seat_lower": hip.ring(8.5,6.5,5,(82,2,327)),
        "yaw_seat_upper": hip.ring(8.5,6.5,5,(82,2,348)),
        "yaw_seat_to_chest_lower": rod((82,-32,329),(82,-2,329),3),
        "yaw_seat_to_chest_upper": rod((82,-32,351),(82,-2,351),3),
        "yaw_lower_chest_anchor": rod((82,-32,329),(82,-32,341),3),
        "yaw_upper_chest_anchor": rod((82,-32,341),(82,-32,351),3),
    }
    yaw = {
        "yaw_spindle": hip.cyl(3,53,(82,2,327)),
        "yaw_spindle_front_web": rod((82,2,352),(82,-16,352),3),
        "yaw_spindle_rear_web": rod((82,2,352),(82,20,352),3),
        "yaw_to_roll_front": rod((82,-16,352),(105,-16,352),3),
        "yaw_to_roll_rear": rod((82,20,352),(105,20,352),3),
        "roll_bearing_front_6x13x3p5": hip.ring(6.5,3,3.5,(105,-20,347),(0,1,0)),
        "roll_bearing_rear_6x13x3p5": hip.ring(6.5,3,3.5,(105,20,347),(0,1,0)),
        "roll_seat_front": hip.ring(8.5,6.5,5,(105,-21,347),(0,1,0)),
        "roll_seat_rear": hip.ring(8.5,6.5,5,(105,19,347),(0,1,0)),
        # Place the yaw-carried roll drive above the pitch fork. The nearer
        # low/outboard case entered that fork at roll -45 deg.
        "roll_ST3215_case": hip.box((105-ST_OUTPUT_OFFSET,14.2,429),
                                     (ST_LONG,ST_OUTPUT_DEPTH,ST_SHORT)),
        "roll_motor_output_25T_envelope": hip.cyl(3,3.4,
              (105,14.2+ST_OUTPUT_DEPTH/2,429),(0,1,0)),
        "roll_motor_standoff": rod((105,20,352),(105,20,419),3),
    }
    pitch = {
        # The 9 mm belt claim is centred at Y=37; the remote shaft reaches
        # beyond its outboard flange plane at Y=41.5.
        "roll_spindle": hip.cyl(3,76,(105,-33,347),(0,1,0)),
        "roll_to_pitch_front": rod((105,-18,345),(119,-18,345),3),
        "roll_to_pitch_rear": rod((105,18,345),(119,18,345),3),
        "pitch_front_fork_to_seat": rod((119,-18,345),(115,2,365),3),
        "pitch_rear_fork_to_seat": rod((119,18,345),(138,2,365),3),
        "pitch_bearing_in_6x13x3p5": hip.ring(6.5,3,3.5,(111,2,347),(1,0,0)),
        "pitch_bearing_out_6x13x3p5": hip.ring(6.5,3,3.5,(137,2,347),(1,0,0)),
        "pitch_seat_in": hip.ring(8.5,6.5,5,(110,2,347),(1,0,0)),
        "pitch_seat_out": hip.ring(8.5,6.5,5,(136,2,347),(1,0,0)),
        "pitch_seat_top_bridge": rod((110,2,365),(141,2,365),3),
        "pitch_seat_in_web": rod((112,2,353),(112,2,365),3),
        "pitch_seat_out_web": rod((138,2,353),(138,2,365),3),
        "pitch_ST3215_case": hip.box((151,63-ST_OUTPUT_OFFSET,306),
                                      (ST_OUTPUT_DEPTH,ST_LONG,ST_SHORT)),
        "pitch_motor_output_25T_envelope": hip.cyl(3,3.4,
              (151+ST_OUTPUT_DEPTH/2,63,306),(1,0,0)),
        "pitch_motor_standoff": rod((119,18,345),(151,45,312),3),
    }
    arm = {
        # Remote 9 mm pulley at X=174; former shaft ended at X=158.
        "pitch_spindle": hip.cyl(3,70,(110,2,347),(1,0,0)),
        "arm_root_pad": hip.box((143,2,338),(6,31,12)),
        "pitch_spindle_to_arm_pad": rod((143,2,347),(143,2,339),3),
        "upper_front_rail": rod((143,-11,338),(143,-11,275),3),
        "upper_rear_rail": rod((143,15,338),(143,15,275),3),
        "elbow_front_seat": hip.ring(8.5,6.5,5,(142,-11,273),(1,0,0)),
        "elbow_rear_seat": hip.ring(8.5,6.5,5,(142,15,273),(1,0,0)),
        "elbow_bearing_front_6x13x3p5": hip.ring(6.5,3,3.5,(143,-11,273),(1,0,0)),
        "elbow_bearing_rear_6x13x3p5": hip.ring(6.5,3,3.5,(143,15,273),(1,0,0)),
        "elbow_ST3215_case": hip.box((169,-36+ST_OUTPUT_OFFSET,309),
                                      (ST_OUTPUT_DEPTH,ST_LONG,ST_SHORT)),
        "elbow_motor_output_25T_envelope": hip.cyl(3,3.4,
              (169+ST_OUTPUT_DEPTH/2,-36,309),(1,0,0)),
        "elbow_motor_standoff": rod((143,-11,309),(158,-20,309),3),
    }
    forearm = {
        "elbow_spindle": hip.cyl(3,59,(140,0,273),(1,0,0)),
        "elbow_front_web": rod((154,0,273),(154,-12,267),3),
        "elbow_rear_web": rod((154,0,273),(154,15,267),3),
        "forearm_front_rail": rod((154,-12,267),(145,-9,144),3),
        "forearm_rear_rail": rod((154,15,267),(145,13,144),3),
        "forearm_hand_bridge": hip.box((145,2,144),(13,29,6)),
        "gripper_XL330_case": hip.box((146,40,164),(20,34,26)),
        "gripper_motor_standoff": rod((146,13,164),(146,23,164),3),
        "paw_contact_proxy": hip.box(HAND,(31,28,16)),
        "paw_mount": rod((145,2,144),(136,-5,137),3),
    }
    # 1:1 pulley routes follow the catalog shaft normals. Fixed belt runs
    # belong to the motor-carrying stage; the joint pulley turns with the
    # child stage. Centres are anatomy-fixed during that joint's rotation.
    yaw["yaw_joint_pulley_20T_claim"] = drive_route(
        fixed,"yaw",(45,2,362),(82,2,362),(0,0,1))
    pitch["roll_joint_pulley_20T_claim"] = drive_route(
        yaw,"roll",(105,37,429),(105,37,347),(0,1,0))
    arm["pitch_joint_pulley_20T_claim"] = drive_route(
        pitch,"pitch",(174,63,306),(174,2,347),(1,0,0))
    forearm["elbow_joint_pulley_20T_claim"] = drive_route(
        arm,"elbow",(192,-36,309),(192,0,273),(1,0,0))
    return {"torso":torso,"head":head,"fixed":fixed,"yaw":yaw,
            "pitch":pitch,"arm":arm,"forearm":forearm}


def scene_for(parts, leg, shared, angles, file):
    scene = trimesh.Scene()
    thigh, shin, ankle = lower_package()
    feet = lower_foot(0)
    def put(name, shape, color):
        scene.add_geometry(mesh(shape,color), geom_name=name,node_name=name)
    for name, shape in shared.items(): put(name,shape,(108,117,116,255))
    for side, sign in (("right",1),("left",-1)):
        for stage, group in hip.frame(leg,sign,0,0,0).items():
            for name, shape in group.items():
                put(f"{side}_hip_{stage}_{name}",shape,(90,130,153,255) if "case" in name else (156,143,114,255))
        for group_name,group in (("thigh",thigh),("shin",shin),("ankle",ankle),("foot",feet)):
            for name,raw in group.items():
                shape=raw if sign==1 else raw.mirror(bd.Plane.YZ)
                put(f"{side}_{group_name}_{name}",shape,(80,127,157,255) if "case" in name else (157,149,116,255))
        for stage in ("fixed","yaw","pitch","arm","forearm"):
            for name, raw in parts[stage].items():
                shape = pose(raw,stage,angles)
                if sign == -1: shape = shape.mirror(bd.Plane.YZ)
                put(f"{side}_{stage}_{name}",shape,(76,124,158,255) if "case" in name else (188,153,97,255))
    for name,shape in parts["torso"].items(): put(name,shape,(121,132,128,255) if "case" not in name else (75,125,158,255))
    for name,shape in parts["head"].items(): put(name,shape,(185,184,158,255))
    scene.export(file)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    leg, shared = pelvis_candidate()
    parts = build()
    # Check motor/battery/sensor envelopes against each other at neutral.
    solid_names = ("case","battery_4s850","imu_reserve","camera_module")
    envelope = [(f"{stage}_{name}",shape) for stage,group in parts.items()
                for name,shape in group.items() if any(s in name for s in solid_names)]
    envelope += [(f"left_{name}",shape.mirror(bd.Plane.YZ)) for name,shape in envelope if "camera" not in name and "battery" not in name and "imu" not in name and "head_yaw" not in name]
    envelope_hits = [[an,bn,round(intersection(a,b),3)] for i,(an,a) in enumerate(envelope)
                     for bn,b in envelope[i+1:] if intersection(a,b)>.05]
    # The upper fixed frame may intentionally mate with the old front spine,
    # but its moving motor cases may not enter the fixed torso or opposite case.
    poses = {"stand":(0,0,0,0),"forward":(25,0,-45,-50),
             "out":(0,-45,0,-40),"reach":(25,-25,-65,-65),
             "crawl":(0,-15,-75,-80)}
    pose_hits={}
    stage_case_hits={}
    fixed_obstacles = {**shared, **parts["torso"], **parts["fixed"]}
    for pose_name,angles in poses.items():
        hits=[]
        for stage in ("yaw","pitch","arm","forearm"):
            for name,raw in parts[stage].items():
                if not ("case" in name or "rail" in name or "spindle" in name or "pad" in name or "standoff" in name): continue
                moving=pose(raw,stage,angles)
                for other_name,other in fixed_obstacles.items():
                    if other_name.startswith(("yaw_bearing","yaw_seat","yaw_motor_shaft")) and stage=="yaw": continue
                    vol=intersection(moving,other)
                    if vol>.05: hits.append([f"{stage}_{name}",other_name,round(vol,3)])
        pose_hits[pose_name]=hits
        cross=[]
        for early,later in (("yaw","pitch"),("yaw","arm"),("yaw","forearm"),
                            ("pitch","arm"),("pitch","forearm"),("arm","forearm")):
            for case_name,case in parts[early].items():
                if "case" not in case_name: continue
                moving_case=pose(case,early,angles)
                for other_name,other in parts[later].items():
                    amount=intersection(moving_case,pose(other,later,angles))
                    if amount>.05: cross.append([f"{early}_{case_name}",f"{later}_{other_name}",round(amount,3)])
        stage_case_hits[pose_name]=cross
        scene_for(parts,leg,shared,angles,OUT/f"upper-{pose_name}.glb")
    joins=[]
    for a,b in (("chest_left_rail","chest_crossbar"),("chest_right_rail","chest_crossbar"),
                ("neck_spine","neck_bridge"),("neck_spine","neck_lower_seat_bridge"),
                ("neck_lower_seat_bridge","neck_lower_seat"),
                ("yaw_seat_to_chest_lower","yaw_seat_lower"),
                ("yaw_seat_to_chest_upper","yaw_seat_upper"),
                ("yaw_seat_to_chest_lower","yaw_lower_chest_anchor"),
                ("yaw_seat_to_chest_upper","yaw_upper_chest_anchor"),
                ("yaw_lower_chest_anchor","chest_crossbar"),
                ("yaw_upper_chest_anchor","chest_crossbar")):
        group_a=parts["torso"] if a in parts["torso"] else parts["fixed"]
        group_b=parts["torso"] if b in parts["torso"] else parts["fixed"]
        joins.append([a,b,round(intersection(group_a[a],group_b[b]),3)])
    joins.append(["chest_left_rail","old_front_spine",round(intersection(parts["torso"]["chest_left_rail"],shared["front_spine"]),3)])
    for a,b in (("roll_to_pitch_front","pitch_front_fork_to_seat"),
                ("roll_to_pitch_rear","pitch_rear_fork_to_seat"),
                ("pitch_front_fork_to_seat","pitch_seat_top_bridge"),
                ("pitch_rear_fork_to_seat","pitch_seat_top_bridge")):
        joins.append([a,b,round(intersection(parts["pitch"][a],parts["pitch"][b]),3)])
    motor_mount_pairs=(("torso","head_motor_standoff","torso","head_yaw_XL330_case"),
                       ("torso","yaw_motor_standoff","fixed","yaw_ST3215_case"),
                       ("yaw","roll_motor_standoff","yaw","roll_ST3215_case"),
                       ("pitch","pitch_motor_standoff","pitch","pitch_ST3215_case"),
                       ("arm","elbow_motor_standoff","arm","elbow_ST3215_case"),
                       ("forearm","gripper_motor_standoff","forearm","gripper_XL330_case"))
    motor_mount_gaps={f"{stage}_{name}":round(parts[stage][name].distance(parts[case_stage][case]),3)
                      for stage,name,case_stage,case in motor_mount_pairs}
    report={"stage":"exposed torso and arm candidate, sampled screen only",
            "anatomy":"approved Cove 500 mm ape; shell absent",
            "right_axes_mm":{"shoulder_yaw":YAW,"shoulder_roll":ROLL,
                             "shoulder_pitch":PITCH,"elbow_pitch":ELBOW,
                             "hand_reference":HAND,"neck_yaw":NECK},
            "motor_cases":"Feetech ST3215-C018, 55 g, rated 0.981 N m at 12 V; XL330-M288-T, 18 g, 0.52 N m stall at 5 V",
            "support":"6x13x3.5 mm W618/6 envelopes; seat and shaft details candidate",
            "intended_join_overlaps_mm3":joins,
            "motor_standoff_to_case_gap_mm":motor_mount_gaps,
            "neutral_component_hits":envelope_hits,
            "moving_vs_fixed_hits":pose_hits,
            "motor_case_vs_later_moving_stage_hits":stage_case_hits,
            "frame_volume_gross_mm3":{stage:round(sum(shape.volume for name,shape in group.items()
                 if not any(s in name for s in ("case","battery_4s850","imu_reserve","camera_module","bearing"))),1)
                 for stage,group in parts.items()},
            "upper_drive_routes":{"status":"20T/20T 3MGT pitch-path and 9 mm belt-width claims, unqualified",
                 "pitch_radius_mm":round(BELT_PITCH_RADIUS,3),"belt_width_mm":BELT_PLANE_THICKNESS,
                 "centre_distance_mm":{"yaw":37,"roll":82,"pitch":round(math.hypot(61,41),3),
                                       "elbow":round(math.hypot(36,36),3)},
                 "nearest_catalog_pitch_length_mm":{"yaw":135,"roll":225,"pitch":210,"elbow":165},
                 "source":"Gates Industrial Timing Belt Design Manual, 3M & 3MGT centre-distance table, 20/20 row"},
            "open":["9 mm belt stock lengths, tension, cyclic tooth loading, rated joint speed and pulley retention","motor clamps, shaft adapters/fasteners/loaded support",
                    "stage-to-stage, bilateral and leg cross-link sweeps","contact force and paw kinematics",
                    "per-link stock mass/inertia, harness and supply rails"]}
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    def to_sim(point): return [-point[1]/1000,point[0]/1000,point[2]/1000]
    def case_size(shape):
        bounds=shape.bounding_box()
        return [float(getattr(bounds.max,axis)-getattr(bounds.min,axis))/1000 for axis in ("Y","X","Z")]
    def case_center(shape):
        bounds=shape.bounding_box()
        return to_sim([(getattr(bounds.max,axis)+getattr(bounds.min,axis))/2 for axis in "XYZ"])
    motor_stage={"shoulder_yaw":("fixed","yaw_ST3215_case"),
                 "shoulder_roll":("yaw","roll_ST3215_case"),
                 "shoulder_pitch":("pitch","pitch_ST3215_case"),
                 "elbow_pitch":("arm","elbow_ST3215_case"),
                 "gripper":("forearm","gripper_XL330_case")}
    upper={"status":"sampled supported upper-body candidate; dynamics/drive details not qualified",
           "source_report":"artifacts/ape-frame50-upper-frame/report.json",
           "left_joint_m":{k:to_sim(v) for k,v in
               (("shoulder_yaw",YAW),("shoulder_roll",ROLL),("shoulder_pitch",PITCH),
                ("elbow_pitch",ELBOW),("gripper",HAND))},
           "neck_yaw_m":to_sim(NECK),
           "left_motor_center_m":{k:case_center(parts[stage][name]) for k,(stage,name) in motor_stage.items()},
           "motor_size_m":{k:case_size(parts[stage][name]) for k,(stage,name) in motor_stage.items()},
           "head_motor_center_m":case_center(parts["torso"]["head_yaw_XL330_case"]),
           "head_motor_size_m":case_size(parts["torso"]["head_yaw_XL330_case"]),
           "frame_stock_gross_volume_mm3":report["frame_volume_gross_mm3"],
           "candidate_joint_range_rad":{"shoulder_yaw":[-1.57,1.57],
               "shoulder_roll":[-1.31,1.31],"shoulder_pitch":[-1.57,1.57],
               "elbow_pitch":[-2.09,0.15]}}
    (HERE/"upper_kinematics_supported.json").write_text(json.dumps(upper,indent=2)+"\n")
    # Export the same bilateral neutral assembly shown in the GLB viewer.
    # Earlier this STEP accidentally contained only the right hip and arm.
    thigh,shin,ankle=lower_package()
    foot_stock=lower_foot(0)
    all_shapes=[*shared.values(),*parts["torso"].values(),*parts["head"].values()]
    for sign in (1,-1):
        all_shapes.extend(s for group in hip.frame(leg,sign,0,0,0).values()
                          for s in group.values())
        for group in (thigh,shin,ankle,foot_stock,
                      *(parts[stage] for stage in ("fixed","yaw","pitch","arm","forearm"))):
            all_shapes.extend(s if sign==1 else s.mirror(bd.Plane.YZ)
                              for s in group.values())
    bd.export_step(bd.Compound(children=all_shapes),str(OUT/"upper-neutral.step"))
    shutil.copy2(HERE/"upper_frame_viewer.html",OUT/"index.html")
    print(json.dumps({"joins":joins,"neutral_component_hits":envelope_hits,
                      "pose_hit_counts":{k:len(v) for k,v in pose_hits.items()},
                      "motor_case_cross_stage_hit_counts":{k:len(v) for k,v in stage_case_hits.items()}},indent=2))


if __name__ == "__main__": main()
