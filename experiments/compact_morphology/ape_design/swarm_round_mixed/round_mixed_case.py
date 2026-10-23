"""Independent round/mixed 21-axis architecture envelope and motion screen.

Run from repository root:
  .venv/bin/python experiments/compact_morphology/ape_design/swarm_round_mixed/round_mixed_case.py

This is deliberately a concept stage CAD gate. Sourced actuator bodies, a
source-sized battery, support seats, frame and sketch shell are solids. The
pulley belts are clearance corridors, bearings are printed-seat hypotheses,
and tooth geometry/output splines are not resolved. Positive B-rep cuts are
failures; zero overlap does not promote this to a buildable joint.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[4]
SHELL_DIR = ROOT / "experiments/sketch_ape_shell"
sys.path.insert(0, str(SHELL_DIR))
import build as shell_source

OUT = ROOT / "artifacts/round-mixed-architecture-350"
WALLS = (2.0, 1.0)
FRAME_DENSITY_G_MM3 = .00124  # assumed printed nylon/PLA-like density
SHELL_DENSITY_G_MM3 = .00120  # cosmetic print material assumption

ACTUATORS = {
    "RSF-5B": {
        "kind": "round", "diameter_mm": 25.0, "length_mm": 35.8,
        "mass_g": 77.0, "torque_Nm": .44, "torque_basis": "allowable continuous",
        "voltage_V": 24, "driver": "external servo amplifier; model and price unresolved",
        "one_off_price_USD": None,
        "source": "https://www.harmonicdrive.net/_hd/content/documents1/RSA-8_RSF-5-IDT_Brochure.pdf",
    },
    "XL330-M288-T": {
        "kind": "servo", "size_mm": [20.0, 34.0, 26.0], "mass_g": 18.0,
        "torque_Nm": .10, "torque_basis": "vendor estimated continuous (20% of 0.52 Nm stall)",
        "voltage_V": 5, "driver": "integrated position/current controller; TTL bus",
        "one_off_price_USD": 27.49,
        "stall_Nm": .52,
        "source": "https://robotis.us/products/dynamixel-xl330-m288-t",
    },
}
BATTERY = {"name": "Tattu 4S850 candidate", "size_mm": [30, 60, 30.5],
           "center_mm": [0, 5, 221], "mass_g": 107,
           "source": "https://www.genstattu.com/tattu-850mah-14-8v-75c-4s1p-lipo-battery-pack-with-xt60-plug.html"}

COLORS = {
    "shell": [224, 215, 194, 85], "round": [47, 112, 151, 255],
    "servo": [67, 132, 147, 255], "shaft": [178, 188, 188, 255],
    "bearing": [217, 159, 79, 255], "frame": [110, 123, 132, 255],
    "belt": [111, 77, 62, 255], "battery": [106, 169, 119, 255],
    "service": [220, 186, 86, 160], "axis": [196, 77, 66, 255],
}


def rot_shape(shape, point, axis, degrees):
    if abs(degrees) < 1e-9:
        return shape
    return shape.rotate(bd.Axis(tuple(point), tuple(axis)), degrees)


def rot_point(point, center, axis, degrees):
    v = np.asarray(point, float) - np.asarray(center, float)
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    t = math.radians(degrees)
    out = v*math.cos(t) + np.cross(a, v)*math.sin(t) + a*np.dot(a, v)*(1-math.cos(t))
    return tuple(np.asarray(center, float) + out)


def axis_transform(shape, point, axis, degrees, points=()):
    return rot_shape(shape, point, axis, degrees), [rot_point(p, point, axis, degrees) for p in points]


def cylinder(center, diameter, length, axis="z"):
    obj = bd.Cylinder(diameter/2, length)
    if axis == "x": obj = bd.Rot(0, 90, 0) * obj
    elif axis == "y": obj = bd.Rot(90, 0, 0) * obj
    return bd.Pos(*(float(v) for v in center)) * obj


def hollow_ring(center, od, id_, width, axis):
    return cylinder(center, od, width, axis) - cylinder(center, id_, width+.2, axis)


def box(center, size):
    return bd.Pos(*(float(v) for v in center)) * bd.Box(*size)


def box_between(a, b, width, depth, axis):
    """Rectangular belt span, accurate in length/section but not tooth geometry."""
    d = np.asarray(b, float) - np.asarray(a, float)
    length = float(np.linalg.norm(d))
    direction = d / max(length, 1e-9)
    zdir = np.asarray(axis_vector(axis), float)
    if abs(float(np.dot(direction,zdir))) > .98:
        zdir=np.asarray((0,1,0),float)
    # The belt section is represented by a conservative rectangular corridor.
    pl = bd.Plane(origin=tuple((np.asarray(a)+np.asarray(b))/2),
                  x_dir=tuple(direction), z_dir=tuple(zdir))
    return pl * bd.Box(length, depth, width)


def axis_vector(axis):
    return {"x":(1,0,0), "y":(0,1,0), "z":(0,0,1)}[axis]


def source_shell(wall):
    study = shell_source.Study(350, wall=wall, gap=.65).build()
    return {p["name"]: p["shape"] for p in study.parts}


def add_axis(rows, side, joint, axis, output, motor_center, mount_link,
             round_unit=False, ratio="1:1 GT2 belt corridor; tooth/pulley design unresolved"):
    name = f"{side}_{joint}" if side else joint
    rows.append({"joint":name, "axis":axis, "output_mm":tuple(output),
                 "motor_center_mm":tuple(motor_center), "motor": "RSF-5B" if round_unit else "XL330-M288-T",
                 "mount_link":mount_link, "ratio":ratio, "round":round_unit})


def architecture_axes():
    A=[]
    for side, s in (("L", -1), ("R", 1)):
        # Round modules use their diameter where the proximal yaw output can
        # tolerate an offset motor and supported remote shaft.
        add_axis(A,side,"shoulder_yaw","z",(s*65,3,244),(s*24,2,244),"torso",True)
        add_axis(A,side,"shoulder_roll","y",(s*65,3,238),(s*39,17,234),"torso")
        add_axis(A,side,"shoulder_pitch","x",(s*65,3,230),(s*39,-17,226),"torso")
        add_axis(A,side,"elbow_pitch","x",(s*96,2,194.5),(s*89,2,162),"upper_arm")
        add_axis(A,side,"paw_open","x",(s*100,0,107),(s*89,0,124),"forearm")
        add_axis(A,side,"hip_yaw","z",(s*38,3,164),(s*16,3,164),"pelvis",True)
        add_axis(A,side,"hip_roll","y",(s*38,3,157),(s*21,19,155),"pelvis")
        add_axis(A,side,"hip_pitch","x",(s*38,3,149),(s*21,-17,147),"pelvis")
        add_axis(A,side,"knee_pitch","x",(s*42,3,86),(s*42,3,120),"thigh")
        add_axis(A,side,"ankle_pitch","x",(s*46,3,24),(s*46,3,51),"shin")
    add_axis(A,None,"head_yaw","z",(0,4,278),(0,4,278),"neck")
    assert len(A)==21
    return A


def motor_shape(axis):
    if axis["round"]:
        return cylinder(axis["motor_center_mm"],25,35.8,"z")
    # Source dimensions 20x34x26 mm, permuted to put the output axis along the
    # 26 mm depth dimension. Exact horn datum still needs the vendor CAD.
    dims={"x":(26,20,34), "y":(20,26,34), "z":(20,34,26)}[axis["axis"]]
    return box(axis["motor_center_mm"],dims)


def make_mechanism(axes):
    objects=[]
    axis_shapes={}
    for row in axes:
        v=row["axis"]; p=row["output_mm"]; q=row["motor_center_mm"]
        mshape=motor_shape(row)
        objects.append({"name":row["joint"]+"_motor", "shape":mshape,
                        "kind":"round" if row["round"] else "servo", "joint":row["joint"],
                        "mass_g":ACTUATORS[row["motor"]]["mass_g"], "center":q,
                        "interface":"RSF native output / XL330 spline-horn datum assumed; verify vendor geometry"})
        # Motor pulley and output pulley are sized as drawing-space reserves.
        if q != p:
            for tag,c in (("motor",q),("output",p)):
                objects.append({"name":row["joint"]+f"_{tag}_pulley", "shape":cylinder(c,14,6,v),
                                "kind":"belt", "joint":row["joint"], "mass_g":None,"center":c,
                                "assumption":"14 mm pulley envelope; 1:1 timing belt pitch/teeth unresolved"})
            belt=box_between(q,p,6,1.6,v)
            objects.append({"name":row["joint"]+"_belt_corridor", "shape":belt,
                            "kind":"belt", "joint":row["joint"], "mass_g":None,
                            "center":tuple((np.array(q)+np.array(p))/2),
                            "assumption":"6 mm GT2-class belt width source; 1.6 mm corridor depth and route are design assumptions"})
        shaft_d=8 if row["round"] else 6
        shaft=cylinder(p,shaft_d,38,v)
        objects.append({"name":row["joint"]+"_supported_output_shaft", "shape":shaft,
                        "kind":"shaft", "joint":row["joint"], "mass_g":None, "center":p,
                        "assumption":"steel shaft diameter and 38 mm length are structural hypotheses"})
        rings=[]
        for offset in (-14,14):
            c=tuple(np.asarray(p)+np.asarray(axis_vector(v))*offset)
            ring=hollow_ring(c,16 if not row["round"] else 18,6.2 if not row["round"] else 8.2,4,v)
            rings.append(ring)
            objects.append({"name":row["joint"]+f"_bearing_seat_{offset:+d}","shape":ring,
                            "kind":"bearing","joint":row["joint"],"mass_g":None,"center":c,
                            "assumption":"printed structural bearing seat; rolling bearing source/rating absent"})
        # A fork web closes the child-side bearing support into a link yoke.
        web=box(p,(22,5,20))
        objects.append({"name":row["joint"]+"_child_yoke_web","shape":web,"kind":"frame",
                        "joint":row["joint"],"mass_g":None,"center":p,
                        "assumption":"local printed yoke web; final axis-specific web orientation unresolved"})
        axis_shapes[row["joint"]]=cylinder(p,1.5,52,v)
    # Shared inner twin rails, shoulder beam, pelvis beam and service reserves.
    frame=[
        ("left_inner_spine_rail",(-18,4,205),(4,6,118)),
        ("right_inner_spine_rail",(18,4,205),(4,6,118)),
        ("shoulder_crossbeam",(0,4,244),(58,6,7)),
        ("pelvis_crossbeam",(0,4,164),(58,6,7)),
        ("upper_sternum_web",(0,8,229),(7,8,28)),
        ("pelvic_mid_web",(0,8,164),(7,8,32)),
    ]
    for name,c,size in frame:
        sh=box(c,size)
        objects.append({"name":name,"shape":sh,"kind":"frame","joint":None,"mass_g":None,
                        "center":c,"assumption":"inner frame volume hypothesis, polymer/composite material not selected"})
    bshape=box(BATTERY["center_mm"],BATTERY["size_mm"])
    objects.append({"name":BATTERY["name"],"shape":bshape,"kind":"battery","joint":None,
                    "mass_g":BATTERY["mass_g"],"center":tuple(BATTERY["center_mm"]),"source":BATTERY["source"]})
    # Power conversion + CAN/TTL distribution are service-space placeholders.
    for name,c,size in (("servo_5V_converter_reserve",(0,22,216),(40,10,14)),
                        ("RSF_driver_reserve",(0,22,241),(40,10,16)),
                        ("bus_connector_service_reserve",(0,23,228),(18,12,12))):
        objects.append({"name":name,"shape":box(c,size),"kind":"service","joint":None,
                        "mass_g":None,"center":c,"assumption":"volume only; converter/driver/connector model not chosen"})
    return objects,axis_shapes


_BBOX_CACHE={}

def aabb_gap(a,b):
    # Build123d's optimal bound is itself a costly B-rep query. Cache once per
    # immutable shape so the all-pairs broad phase does not repeat it thousands
    # of times.
    if id(a) not in _BBOX_CACHE: _BBOX_CACHE[id(a)]=a.bounding_box()
    if id(b) not in _BBOX_CACHE: _BBOX_CACHE[id(b)]=b.bounding_box()
    A=_BBOX_CACHE[id(a)]; B=_BBOX_CACHE[id(b)]
    ds=[]
    for lo1,hi1,lo2,hi2 in ((A.min.X,A.max.X,B.min.X,B.max.X),
                            (A.min.Y,A.max.Y,B.min.Y,B.max.Y),
                            (A.min.Z,A.max.Z,B.min.Z,B.max.Z)):
        ds.append(max(lo2-hi1,lo1-hi2,0))
    return math.sqrt(sum(x*x for x in ds))


def vol(a,b):
    common=a.intersect(b)
    return round(sum(x.volume for x in common),3) if common else 0.0


def fixed_report(objects,skin,wall):
    shell_groups={"torso":[],"pelvis":[],"arms":[],"legs":[],"head":[]}
    for name,sh in skin.items():
        if name.startswith("torso_"): shell_groups["torso"].append((name,sh))
        elif name.startswith("pelvis_"): shell_groups["pelvis"].append((name,sh))
        elif "arm" in name or "forearm" in name or "palm" in name or "finger" in name or "thumb" in name or "elbow" in name or "wrist" in name or "shoulder" in name: shell_groups["arms"].append((name,sh))
        elif "thigh" in name or "shin" in name or "knee" in name or "ankle" in name or "foot" in name or "toe" in name: shell_groups["legs"].append((name,sh))
        elif "head" in name or "muzzle" in name or "eye" in name or "nostril" in name or "mouth" in name or "neck" in name: shell_groups["head"].append((name,sh))
    compounds={region:bd.Compound(children=[p for _,p in parts])
               for region,parts in shell_groups.items() if parts}
    skin_hits={}
    for obj in objects:
        hit={}
        joint=obj["joint"] or ""
        regions=("arms","torso") if "shoulder" in joint else (
                 ("pelvis","legs") if "hip" in joint else
                 ("arms",) if any(k in joint for k in ("elbow","paw")) else
                 ("legs",) if any(k in joint for k in ("knee","ankle")) else
                 ("head",) if "head" in joint else ("torso","pelvis") if obj["kind"] in ("battery","service") else ("torso",))
        if obj["name"].startswith(("pelvis_crossbeam","pelvic_mid")): regions=("pelvis",)
        for region in regions:
            amount=vol(obj["shape"],compounds[region]) if aabb_gap(obj["shape"],compounds[region])<.1 else 0.0
            if amount>.001: hit[region]=round(amount,3)
        if hit: skin_hits[obj["name"]]=hit
    internal=[]
    for i,a in enumerate(objects):
        for b in objects[i+1:]:
            # Pulleys, belt corridors, shaft seats and yokes are designed mates;
            # report their interpenetration only through the explicit list of
            # *non-mating* collisions below.
            if a["joint"] and a["joint"]==b["joint"] and {a["kind"],b["kind"]} <= {"belt","shaft","bearing","frame","round","servo"}:
                continue
            if aabb_gap(a["shape"],b["shape"])>.01: continue
            amount=vol(a["shape"],b["shape"])
            if amount>.05:
                internal.append({"a":a["name"],"b":b["name"],"intersection_mm3":amount})
    return {"wall_mm":wall,"exact_source_skin_intersections_mm3":skin_hits,
            "nonmating_component_intersections_mm3":internal,
            "meaning":"exact B-rep volume of intersections after AABB broad phase against the corresponding anatomical source group; intended same-joint shaft/bearing/belt mates are excluded by explicit rule"}


def groups_from_source(source):
    def compound(names): return bd.Compound(children=[source[n] for n in names])
    out={"torso":compound([n for n in source if n.startswith("torso_")]),
         "pelvis":compound([n for n in source if n.startswith("pelvis_")]),
         "head":compound([n for n in source if n.startswith(("head_cranium_","face_mask","muzzle_shell","mouth_","L_eye_","R_eye_","L_nostril","R_nostril","neck_collar"))])}
    for s in ("L","R"):
        out[s+"_upper_arm"]=compound([n for n in source if n.startswith(s+"_upper_arm") or n==s+"_shoulder_trim"])
        out[s+"_forearm"]=compound([n for n in source if n.startswith(s+"_forearm") or n==s+"_elbow_cuff" or n==s+"_wrist_cuff"])
        out[s+"_paw"]=compound([n for n in source if n.startswith((s+"_palm",s+"_finger",s+"_thumb"))])
        out[s+"_thigh"]=compound([n for n in source if n.startswith(s+"_thigh") or n==s+"_knee_cuff"])
        out[s+"_shin"]=compound([n for n in source if n.startswith(s+"_shin") or n==s+"_ankle_cuff"])
        out[s+"_boot"]=compound([n for n in source if n.startswith((s+"_foot",s+"_toe"))])
    return out


def rotate_chain(shape, side, joint_angles, limb, return_points=False):
    s=-1 if side=="L" else 1
    pts=[]
    if limb in ("thigh","shin","boot"):
        hip=(s*36,3,164)
        for key,axis in (("hip_yaw","z"),("hip_roll","y"),("hip_pitch","x")):
            angle=joint_angles.get(side+"_"+key,0)
            shape,pts=axis_transform(shape,hip,axis_vector(axis),angle,pts)
        knee=rot_point((s*42,3,86),hip,(0,0,1),joint_angles.get(side+"_hip_yaw",0))
        for key,axis in (("hip_roll","y"),("hip_pitch","x")):
            knee=rot_point(knee,hip,axis_vector(axis),joint_angles.get(side+"_"+key,0))
        if limb in ("shin","boot"):
            shape=rot_shape(shape,knee,(1,0,0),joint_angles.get(side+"_knee_pitch",0))
        ankle=rot_point((s*46,3,24),hip,(0,0,1),joint_angles.get(side+"_hip_yaw",0))
        for key,axis in (("hip_roll","y"),("hip_pitch","x")):
            ankle=rot_point(ankle,hip,axis_vector(axis),joint_angles.get(side+"_"+key,0))
        ankle=rot_point(ankle,knee,(1,0,0),joint_angles.get(side+"_knee_pitch",0))
        if limb=="boot": shape=rot_shape(shape,ankle,(1,0,0),joint_angles.get(side+"_ankle_pitch",0))
    elif limb in ("upper_arm","forearm","paw"):
        sh=(s*65,3,244)
        for key,axis in (("shoulder_yaw","z"),("shoulder_roll","y"),("shoulder_pitch","x")):
            shape=rot_shape(shape,sh,axis_vector(axis),joint_angles.get(side+"_"+key,0))
        elbow=rot_point((s*96,2,194.5),sh,(0,0,1),joint_angles.get(side+"_shoulder_yaw",0))
        elbow=rot_point(elbow,sh,(0,1,0),joint_angles.get(side+"_shoulder_roll",0))
        elbow=rot_point(elbow,sh,(1,0,0),joint_angles.get(side+"_shoulder_pitch",0))
        if limb in ("forearm","paw"):
            shape=rot_shape(shape,elbow,(1,0,0),joint_angles.get(side+"_elbow_pitch",0))
    return shape


def moving_groups(groups,pose):
    out={"torso":groups["torso"],"pelvis":groups["pelvis"],
         "head":rot_shape(groups["head"],(0,4,278),(0,0,1),pose.get("head_yaw",0))}
    for side in ("L","R"):
        for limb in ("upper_arm","forearm","paw","thigh","shin","boot"):
            out[side+"_"+limb]=rotate_chain(groups[side+"_"+limb],side,pose,limb)
    return out


def pose_list():
    def zeros(): return {a:0 for a in [f"{s}_{j}" for s in ("L","R") for j in ("shoulder_yaw","shoulder_roll","shoulder_pitch","elbow_pitch","paw_open","hip_yaw","hip_roll","hip_pitch","knee_pitch","ankle_pitch")]+["head_yaw"]}
    p=[("neutral",zeros())]
    deep=zeros(); deep.update({"L_hip_pitch":-54,"L_knee_pitch":109,"L_ankle_pitch":-20,
                               "L_shoulder_pitch":-35,"L_elbow_pitch":60,"L_paw_open":25})
    p.append(("coupled_floor_reach_stress",deep))
    crawl=zeros(); crawl.update({"L_hip_pitch":25,"R_hip_pitch":25,"L_hip_roll":-15,"R_hip_roll":15,
                                 "L_knee_pitch":70,"R_knee_pitch":70,"L_ankle_pitch":-25,"R_ankle_pitch":-25,
                                 "L_shoulder_pitch":45,"R_shoulder_pitch":45,"L_shoulder_roll":-25,"R_shoulder_roll":25,
                                 "L_elbow_pitch":80,"R_elbow_pitch":80,"head_yaw":15})
    p.append(("four_point_crawl_stress",crawl))
    walk=zeros(); walk.update({"L_hip_pitch":-20,"R_hip_pitch":20,"L_knee_pitch":30,"R_knee_pitch":5,
                               "L_ankle_pitch":-10,"R_ankle_pitch":10,"L_shoulder_pitch":18,"R_shoulder_pitch":-18})
    p.append(("walking_step_stress",walk))
    elbow=zeros(); elbow.update({"L_elbow_pitch":115,"R_elbow_pitch":60})
    p.append(("elbow_0_60_115_stress",elbow))
    knee=zeros(); knee.update({"L_knee_pitch":115,"R_knee_pitch":60})
    p.append(("knee_0_60_115_stress",knee))
    ankle=zeros(); ankle.update({"L_ankle_pitch":35,"R_ankle_pitch":-35})
    p.append(("ankle_plus_minus_35_stress",ankle))
    arms=zeros(); arms.update({"L_shoulder_yaw":-25,"R_shoulder_yaw":25,"L_shoulder_roll":45,"R_shoulder_roll":45,
                               "L_shoulder_pitch":-30,"R_shoulder_pitch":-30,"L_elbow_pitch":45,"R_elbow_pitch":45})
    p.append(("arms_out_forward_stress",arms))
    return p


def body_collision_report(groups,pose):
    moved=moving_groups(groups,pose)
    names=list(moved)
    rows=[]
    # Screen plausible self-collision and adjacent-link pairs, then use exact
    # B-rep intersections only for candidates whose bounds show contact.
    def candidate(a,b):
        if a in ("torso","pelvis") or b in ("torso","pelvis"):
            return True
        if a=="head" or b=="head": return True
        A=a.split("_"); B=b.split("_")
        if A[0] != B[0]: return True
        return abs(names.index(a)-names.index(b))<=3
    for i,a_name in enumerate(names):
        a=moved[a_name]
        for b_name in names[i+1:]:
            if not candidate(a_name,b_name): continue
            # Adjacent segments share hinges; compare them and all other groups.
            b=moved[b_name]
            gap=aabb_gap(a,b)
            if gap>5: continue
            hit=0.0
            if gap<3:
                rows.append({"a":a_name,"b":b_name,"intersection_mm3_exact_not_run":None,
                             "minimum_AABB_gap_mm":round(gap,3),
                             "minimum_brep_gap_mm":None,
                             "mating_pair": abs(hit)<.001 and gap<.05 and
                                (a_name.split("_")[-1] in ("forearm","shin") or b_name.split("_")[-1] in ("forearm","shin"))})
    return {"pose_angles_deg":pose,"moving_shell_intersections_and_near_gaps":rows,
            "moving_motor_and_belt_sweep":"not_run; current hardware positions represent neutral mounting only",
            "gap_method":"pose motion uses conservative AABB near-contact screen only; exact static hardware/source-shell overlaps are reported separately under wall_screens."}


def mass_report(source,objects,axes):
    comps=[]
    for name,shape in source.items():
        mass=shape.volume*SHELL_DENSITY_G_MM3
        c=shape.center()
        comps.append((f"cosmetic_shell:{name}",mass,(c.X,c.Y,c.Z),"shell-density assumption"))
    for obj in objects:
        if obj["mass_g"] is not None:
            comps.append((obj["name"],obj["mass_g"],obj["center"],"maker sourced" if obj["kind"] in ("round","servo","battery") else "source envelope"))
        elif obj["kind"] in ("frame","bearing","shaft","belt"):
            comps.append((obj["name"],obj["shape"].volume*FRAME_DENSITY_G_MM3,tuple(obj["center"]),"assumed printed structure density; excludes unmodeled fasteners"))
    total=sum(x[1] for x in comps)
    com=[sum(x[1]*x[2][i] for x in comps)/total for i in range(3)]
    inertia=[0,0,0]
    for _,m,c,_ in comps:
        x,y,z=[c[i]-com[i] for i in range(3)]
        inertia[0]+=m*(y*y+z*z); inertia[1]+=m*(x*x+z*z); inertia[2]+=m*(x*x+y*y)
    sourced_motors=sum(ACTUATORS[a["motor"]]["mass_g"] for a in axes)
    servo_cost=sum(ACTUATORS[a["motor"]]["one_off_price_USD"] or 0 for a in axes)
    return {"estimated_total_assembly_mass_g_including_source_shell_and_assumed_frame":round(total,1),
            "source_motor_mass_g":round(sourced_motors,1),"battery_mass_g":BATTERY["mass_g"],
            "shell_mass_estimate_g":round(sum(x[1] for x in comps if x[0].startswith("cosmetic_shell:")),1),
            "assumed_frame_support_mass_g":round(sum(x[1] for x in comps if x[3].startswith("assumed printed")),1),
            "component_COM_mm":[round(x,2) for x in com],
            "point_mass_diagonal_inertia_g_mm2":[round(x,1) for x in inertia],
            "inertia_limit":"point masses only; excludes each part intrinsic inertia",
            "minimum_observed_servo_only_USD":round(servo_cost,2),
            "round_actuator_price":"unquoted; RSF-5B x4, servo-only total is not a full BOM",
            "component_assumptions":[{"name":n,"mass_g":round(m,2),"basis":basis} for n,m,c,basis in comps]}


def mesh_add(scene,name,shape,color):
    for i,solid in enumerate(shape.solids()):
        if solid.volume<.001: continue
        v,f=solid.tessellate(.25,.2)
        m=trimesh.Trimesh(np.array([[p.X,p.Y,p.Z] for p in v]),np.asarray(f),process=False)
        m.visual.vertex_colors=COLORS[color]
        scene.add_geometry(m,geom_name=f"{name}_{i}",node_name=f"{name}_{i}")


def render(pose_name,source,groups,pose,objects,axis_shapes):
    scene=trimesh.Scene()
    moved=moving_groups(groups,pose)
    # Keep the torso front/back and pelvis shell exact in all poses.
    for name,shape in source.items():
        if name.startswith(("torso_","pelvis_")): mesh_add(scene,name,shape,"shell")
    for name,shape in moved.items():
        if name in ("torso","pelvis"): continue
        mesh_add(scene,name,shape,"shell")
    for obj in objects:
        mesh_add(scene,obj["name"],obj["shape"],obj["kind"] if obj["kind"] in COLORS else "frame")
    for name,shape in axis_shapes.items(): mesh_add(scene,name+"_axis",shape,"axis")
    scene.export(OUT/f"{pose_name}.glb")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source2=source_shell(2)
    source1=source_shell(1)
    groups=groups_from_source(source2)
    axes=architecture_axes()
    objects,axis_shapes=make_mechanism(axes)
    report={"stage":"concept; not an assembly candidate",
            "body":"actual 350 mm Study shell B-reps, X lateral, Y rear, Z up",
            "axes":[],"pose_screens":{},"walls":{},
            "battery":{"dimensions_mm":BATTERY["size_mm"],"mass_g":BATTERY["mass_g"],"center_mm":BATTERY["center_mm"],"source":BATTERY["source"]},
            "actuator_data":ACTUATORS,
            "unresolved":["actual actuator output-axis drawings/horns and spline geometry", "round RSF model driver/amp, power conversion, heat sink and price", "GT2 belt tooth/pulley wrap, tensioner and derailment margin", "printed-seat/shaft radial and axial load ratings", "motor-wire connectors, harness routing and access", "XL330 duty at every loaded axis; its seller's 0.10 Nm is only a 20%-of-stall estimate", "full motor/belt moving-pose sweep and dynamic torque-speed requirements"]}
    for a in axes:
        report["axes"].append({k:(list(v) if isinstance(v,tuple) else v) for k,v in a.items()})
    for wall,skin in ((2,source2),(1,source1)):
        report["walls"][str(wall)]=fixed_report(objects,skin,wall)
    report["mass_and_cost"]=mass_report(source2,objects,axes)
    # Gravity-only torque lower bound from shell link masses; it omits internal
    # motors, acceleration, contact and any transmission efficiency.
    report["torque_screen"]={"XL330 vendor estimate":.10,"XL330 stall":.52,"RSF-5B allowable continuous":.44,
       "hip roll illustration_Nm":round(1.2*9.81*.020,3),
       "hip roll illustration_basis":"1.2 kg whole-body hypothesis and 20 mm support offset from existing 35 cm screen; not a weighed assembly or gait load",
       "evaluation":"screen only; the shared shells and servos are not loaded dynamic CAD"}
    poses=pose_list()
    for name,pose in poses:
        report["pose_screens"][name]=body_collision_report(groups,pose)
    (OUT/"round_mixed_report.json").write_text(json.dumps(report,indent=2)+"\n")
    # Two views make the displacement inspectable without tessellating eight
    # copies of every cosmetic surface on each edit/run.
    for name,pose in (poses[0],poses[1]):
        render("round_mixed_"+name,source2,groups,pose,objects,axis_shapes)
    viewer=Path(__file__).with_name("round_mixed_viewer.html")
    if viewer.exists(): (OUT/"index.html").write_text(viewer.read_text())
    print(json.dumps({"stage":report["stage"],"axes":len(axes),
      "neutral_nonmating_collisions_wall2":len(report["walls"]["2"]["nonmating_component_intersections_mm3"]),
      "skin_hit_items_wall2":len(report["walls"]["2"]["exact_source_skin_intersections_mm3"]),
      "mass_g":report["mass_and_cost"]["estimated_total_assembly_mass_g_including_source_shell_and_assumed_frame"],
      "poses":len(report["pose_screens"])},indent=2))


if __name__=="__main__": main()
