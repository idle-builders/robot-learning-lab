"""Bounded source-BRep screen for a covered, locally driven right shoulder.

Three XL330 motor envelopes occupy one shallow deltoid cartridge.  Each drives
an independent supported output through a conceptual 1:1 belt/sector pair.
This is packaging evidence, not a tooth, bearing-life, or duty qualification.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import build123d as bd
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"experiments/sketch_ape_shell"))
from build import Study
OUT=ROOT/"artifacts/swarm-round2-shoulder-350"
PIVOT=(66.0,0.0,246.0)
PACK=(0.0,0.0,200.0,60.0,30.0,30.5)
XL=(20.0,34.0,26.0)
# Case centres/orientations are absolute source-sized placements. The output
# shafts are explicit axes; case-to-shaft transfer is an envelope placeholder.
CONFIGS={
 "first": [
   {"name":"yaw_XL330","center":(51,-5,246),"size":(34,26,20),"axis":"z","axis_point":(66,0,246)},
   {"name":"roll_XL330","center":(72,-20,250),"size":(20,34,26),"axis":"y","axis_point":(66,0,246)},
   {"name":"pitch_XL330","center":(77,17,233),"size":(26,20,34),"axis":"x","axis_point":(66,0,246)},
 ],
 "revised": [
   # The first roll case conflicted with yaw. Move it outboard, forward, and up;
   # retain all other datums so the effect remains attributable.
   {"name":"yaw_XL330","center":(53,-12,246),"size":(34,26,20),"axis":"z","axis_point":(66,0,246)},
   {"name":"roll_XL330","center":(76,15,259),"size":(20,34,26),"axis":"y","axis_point":(66,0,246)},
   {"name":"pitch_XL330","center":(80,16,226),"size":(26,20,34),"axis":"x","axis_point":(66,0,246)},
 ]
}
CONFIGS["axis_offset_10mm"]=[{**c,"axis_point":(76.0,0.0,246.0)} for c in CONFIGS["revised"]]

def vol(s): return round(sum(x.volume for x in s.solids()),3) if s else 0.0
def rot(shape,axis,deg,p=None):
    if p is None:p=PIVOT
    r={"x":bd.Rot(deg,0,0),"y":bd.Rot(0,deg,0),"z":bd.Rot(0,0,deg)}[axis]
    return bd.Pos(*p)*r*bd.Pos(*(-v for v in p))*shape
def shoulder_point_global(p,yaw,roll,pitch):
    q=np.array(p,dtype=float)-np.array(PIVOT,dtype=float)
    for ax,deg in (("z",yaw),("y",roll),("x",pitch)):
        t=math.radians(deg); c,s=math.cos(t),math.sin(t)
        R={"x":np.array([[1,0,0],[0,c,-s],[0,s,c]]),
           "y":np.array([[c,0,s],[0,1,0],[-s,0,c]]),
           "z":np.array([[c,-s,0],[s,c,0],[0,0,1]])}[ax]
        q=R@q
    return tuple(q+np.array(PIVOT,dtype=float))
def mesh(scene,name,shape,color):
    v,f=shape.tessellate(.35,.35)
    m=trimesh.Trimesh(np.array([[p.X,p.Y,p.Z] for p in v]),np.asarray(f),process=False)
    m.visual.vertex_colors=color; scene.add_geometry(m,geom_name=name,node_name=name)
def motor(item): return bd.Pos(*item["center"])*bd.Box(*item["size"])
def cylinder_axis(axis,rad,depth,p):
    if axis=="x": return bd.Pos(*p)*bd.Rot(0,90,0)*bd.Cylinder(rad,depth)
    if axis=="y": return bd.Pos(*p)*bd.Rot(90,0,0)*bd.Cylinder(rad,depth)
    return bd.Pos(*p)*bd.Cylinder(rad,depth)
_PART_CACHE={}
def build(wall,config):
    global PIVOT
    PIVOT=tuple(config[0]["axis_point"])
    if wall not in _PART_CACHE:
        study=Study(height=350,wall=wall,gap=.65); study.build()
        _PART_CACHE[wall]={p["name"]:p["shape"] for p in study.parts}
    parts=_PART_CACHE[wall]
    torso=parts["torso_front"].fuse(parts["torso_back"])
    upper_source=parts["R_upper_arm_front"].fuse(parts["R_upper_arm_back"])
    # The sketch's shoulder seam overlaps the torso skin by a small amount.
    # Trim that exact hard-volume penetration from the moving parent link.
    upper=upper_source-torso
    fore=parts["R_forearm_front"].fuse(parts["R_forearm_back"])
    paw=parts["R_palm_front"].fuse(parts["R_palm_back"])
    cases=[motor(c) for c in config]
    elbow_case=bd.Pos(90,1,232.55)*bd.Box(20,26,34)
    paw_case=bd.Pos(102,1,150)*bd.Box(20,34,26)
    # Add a smooth, removable shallow-deltoid cap. Its interior is a true
    # cavity large enough for the three cases and bearing carriers, while its
    # overlap with original shell defines a closed 0.65 mm seam land.
    # Outer limits are intentionally explicit and reported against source skin.
    pod_outer=bd.Sphere(1).scale((39,43,38)).translate((72,0,246))
    pod_inner=bd.Sphere(1).scale((36,40,35)).translate((72,0,246))
    pod=pod_outer-pod_inner
    # Output seat rings with a separated pair of radial supports per axis.
    supports=[]; outputs=[]
    for i,c in enumerate(config):
        ax=c["axis"]; p=c["axis_point"]
        # Bearing envelopes are two 8x4x4 mm rings, 20 mm apart. Their seat
        # webs return to a common cartridge, not to cosmetic skin alone.
        pair=[]
        for d in (-10,10):
            q=list(p); q["xyz".index(ax)]+=d
            ring=cylinder_axis(ax,5,4,q)-cylinder_axis(ax,2,5,q)
            pair.append(ring)
        supports.extend(pair)
        outputs.append(cylinder_axis(ax,2,24,p))
    # Elbow flex and single-axis paw opening each return load through opposed
    # 8 mm bearing seats; outputs are not supported by the servo horn alone.
    for ax,p,span in (("y",(96,2,194.5),17),("y",(108,0,106),10)):
        for d in (-span,span):
            q=list(p); q["xyz".index(ax)]+=d
            supports.append(cylinder_axis(ax,5,4,q)-cylinder_axis(ax,2,5,q))
        outputs.append(cylinder_axis(ax,2,2*span+8,p))
    # Fused yoke volumes approximate independent nested rings and arm flange.
    frame=bd.Pos(PIVOT[0]+6,0,246)*bd.Box(23,19,19)
    for c in config:
        frame=frame.fuse(cylinder_axis(c["axis"],13,5,c["axis_point"]))
    for s in supports: frame=frame.fuse(s)
    battery=bd.Pos(PACK[0],PACK[1],PACK[2])*bd.Box(PACK[3],PACK[4],PACK[5])
    # Reservation is a forbidden envelope plus a service connector margin.
    pack_service=bd.Pos(0,25,200)*bd.Box(60,20,34)
    route_points=[(66,9,246),(78,8,231),(90,4,215),(96,2,194.5)]
    route_edge=bd.Edge.make_spline(route_points)
    cable=bd.Solid.sweep(bd.Wire.make_circle(3,bd.Plane(origin=route_points[0],z_dir=(1,0,-1))),route_edge)
    fixed=[("torso",torso), ("upper_arm",upper), ("forearm",fore), ("palm",paw), ("deltoid_cover",pod)]
    clashes=[]
    for c,shape in zip(config,cases):
        for name,skin in fixed[:2]:
            x=vol(shape.intersect(skin))
            if x>1: clashes.append({"pair":f"{c['name']} / source {name}","mm3":x})
    for i in range(len(cases)):
        for j in range(i+1,len(cases)):
            x=vol(cases[i].intersect(cases[j]))
            if x>1: clashes.append({"pair":f"{config[i]['name']} / {config[j]['name']}","mm3":x})
    cavity_leak=[{"motor":c["name"],"outside_inner_cover_mm3":vol(shape-pod_inner)} for c,shape in zip(config,cases)]
    distal_motor_clashes={"elbow_case_vs_upper_arm_mm3":vol(elbow_case.intersect(upper)),
      "elbow_case_vs_torso_mm3":vol(elbow_case.intersect(torso)),
      "paw_case_vs_forearm_mm3":vol(paw_case.intersect(fore)),"paw_case_vs_palm_mm3":vol(paw_case.intersect(paw))}
    pack_clash=[vol(battery.intersect(torso)),vol(pack_service.intersect(torso))]
    # The arm chain follows source pivots, then samples coupled arm-down,
    # forward-reach, abducted/out, floor-reach, and crawl positions.
    elbow=(96,2,194.5)
    def shoulder_pose(shape,yaw,roll,pitch):
        return rot(rot(rot(shape,"z",yaw),"y",roll),"x",pitch)
    def arm_pose(yaw,roll,pitch,elbow_deg):
        moved_upper=shoulder_pose(upper,yaw,roll,pitch)
        moved_fore0=shoulder_pose(fore,yaw,roll,pitch)
        moved_paw0=shoulder_pose(paw,yaw,roll,pitch)
        # Elbow pivot moves with the shoulder carrier before elbow flex.
        q=shoulder_point_global(elbow,yaw,roll,pitch)
        moved_fore=rot(moved_fore0,"y",-elbow_deg,q)
        moved_paw=rot(moved_paw0,"y",-elbow_deg,q)
        return moved_upper,moved_fore,moved_paw
    poses={
       "arms_down":(0,0,0,0),"forward_reach":(0,0,-42,70),
       "out":(0,38,0,35),"floor_reach":(0,0,-34,105),"crawl":(0,24,-24,82),
    }
    motion=[]
    for name,angles in poses.items():
        u,f,h=arm_pose(*angles)
        motion.append({"pose":name,"angles_deg":{"yaw":angles[0],"roll":angles[1],"pitch":angles[2],"elbow":angles[3]},
          "upper_vs_torso_mm3":vol(u.intersect(torso)),"fore_vs_torso_mm3":vol(f.intersect(torso)),
          "paw_vs_torso_mm3":vol(h.intersect(torso)),"upper_vs_fore_mm3":vol(u.intersect(f)),
          "fore_vs_paw_mm3":vol(f.intersect(h)),"arm_vs_cover_mm3":round(vol(u.intersect(pod))+vol(f.intersect(pod))+vol(h.intersect(pod)),3)})
    # Exact outer clearance at motor case corners and cartridge cover bounds.
    # Added cover is a shell replacement over the local deltoid, not a hidden
    # source skin pass; volume needed to clear the envelope is part of the cost.
    bounds={"source_upper_arm": [list(upper.bounding_box().min),list(upper.bounding_box().max)],
            "cover": [list(pod.bounding_box().min),list(pod.bounding_box().max)]}
    report={"wall_mm":wall,"configuration":config,"motor_specs":{"XL330-M288-T":"20x34x26 mm; 18 g; 0.52 N m stall at 5 V; continuous torque unpublished"},
      "source_shell":"Study(height=350, wall=wall, gap=.65) from experiments/sketch_ape_shell/build.py",
      "source_coordinate_note":"Sketch source uses X lateral, Y rearward, Z up; right arm is +X.",
      "pivot_mm":PIVOT,"battery_reservation_mm":{"center":PACK[:3],"size":PACK[3:],"service_clearance_box_center":(0,25,200),"service_clearance_box_size":(60,20,34)},
      "upper_arm_torso_relief_mm3":round(upper_source.volume-upper.volume,3),
      "clashes":clashes,"case_vs_cover_cavity_leak":cavity_leak,"battery_torso_intersection_mm3":pack_clash[0],"service_torso_intersection_mm3":pack_clash[1],
      "distal_motors":{"elbow":{"source":"XL330-M288-T, 20x26x34 envelope, 18g","center_mm":(90,1,232.55),"case_shell_intersections_mm3":distal_motor_clashes},"paw":{"source":"XL330-M288-T, 20x34x26 envelope, 18g","center_mm":(102,1,150)}},
      "service":{"route_shape":"6 mm sweep from shoulder to elbow","route_vs_motors_mm3":[vol(cable.intersect(c)) for c in cases+[elbow_case,paw_case]],"connector_bend":"not modeled; margin shown with pack reservation"},
      "pose_screen":motion,"cover_bounds":bounds,
      "mass_screen_g":{"5_XL330_cases":90,"assumed_bearing_and_transmission_reserve":30,"source_shell_at_1.24g_cm3":round((upper.volume+fore.volume+paw.volume)*.00124,1),"moving_arm_module_total_estimate":round((upper.volume+fore.volume+paw.volume)*.00124+120,1),"COM_mm":list(np.average(np.array([list(upper.center()),list(fore.center()),list(paw.center()),[53,-12,246],[76,15,259],[80,16,226],[90,1,232.55],[102,1,150]]),axis=0,weights=np.array([upper.volume*.00124,fore.volume*.00124,paw.volume*.00124,18,18,18,18,18])))},
      "source_torque":{"XL330_stall_Nm":0.52,"XL330_continuous":"not published; not qualified","transmission":"1:1 conceptual at output; no torque multiplication claimed"},
      "limits":["Pod is a shell-volume revision; protrusion and seam are measured against source, not a stock-skin pass.","Carrier, bearings, outputs, and belt envelopes are geometric hypotheses; exact bearing part, fits, fasteners and structural ratings remain open.","Cable loop is reserved as a 6 mm round channel along the upper-arm interior to the forearm; connector bend and swept cable remain unverified.","No actuator torque/duty pass; all three axes use XL330 stall label only."]}
    return report,parts,torso,upper,fore,paw,pod,cases+[elbow_case,paw_case],frame,supports,outputs,battery,pack_service,cable,motion,arm_pose

def main():
    global PIVOT
    OUT.mkdir(parents=True,exist_ok=True)
    allreports=[]
    for wall in (1,2):
      for key,config in CONFIGS.items():
        result=build(wall,config)
        report,parts,torso,upper,fore,paw,pod,cases,frame,supports,outputs,battery,pack_service,cable,motion,arm_pose=result
        allreports.append({"layout":key,**report})
        if wall==2:
            PIVOT=tuple(config[0]["axis_point"])
            for pname,angles in (("arms_down",(0,0,0,0)),("forward_reach",(0,0,-42,70)),("out",(0,38,0,35)),("floor_reach",(0,0,-34,105)),("crawl",(0,24,-24,82))):
                u,f,h=arm_pose(*angles); scene=trimesh.Scene()
                mesh(scene,"source_torso_skin",torso,(222,214,196,255)); mesh(scene,"source_upper_arm_skin",upper,(235,229,214,255))
                mesh(scene,"source_forearm_pose",f,(231,222,204,255));mesh(scene,"source_paw_pose",h,(245,236,219,255))
                mesh(scene,"upper_pose",u,(237,228,211,255));mesh(scene,"covered_deltoid_cartridge",pod,(126,139,145,150))
                for c,s in zip(config,cases[:3]): mesh(scene,c["name"],s,(57,104,149,255))
                # The distal cases are carried with their link in each pose.
                elbow_motor_pose=rot(rot(rot(cases[3],"z",angles[0]),"y",angles[1]),"x",angles[2])
                elbow_point=shoulder_point_global((96,2,194.5),*angles[:3])
                paw_motor_pose=rot(rot(rot(rot(cases[4],"z",angles[0]),"y",angles[1]),"x",angles[2]),"y",-angles[3],elbow_point)
                mesh(scene,"elbow_XL330_case",elbow_motor_pose,(57,104,149,255));mesh(scene,"paw_XL330_case",paw_motor_pose,(57,104,149,255))
                mesh(scene,"supported_nested_carrier",frame,(86,97,102,255))
                for i,s in enumerate(supports): mesh(scene,f"bearing_seat_{i}",s,(181,115,54,255))
                for i,s in enumerate(outputs): mesh(scene,f"output_shaft_{i}",s,(68,72,75,255))
                mesh(scene,"hip_agent_battery_reservation",battery,(208,145,63,85));mesh(scene,"battery_connector_service_margin",pack_service,(221,166,80,70))
                mesh(scene,"neutral_6mm_internal_service_route",cable,(179,91,76,255))
                scene.export(OUT/f"shoulder-{key}-{pname}.glb")
        print(f"wall={wall} layout={key} exact clashes={report['clashes']} poses={report['pose_screen']}",flush=True)
    (OUT/"report.json").write_text(json.dumps(allreports,indent=2,default=list)+"\n")
    (OUT/"index.html").write_text(Path(__file__).with_name("round2_shoulder_viewer.html").read_text())

if __name__=="__main__":main()
