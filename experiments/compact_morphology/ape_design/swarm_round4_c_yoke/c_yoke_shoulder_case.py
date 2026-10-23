"""Exact source-shell falsifier for an offset serial C-yoke shoulder."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import build123d as b
from build123d import SkipClean
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'experiments/sketch_ape_shell'))
from build import Study, loft
OUT=ROOT/'artifacts/swarm-round4-c-yoke-350'
XL_STEP=ROOT/'artifacts/round2-hip-350/vendor-cad/XL_XC330_official.stp'
PACK_C=(0,0,200); PACK_S=(60,30,30.5)
SERVICE=[('buck',(42,0,200),(16,18,8)),('can',(42,0,210),(16,18,8)),('connectors',(42,0,219),(16,14,6))]
YAW=(66.,4.,244.)
ROLL=(77.,4.,252.)
PITCH=(88.,4.,242.)

def vol(s): return round(sum(x.volume for x in s.solids()),3) if s else 0.
def box(c,sz): return b.Pos(*c)*b.Box(*sz)
def axial(axis,r,h,p):
    rot={'x':b.Rot(0,90,0),'y':b.Rot(90,0,0),'z':b.Rot(0,0,0)}[axis]
    return b.Pos(*p)*rot*b.Cylinder(r,h)
def ring(axis,p,od=13.,bore=5.5,w=3.5):
    return axial(axis,od/2,w,p)-axial(axis,bore/2,w+.2,p)
def step_at(shape,axis,p,twist=0):
    # Official XL330 STEP output datum is local (0,0,6.5); orient around it.
    base={'x':b.Rot(0,90,0),'y':b.Rot(-90,0,0),'z':b.Rot(0,0,0)}[axis]
    spin={'x':b.Rot(twist,0,0),'y':b.Rot(0,twist,0),'z':b.Rot(0,0,twist)}[axis]
    return b.Pos(*p)*spin*base*b.Pos(0,0,-6.5)*shape
def case_envelope_at(axis,p,twist=0):
    # Conservative case box uses the raw STEP local coordinates before
    # step_at normalizes output Z=6.5 to the requested axis point.
    envelope=b.Pos(0,-7.5,-8.0)*b.Box(20,34,29)
    return step_at(envelope,axis,p,twist)
def quick_inter(a,c):
    x=a.bounding_box();y=c.bounding_box()
    if any(max((x.min.X,x.min.Y,x.min.Z)[i],(y.min.X,y.min.Y,y.min.Z)[i]) >
           min((x.max.X,x.max.Y,x.max.Z)[i],(y.max.X,y.max.Y,y.max.Z)[i]) for i in range(3)): return 0.
    with SkipClean():
        return vol(a.intersect(c))
def outside(a,outer_parts):
    remaining=a
    with SkipClean():
        for envelope in outer_parts:
            if remaining is None or not remaining.solids(): break
            remaining=remaining.cut(envelope)
    return vol(remaining)
def belt(axis,p1,p2,pitch_d=7.,thick=1.6,width=3.5):
    # Capsule-like annular keepout around two pulleys, in the plane normal to axis.
    v=np.array(p2,float)-np.array(p1,float); n=np.array({'x':(1,0,0),'y':(0,1,0),'z':(0,0,1)}[axis],float)
    v-=np.dot(v,n)*n; d=float(np.linalg.norm(v)); u=v/d
    mid=tuple((np.array(p1,float)+np.array(p2,float))/2)
    pl=b.Plane(origin=mid,x_dir=tuple(u),z_dir=tuple(n))
    outer=b.extrude(b.SlotCenterToCenter(d,pitch_d+2*thick),amount=width/2,both=True)
    inner=b.extrude(b.SlotCenterToCenter(d,pitch_d),amount=width/2+.15,both=True)
    return pl.location*(outer-inner)
def add_mesh(scene,name,shape,color):
    vv,ff=shape.tessellate(.35,.35)
    m=trimesh.Trimesh(np.array([[v.X,v.Y,v.Z] for v in vv]),np.asarray(ff),process=False)
    m.visual.vertex_colors=color;scene.add_geometry(m,geom_name=name,node_name=name)

def source_outer():
    torso=loft([(160,0,2,42,25),(173,0,2,49,29),(199,0,3,58,34),(229,0,4,63,35),(253,0,5,63,30),(264,0,5,60,27),(274,0,5,35,22),(278,0,5,23,18)])
    upper=loft([(199,93,2,14,17),(207,94,2,17,20),(228,89,3,20,22),(249,80,4,19,21),(260,73,4,14,17),(265,70,4,6,9),(267.8,70,4,6,9)])
    # Exact geometric outer shoulder trim used by build.py before hollowing.
    trim=b.Solid.make_loft([b.Wire.make_ellipse(ry,rz,b.Plane(origin=(x,4,245),x_dir=(0,1,0),z_dir=(1,0,0))) for x,ry,rz in [(63,17,19),(65.5,17,19),(68,16,18)]])
    return [torso,upper,trim]

def build(wall,pitch_relocated=False):
    st=Study(height=350,wall=wall,gap=.65);st.build()
    parts={p['name']:p['shape'] for p in st.parts}
    torso=[parts['torso_front'],parts['torso_back']]
    upper=[parts['R_upper_arm_front'],parts['R_upper_arm_back']]
    trim=parts['R_shoulder_trim']
    skins=[('torso_front',torso[0]),('torso_back',torso[1]),('upper_arm_front',upper[0]),('upper_arm_back',upper[1]),('shoulder_trim',trim)]
    outer=source_outer()
    # Keep vendor STEP for provenance/visual inspection only. Its compound has
    # overlapping solids; OCC common volume is not a valid physical overlap sum.
    xl=b.import_step(str(XL_STEP))

    # Three source-sized actuators: torso yaw, fork roll and cradle pitch.
    yaw_point=(-2,3,260) if pitch_relocated else (16,17,245)
    roll_point=(80,16,228) if pitch_relocated else (80,13,241)
    pitch_motor_point=(48,12,247) if pitch_relocated else (96,13,242)
    yaw_motor=case_envelope_at('z',yaw_point,90)
    roll_motor=case_envelope_at('y',roll_point,25)
    pitch_motor=case_envelope_at('x',pitch_motor_point,90 if pitch_relocated else 0)

    # Yaw turntable is carried on two Z bearings; its C-fork is the first moving link.
    yaw_rings=[ring('z',(66,4,237)),ring('z',(66,4,251))]
    yaw_shaft=axial('z',2.5,22,YAW)
    yaw_plate=box((70,4,245),(8,16,4))
    # Front/back cheeks form the open C-fork. They stop short of each other, so
    # they do not bridge across the shoulder's hollow center.
    fork_cheeks=[box((77,-8,251),(18,4,20)),box((77,16,251),(18,4,20))]
    roll_rings=[ring('y',(77,-7,252)),ring('y',(77,15,252))]
    roll_shaft=axial('y',2.5,22,ROLL)

    # A diagonal arm cradle places pitch 14.9 mm from roll; no orthogonal shaft
    # reaches the same point. X trunnions are supported from opposite cheeks.
    link=b.Solid.make_cylinder(4.0,math.dist(ROLL,PITCH),b.Plane(origin=ROLL,z_dir=tuple((np.array(PITCH)-np.array(ROLL))/math.dist(ROLL,PITCH))))
    pitch_rings=[ring('x',(82,4,242)),ring('x',(94,4,242))]
    pitch_shaft=axial('x',2.5,18,PITCH)
    pitch_webs=[box((82,4,248),(4,14,5)),box((94,4,248),(4,14,5))]

    # Short, equal pulleys and closed belt envelopes; limits are packaging proxies.
    yaw_pulleys=[ring('z',(16,17,245),od=9,bore=5.5,w=3.5),ring('z',(66,4,245),od=9,bore=5.5,w=3.5)]
    yaw_belt=belt('z',(16,17,245),(66,4,245),pitch_d=9,thick=1.2,width=3.5)
    roll_pulleys=[ring('y',roll_point,od=8,bore=5.5,w=3.5),ring('y',ROLL,od=8,bore=5.5,w=3.5)]
    roll_belt=belt('y',roll_point,ROLL,pitch_d=8,thick=1.2,width=3.5)
    if pitch_relocated:
        pitch_remote_shaft=axial('x',2.2,40.,(68,12,247))
        pitch_remote_supports=[ring('x',(59,12,247),od=10,bore=4.8),ring('x',(75,12,247),od=10,bore=4.8)]
        pitch_motor_pulley=ring('x',pitch_motor_point,od=7,bore=5.5,w=3.5)
        pitch_out_point=(88,12,247)
    else:
        pitch_remote_shaft=None; pitch_remote_supports=[]
        pitch_motor_pulley=ring('x',pitch_motor_point,od=7,bore=5.5,w=3.5)
        pitch_out_point=pitch_motor_point
    pitch_pulleys=[pitch_motor_pulley,ring('x',pitch_out_point,od=7,bore=5.5,w=3.5)]
    pitch_belt=belt('x',pitch_out_point,PITCH,pitch_d=7,thick=1.2,width=3.5)

    pack=box(PACK_C,PACK_S);services=[(n,box(c,s)) for n,c,s in SERVICE]
    objs=[('yaw_XL330',yaw_motor),('roll_XL330',roll_motor),('pitch_XL330',pitch_motor),
      ('yaw_bearing_lower',yaw_rings[0]),('yaw_bearing_upper',yaw_rings[1]),('yaw_output_shaft',yaw_shaft),('yaw_rotor_plate',yaw_plate),
      ('fork_front_cheek',fork_cheeks[0]),('fork_rear_cheek',fork_cheeks[1]),('roll_bearing_front',roll_rings[0]),('roll_bearing_rear',roll_rings[1]),('roll_output_shaft',roll_shaft),
      ('roll_pitch_link',link),('pitch_bearing_inboard',pitch_rings[0]),('pitch_bearing_outboard',pitch_rings[1]),('pitch_output_shaft',pitch_shaft),('pitch_web_inboard',pitch_webs[0]),('pitch_web_outboard',pitch_webs[1]),
      *([('pitch_remote_x_shaft',pitch_remote_shaft),('pitch_remote_support_inboard',pitch_remote_supports[0]),('pitch_remote_support_outboard',pitch_remote_supports[1])] if pitch_relocated else []),
      ('yaw_motor_pulley',yaw_pulleys[0]),('yaw_output_pulley',yaw_pulleys[1]),('yaw_belt_reserve',yaw_belt),
      ('roll_motor_pulley',roll_pulleys[0]),('roll_output_pulley',roll_pulleys[1]),('roll_belt_reserve',roll_belt),
      ('pitch_motor_pulley',pitch_pulleys[0]),('pitch_output_pulley',pitch_pulleys[1]),('pitch_belt_reserve',pitch_belt),('4S_pack',pack),*services]
    shell_hits=[];contain=[]
    key_geometry={'yaw_XL330','roll_XL330','pitch_XL330','yaw_output_shaft','roll_output_shaft','pitch_output_shaft',
      'fork_front_cheek','fork_rear_cheek','roll_pitch_link','pitch_web_inboard','pitch_web_outboard',
      'yaw_bearing_lower','yaw_bearing_upper','roll_bearing_front','roll_bearing_rear','pitch_bearing_inboard','pitch_bearing_outboard'}
    containment_geometry={'yaw_XL330','roll_XL330','pitch_XL330','fork_front_cheek','fork_rear_cheek','roll_pitch_link'}
    if pitch_relocated: key_geometry |= {'pitch_remote_x_shaft','pitch_remote_support_inboard','pitch_remote_support_outboard'}; containment_geometry |= {'pitch_remote_x_shaft'}
    for n,o in objs:
        if n in ('4S_pack','buck','can','connectors'):
            if n=='4S_pack' or n in ('buck','can','connectors'):
                continue
        elif n not in key_geometry:
            continue
        for sn,ss in skins:
            v=quick_inter(o,ss)
            if v>.01:shell_hits.append({'object':n,'skin':sn,'mm3':v})
        if n in containment_geometry:
            v=outside(o,outer)
            if v>.01:contain.append({'object':n,'outside_source_outer_mm3':v})
    # Report all component contacts, then clearly separate designed seat contacts.
    intended={frozenset(x) for x in [
      ('yaw_bearing_lower','yaw_rotor_plate'),('yaw_bearing_upper','yaw_rotor_plate'),
      ('yaw_output_shaft','yaw_bearing_lower'),('yaw_output_shaft','yaw_bearing_upper'),('yaw_output_shaft','yaw_rotor_plate'),
      ('fork_front_cheek','yaw_rotor_plate'),('fork_rear_cheek','yaw_rotor_plate'),
      ('fork_front_cheek','roll_bearing_front'),('fork_rear_cheek','roll_bearing_rear'),
      ('roll_output_shaft','roll_bearing_front'),('roll_output_shaft','roll_bearing_rear'),('roll_output_shaft','roll_pitch_link'),
      ('roll_pitch_link','pitch_bearing_inboard'),('roll_pitch_link','pitch_bearing_outboard'),
      ('pitch_output_shaft','pitch_bearing_inboard'),('pitch_output_shaft','pitch_bearing_outboard'),
      ('pitch_web_inboard','pitch_bearing_inboard'),('pitch_web_outboard','pitch_bearing_outboard'),
      ('yaw_motor_pulley','yaw_belt_reserve'),('yaw_output_pulley','yaw_belt_reserve'),
      ('roll_motor_pulley','roll_belt_reserve'),('roll_output_pulley','roll_belt_reserve'),
      ('pitch_motor_pulley','pitch_belt_reserve'),('pitch_output_pulley','pitch_belt_reserve'),
      ('yaw_motor_pulley','yaw_XL330'),('roll_motor_pulley','roll_XL330'),('pitch_motor_pulley','pitch_XL330') ]}
    pair_hits=[]
    om=dict(objs)
    candidate_pairs=[('yaw_output_shaft','roll_output_shaft'),('yaw_output_shaft','pitch_output_shaft'),('roll_output_shaft','pitch_output_shaft'),
      ('yaw_XL330','roll_XL330'),('yaw_XL330','pitch_XL330'),('roll_XL330','pitch_XL330')]
    candidate_pairs += [(m,n) for m in ('yaw_XL330','roll_XL330','pitch_XL330') for n in ('4S_pack','buck','can','connectors')]
    candidate_pairs += [(m,n) for m in ('yaw_XL330','roll_XL330','pitch_XL330') for n in ('yaw_rotor_plate','fork_front_cheek','fork_rear_cheek','roll_pitch_link','pitch_web_inboard','pitch_web_outboard')]
    if pitch_relocated:candidate_pairs += [('pitch_XL330','pitch_remote_x_shaft'),('pitch_remote_x_shaft','yaw_output_shaft'),('pitch_remote_x_shaft','roll_output_shaft'),('pitch_remote_x_shaft','pitch_output_shaft')]
    for na,nb in candidate_pairs:
        if frozenset((na,nb)) in intended:continue
        v=quick_inter(om[na],om[nb])
        if v>.01:pair_hits.append({'a':na,'b':nb,'mm3':v})
    pack_hit=sum(quick_inter(pack,t) for t in torso)
    serv_hit={n:sum(quick_inter(o,t) for t in torso) for n,o in services}
    ext={}
    for n,s in skins:
        bb=s.bounding_box();ext[n]=[bb.min.X,bb.max.X,bb.min.Y,bb.max.Y,bb.min.Z,bb.max.Z]
    # Relevant outside shell visibility metric: maximum projected excess over
    # the intact shoulder/upper-arm/torso outer BRep envelope is zero when contained.
    report={'wall_mm':wall,'layout':'torso_side_bays_alt' if pitch_relocated else 'primary','source':'Exact Study(height=350,wall=wall,gap=.65) plus reconstructed outer lofts from sketch_ape_shell/build.py','skin_bboxes_mm':ext,
      'axes_mm':{'yaw_Z':YAW,'roll_Y':ROLL,'pitch_X':PITCH,'roll_pitch_axis_offset_mm':round(math.dist(ROLL,PITCH),3)},
      'motors':{'yaw':{'model':'ROBOTIS XL330-M288-T official STEP','case_bbox_mm':[20,34,29],'mass_g':18,'rated_working_torque':'unpublished; 0.52 Nm stall, seller-estimated continuous 0.10 Nm only'},
       'roll':{'model':'ROBOTIS XL330-M288-T official STEP','case_bbox_mm':[20,34,29],'mass_g':18,'rated_working_torque':'unpublished; 0.52 Nm stall, seller-estimated continuous 0.10 Nm only'},
       'pitch':{'model':'ROBOTIS XL330-M288-T official STEP','case_bbox_mm':[20,34,29],'mass_g':18,'rated_working_torque':'unpublished; 0.52 Nm stall, seller-estimated continuous 0.10 Nm only'}},
      'transmission':{'yaw':'1:1 closed-belt reserve; '+str(round(math.dist(yaw_point[:2],YAW[:2]),3))+'mm center distance','roll':'1:1 closed-belt reserve; '+str(round(math.dist((roll_point[0],roll_point[2]),(ROLL[0],ROLL[2])),3))+'mm center distance','pitch':'motor bay → '+str(40 if pitch_relocated else round(math.dist(pitch_motor_point,PITCH),3))+'mm supported X shaft → 1:1 short offset belt; plane center distance '+str(round(math.dist(pitch_out_point[1:],PITCH[1:]),3))+'mm'},
      'pitch_motor_output_point_mm':pitch_motor_point,
      'pack':{'center_mm':PACK_C,'size_mm':PACK_S,'torso_intersection_mm3':pack_hit},
      'service':{n:{'center_mm':c,'size_mm':s,'torso_intersection_mm3':serv_hit[n]} for n,c,s in SERVICE},
      'exact_skin_intersections':shell_hits,'outside_source_outer_intersections':contain,
      'nonmating_component_intersections':pair_hits,
      'verdict':'pass_static_candidate' if not shell_hits and not contain and not pair_hits and pack_hit==0 and all(v==0 for v in serv_hit.values()) else 'reject_at_static_package_gate',
      'step_audit':{'raw_compound_volume_mm3':round(xl.volume,3),'raw_solid_count':len(xl.solids()),'raw_intersection_volumes_rejected':'A raw compound common summed to more than the whole package; sequential solid fuse was non-monotonic/invalid. Therefore all exact case-wall and package collision checks use conservative 20x34x29 mm solid case envelope oriented at STEP output datum. STEP remains visual/provenance only.'},
      'limits':['XL330 working torque is not published; 0.10 Nm is a seller estimate, not a measured rating.','Conservative case envelope is exact as a box solid, but overstates local STEP geometry; package dimensions include output boss.','Belt reserves and pitch pulley geometry are keep-outs; belt teeth and production routing are not verified.','Ring bearings, webs and link are load-path proxies; material, stiffness, tolerance, backlash and lifetime are unqualified.','No source cover is added; source outer-loft containment is checked separately from wall intersection.']}
    objects=[(n,o) for n,o in objs]
    return report,skins,objects,pack,services

def export_view(wall,skins,objects,layout='primary'):
    scene=trimesh.Scene()
    for n,s in skins:add_mesh(scene,'source_'+n,s,(226,219,203,180))
    for n,s in objects:
        if n.endswith('XL330'):c=(55,108,158,255)
        elif 'pack' in n:c=(211,147,58,190)
        elif n in ('buck','can','connectors'):c=(174,109,82,180)
        elif 'belt' in n:c=(194,139,56,150)
        else:c=(92,122,82,230)
        add_mesh(scene,n,s,c)
    scene.export(OUT/f'c-yoke-{layout}.glb')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--alt-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    report_path=OUT/'report.json'
    reports=json.loads(report_path.read_text()) if args.alt_only and report_path.exists() else []
    if not args.alt_only:
      for wall in (1,2):
        r,skins,objs,pack,services=build(wall);reports.append(r)
        print('wall',wall,r['verdict'],'skin',r['exact_skin_intersections'],'contain',r['outside_source_outer_intersections'],'pairs',r['nonmating_component_intersections'],flush=True)
        if wall==2:export_view(wall,skins,objs)
    r,skins,objs,pack,services=build(2,pitch_relocated=True);reports.append(r)
    print('wall 2 torso_side_bays_alt',r['verdict'],'skin',r['exact_skin_intersections'],'contain',r['outside_source_outer_intersections'],'pairs',r['nonmating_component_intersections'],flush=True)
    export_view(2,skins,objs,'torso_side_bays_alt')
    report_path.write_text(json.dumps(reports,indent=2)+'\n')
    (OUT/'index.html').write_text(Path(__file__).with_name('c_yoke_shoulder_viewer.html').read_text())
if __name__=='__main__':main()
