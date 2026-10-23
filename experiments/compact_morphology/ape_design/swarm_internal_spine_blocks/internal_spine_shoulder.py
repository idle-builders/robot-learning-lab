"""Central paired-drive, independent-bearing shoulder-yaw coupon at 500 mm.

Only original parametric ape-shell geometry and analytic mechanism envelopes are
created here. No Microban CAD is imported or copied.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import build123d as b
from build123d import SkipClean
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'experiments/sketch_ape_shell'))
from build import Study, loft
OUT=ROOT/'artifacts/internal-spine-shoulder-500'
OUT.mkdir(parents=True,exist_ok=True)
H=500.; PIVOTS={'R':(80.,5.,347.),'L':(-80.,5.,347.)}
PACK_C=(0.,14.,285.7); PACK_SIZE=(60.,30.,30.5)
SERVICE_C=(0.,-20.,365.); SERVICE_SIZE=(25.,20.,14.)
MOTOR_C={'R':(30.,-22.,347.),'L':(-30.,-22.,347.)}
RSF_D=25.; RSF_L=35.8

def box(c,s):return b.Pos(*c)*b.Box(*s)
def cyl(axis,r,h,p):
    rot={'x':b.Rot(0,90,0),'y':b.Rot(90,0,0),'z':b.Rot(0,0,0)}[axis]
    return b.Pos(*p)*rot*b.Cylinder(r,h)
def annulus_z(p,od=13.,id=6.,width=3.5):return cyl('z',od/2,width,p)-cyl('z',id/2,width+.2,p)
def ellipsoid_shell(c,outer,inner):
    a=b.Solid.make_sphere(1).scale(outer).translate(c)
    d=b.Solid.make_sphere(1).scale(inner).translate(c)
    return a-d
def vol(s):return round(sum(x.volume for x in s.solids()),3) if s else 0.
def bbox(s):
    q=s.bounding_box();return [[round(q.min.X,3),round(q.min.Y,3),round(q.min.Z,3)],[round(q.max.X,3),round(q.max.Y,3),round(q.max.Z,3)]]
def overlap(a,c):
    A=a.bounding_box();C=c.bounding_box()
    if any(max((A.min.X,A.min.Y,A.min.Z)[i],(C.min.X,C.min.Y,C.min.Z)[i])>min((A.max.X,A.max.Y,A.max.Z)[i],(C.max.X,C.max.Y,C.max.Z)[i]) for i in range(3)):return 0.
    with SkipClean():q=a.intersect(c)
    return vol(q)
def belt(p1,p2,width=4.,thick=1.5,pitch_d=8.):
    v=np.array(p2,float)-np.array(p1,float);d=float(np.linalg.norm(v));u=v/d
    mid=tuple((np.array(p1,float)+np.array(p2,float))/2)
    pl=b.Plane(origin=mid,x_dir=tuple(u),z_dir=(0,0,1))
    outer=b.extrude(b.SlotCenterToCenter(d,pitch_d+2*thick),amount=width/2,both=True)
    inner=b.extrude(b.SlotCenterToCenter(d,pitch_d),amount=width/2+.15,both=True)
    return pl.location*(outer-inner)
def point_cable(points,r=2.):
    # Union of conservative cylinders and bend spheres along the nominal route.
    bits=[]
    for a,c in zip(points,points[1:]):
        d=np.asarray(c,float)-np.asarray(a,float);L=float(np.linalg.norm(d))
        bits.append(b.Solid.make_cylinder(r,L,b.Plane(origin=tuple(a),z_dir=tuple(d/L))))
    for p in points:bits.append(b.Pos(*p)*b.Sphere(r))
    return b.Compound(children=bits)
def turn(shape,deg,p):return b.Pos(*p)*b.Rot(0,0,deg)*b.Pos(*(-np.asarray(p))) * shape
def add_mesh(scene,name,shape,color):
    v,f=shape.tessellate(.45,.35)
    m=trimesh.Trimesh(vertices=np.asarray([[x.X,x.Y,x.Z] for x in v]),faces=np.asarray(f),process=False)
    m.visual.vertex_colors=color;scene.add_geometry(m,geom_name=name,node_name=name)

def main():
    rows=[]; exports={}
    for wall in (1.,2.):
        st=Study(H,wall,.65);st.build();parts={x['name']:x['shape'] for x in st.parts}
        skin_names=['torso_front','torso_back','R_upper_arm_front','R_upper_arm_back','R_shoulder_trim','L_upper_arm_front','L_upper_arm_back','L_shoulder_trim']
        skin={n:parts[n] for n in skin_names}
        service_skin={**skin,**{n:parts[n] for n in ('neck_collar','head_cranium_front','head_cranium_back')}}
        outer_torso=loft([(z*H/350,x*H/350,y*H/350,rx*H/350,ry*H/350) for z,x,y,rx,ry in [
            (160,0,2,42,25),(173,0,2,49,29),(199,0,3,58,34),(229,0,4,63,35),(253,0,5,63,30),(264,0,5,60,27),(274,0,5,35,22),(278,0,5,23,18)]])
        objs=[];motors={}
        for side,c in MOTOR_C.items():
            # Vertical output datum at the top face; case is below it. Rotating
            # unit around Z changes the PT1 rear cable exit while preserving axis.
            motor=b.Pos(c[0],c[1],c[2]-RSF_L/2)*b.Cylinder(RSF_D/2,RSF_L)
            motors[side]=motor;objs.append((f'{side}_RSF5B50_IDT_case',motor,'motor'))
            # Minimal nominal rear-connector gland at outward side after a 90° yaw.
            ext=1 if side=='R' else -1
            connector=box((c[0]+ext*16.5,c[1],c[2]-20),(8,8,8));objs.append((f'{side}_PT1_connector_reserve',connector,'connector'))
            # Output adapter is an explicit assumed interface, since the
            # brochure's full output spline/shaft drawing is not modeled here.
            adapter=cyl('z',2.5,4.,(c[0],c[1],c[2]+2.));objs.append((f'{side}_assumed_output_adapter',adapter,'shaft'))
            pivot=PIVOTS[side]
            # Two independent W618/6 seats on the internal spine-supported pivot.
            for j,z in enumerate((pivot[2]-13.,pivot[2]+.0)):
                ring=annulus_z((pivot[0],pivot[1],z));objs.append((f'{side}_W618_6_bearing_{j+1}',ring,'bearing'))
            shaft=cyl('z',3.,34.,(pivot[0],pivot[1],pivot[2]+1.));objs.append((f'{side}_supported_output_shaft',shaft,'shaft'))
            # A stepped rotor carrier rises above the top bearing, stays in the
            # torso's front/back split plane, then enters the arm seam below its
            # existing shoulder trim. Its exact shell cuts are reported below.
            ext=1 if side=='R' else -1
            a=np.array((pivot[0],pivot[1],pivot[2]+12.));q=np.array((ext*86.,pivot[1]-.5,pivot[2]+12.))
            d=q-a;bridge=b.Solid.make_cylinder(4.,float(np.linalg.norm(d)),b.Plane(origin=tuple(a),z_dir=tuple(d/np.linalg.norm(d))))
            tongue=box((ext*90.,2.85,pivot[2]+12),(8,0.8,6))
            carrier=b.Compound(children=[bridge,tongue]);objs.append((f'{side}_yaw_rotor_arm_carrier',carrier,'carrier'))
            # Output pulley and exact planar closed-belt keep-out envelope.
            mp=(c[0],c[1],pivot[2]+5.5);op=(pivot[0],pivot[1],pivot[2]+5.5)
            objs.append((f'{side}_motor_pulley',annulus_z(mp,od=10,id=5,width=4),'pulley'))
            objs.append((f'{side}_output_pulley',annulus_z(op,od=10,id=6,width=4),'pulley'))
            objs.append((f'{side}_belt_corridor',belt(mp,op,width=4.,thick=1.3,pitch_d=10.),'belt'))
            # A small straight lead path turns above the pulley plane toward the
            # centerline service reserve; PT1 orientation is pointed laterally.
            way=[(c[0]+ext*16.5,c[1],c[2]-20),(ext*50.,-18.,c[2]-20),(ext*50.,-18.,pivot[2]+15),(0.,-18.,pivot[2]+15),(0.,SERVICE_C[1],pivot[2]+15),(0.,SERVICE_C[1],SERVICE_C[2])]
            objs.append((f'{side}_CAN_power_lead_keepout',point_cable(way,2.),'lead'))
        # Shared continuous internal spine and two transverse bored beams.
        spine=box((0,5,345),(12,12,70))
        # Crossbeams are bored for the independent 13 mm bearing ODs; the 6 mm
        # shafts run through the bearing inner races, not through solid beams.
        for i,z in enumerate((PIVOTS['R'][2]-13.,PIVOTS['R'][2])):
            beam=box((0,5,z),(176,5,6))
            for p in PIVOTS.values():beam=beam-cyl('z',6.65,6.2,(p[0],p[1],z))
            spine=spine.fuse(beam)
        objs.append(('continuous_internal_spine_with_bored_crossbeams',spine,'frame'))
        pack=box(PACK_C,PACK_SIZE);objs.append(('4S850_pack_reservation',pack,'pack'))
        service=box(SERVICE_C,SERVICE_SIZE);objs.append(('CAN_5V_connector_service_reserve',service,'service'))
        # Fixed-component exact shell gate. The upper-arm skin is checked at q=0;
        # motor/frame/support shells remain separate from the intentionally moving arm skins.
        shell_rows=[]
        for n,o,kind in objs:
            relevant={k:v for k,v in skin.items() if k.startswith('torso_')}
            if kind in {'carrier','rotor','cover'}:
                relevant={k:v for k,v in skin.items() if k.startswith('torso_') or 'upper_arm' in k or 'shoulder_trim' in k}
            hits=[]
            for sn,ss in relevant.items():
                q=overlap(o,ss)
                if q>.01:hits.append({'skin':sn,'mm3':round(q,3)})
            outside=vol(o.cut(outer_torso)) if overlap(o,outer_torso)>0 else vol(o)
            shell_rows.append({'component':n,'kind':kind,'bounds_mm':bbox(o),'shell_intersections':hits,'outside_torso_outer_mm3':round(outside,3)})
        # Fixed collisions; intended shaft/bearing and rotor/carrier seat pairs are excluded.
        fixed=[]
        for n,o,k in objs:
            if k in {'belt','lead'}:continue
            fixed.append((n,o,k))
        intentional={frozenset((f'{s}_supported_output_shaft',f'{s}_W618_6_bearing_{j}')) for s in ('R','L') for j in (1,2)}
        intentional|={frozenset((f'{s}_supported_output_shaft',f'{s}_yaw_rotor_arm_carrier')) for s in ('R','L')}
        intentional|={frozenset((f'{s}_assumed_output_adapter',f'{s}_RSF5B50_IDT_case')) for s in ('R','L')}
        intentional|={frozenset((f'{s}_assumed_output_adapter',f'{s}_motor_pulley')) for s in ('R','L')}
        intentional|={frozenset((f'{s}_RSF5B50_IDT_case',f'{s}_PT1_connector_reserve')) for s in ('R','L')}
        pair=[]
        critical=[x for x in fixed if x[2] in {'motor','connector','bearing','shaft','rotor','carrier','cover','frame','pack','service','pulley'}]
        for i,(n,a,ka) in enumerate(critical):
            for m,c,kc in critical[i+1:]:
                if frozenset((n,m)) in intentional:continue
                q=overlap(a,c)
                if q>.01:pair.append({'a':n,'b':m,'mm3':round(q,3)})
        # Belt and lead keep-outs checked against every external fixed piece.
        routed=[]
        for n,o,k in objs:
            if k not in {'belt','lead'}:continue
            hits=[]
            for m,c,kc in critical:
                if (k=='belt' and (m in {n.replace('belt_corridor','motor_pulley'),n.replace('belt_corridor','output_pulley')})):continue
                if (k=='lead' and m in {n.replace('CAN_power_lead_keepout','RSF5B50_IDT_case'),n.replace('CAN_power_lead_keepout','PT1_connector_reserve')}):continue
                if k=='lead' and m=='CAN_5V_connector_service_reserve':continue
                q=overlap(o,c)
                if q>.01:hits.append({'component':m,'mm3':round(q,3)})
            routed.append({'component':n,'kind':k,'fixed_component_hits':hits})
        # Moving bilateral upper-arm shell gate: neutral, symmetric forward reach,
        # and reverse yaw; exact transformed shell vs torso/other arm parts.
        moving=[]
        motions=[('neutral',0,0),('both_forward',-30,30),('both_back',30,-30)]
        baseline={}
        for s in ('R','L'):
            for n in (f'{s}_upper_arm_front',f'{s}_upper_arm_back',f'{s}_shoulder_trim'):
                for sn,ss in skin.items():
                    if sn.startswith('torso_'):
                        q=overlap(parts[n],ss)
                        if q>.01:baseline[(n,sn)]=round(q,3)
        for pose,rr,ll in motions:
            angles={'R':rr,'L':ll};moved={}
            for side in ('R','L'):
                p=PIVOTS[side];a=angles[side]
                for name in (f'{side}_upper_arm_front',f'{side}_upper_arm_back',f'{side}_shoulder_trim'):
                    moved[name]=turn(parts[name],a,p)
            hits=[]
            for n,o in moved.items():
                for sn,ss in skin.items():
                    if sn.startswith('torso_') or (sn.startswith(('R_','L_')) and sn not in moved):
                        q=overlap(o,ss)
                        if q>.01:hits.append({'moving_skin':n,'fixed_skin':sn,'mm3':round(q,3)})
                for mn,mo,mk in objs:
                    if mk=='carrier' and mn.startswith(f'{n[0]}_yaw_rotor_arm_carrier'):continue
                    if mk in {'motor','bearing','shaft','rotor','carrier','cover','frame','pulley','belt'}:
                        q=overlap(o,mo)
                        if q>.01:hits.append({'moving_skin':n,'fixed_component':mn,'mm3':round(q,3)})
            excess=[]
            for h in hits:
                if 'fixed_skin' in h:
                    base=baseline.get((h['moving_skin'],h['fixed_skin']),0.)
                    if h['mm3']>base+1.:excess.append({**h,'source_neutral_mm3':base,'increase_mm3':round(h['mm3']-base,3)})
                else:excess.append(h)
            moving.append({'pose':pose,'arm_yaw_deg':angles,'exact_intersections':hits,'new_intersection_excess_over_source_neutral':excess})
        # Straight-up 100 mm pack withdrawal corridor (conservative AABB).
        path=box((PACK_C[0],PACK_C[1],PACK_C[2]+50),(PACK_SIZE[0],PACK_SIZE[1],PACK_SIZE[2]+100))
        path_hits=[{'skin':n,'mm3':round(overlap(path,s),3)} for n,s in service_skin.items() if overlap(path,s)>.01]
        pair_rows={'motor_motor':round(overlap(motors['R'],motors['L']),3),'pack':round(overlap(pack,service),3)}
        mass={'RSF5B50_motors_g':154.,'W618_6_bearings_g':7.6,
              'printed_frame_proxy_g':round(sum(vol(o) for n,o,k in objs if k=='frame')*1.24/1000,1),
              'shafts_and_belts_g':'unmeasured; not included','total_known_subtotal_g':'motors + bearings + modeled PLA frame only',
              'pack_g':107.,'pack_mass_separate':True}
        row={'wall_mm':wall,'topology':'two central anterior RSF-5B-50-IDT drives; 1:1 short belts to two independently W618-supported bilateral yaw shafts on a transverse spine; yaw carriers attach at the ape shoulder-root station',
             'motor_source':{'model':'Harmonic Drive RSF-5B-IDT 50:1','case_diameter_mm':25.,'axial_mm':35.8,'mass_g_each':77.,'allowable_continuous_torque_Nm_each':.29,'max_torque_Nm_each':.9,'continuous_speed_rpm':90.,'voltage':'7–30 VDC, 24V nominal integrated CANopen','price':'not sourced'},
             'layout':{'motor_output_datums_mm':{s:[MOTOR_C[s][0],MOTOR_C[s][1],MOTOR_C[s][2]] for s in ('R','L')},'yaw_output_datums_mm':PIVOTS,'pack_center_mm':PACK_C,'service_center_mm':SERVICE_C},
             'actuator_bearing_frame_mass':mass,'shell_component_rows':shell_rows,'fixed_nonmating_component_overlaps':pair,'routed_keepout_checks':routed,'pack_service_common_mm3':pair_rows['pack'],'pack_withdrawal_100mm_up_shell_hits':path_hits,'moving_arms':moving,
             'source_neutral_arm_to_torso_overlap_mm3':{f'{n}__{sn}':v for (n,sn),v in baseline.items()},
             'criteria':{'neutral_shell_contacts':sum(bool(x['shell_intersections']) for x in shell_rows),'fixed_nonmating_overlaps':len(pair),'pose_count':len(moving),'moving_no_new_intersection_excess':all(not x['new_intersection_excess_over_source_neutral'] for x in moving),'source_shell_has_neutral_arm_torso_overlaps':bool(baseline),'pack_withdrawal_clear':not path_hits},
             'limitations':['Shoulder yaw coupon only; shoulder roll/pitch, elbow, forearm, hands, legs and full-body mass/COM are not modeled.','The motors are manufacturer-rated 0.29 N·m allowable continuous at 50:1; shoulder duty/thermal load has not been calculated. Enclosed-shell thermal condition is unverified.','Crossbeams, cheeks, carriers, belt corridors and lead paths are conceptual envelopes. Bearing fits, fastening, stiffness, belt teeth/tension, output retention, assembly and tolerance are unverified.','Mass excludes shafts, belts, wiring, connectors, service electronics, fasteners and any detailed bearing block; pack is listed separately.','No claim of full robot or learned-policy readiness follows from this shoulder coupon.']}
        rows.append(row)
        if wall==2:exports={'parts':parts,'skin':skin,'objs':objs,'pack':pack,'service':service,'outer':outer_torso}
    (OUT/'report.json').write_text(json.dumps({'source':'experiments/sketch_ape_shell/build.py, Study(500, wall, gap=.65); no copied Microban geometry','rows':rows},indent=2))
    export_scene(exports)
    print('written',OUT/'report.json')
    for r in rows:print('wall',r['wall_mm'],'shell-hit components',r['criteria']['neutral_shell_contacts'],'fixed pair conflicts',r['criteria']['fixed_nonmating_overlaps'],'moving no-new-excess',r['criteria']['moving_no_new_intersection_excess'],'pack extraction',r['criteria']['pack_withdrawal_clear'],flush=True)

def export_scene(d):
    sc=trimesh.Scene()
    for n,s in d['skin'].items():add_mesh(sc,n,s,(227,220,204,120))
    for n,o,k in d['objs']:
        c={'motor':(50,115,185,255),'bearing':(85,130,80,255),'shaft':(125,138,100,255),'rotor':(142,146,95,255),'carrier':(150,170,110,200),'frame':(84,114,79,235),'belt':(194,140,55,190),'lead':(201,111,77,210),'connector':(203,110,78,220),'pulley':(194,140,55,255),'pack':(220,150,57,220),'service':(175,100,76,220),'cover':(180,100,90,170)}[k]
        add_mesh(sc,n,o,c)
    sc.export(OUT/'internal-spine-shoulder.glb')

if __name__=='__main__':main()
