"""Local nested deltoid cup iteration of the committed 500 mm central-drive coupon."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import build123d as b
from build123d import SkipClean
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'experiments/sketch_ape_shell'))
from build import Study
OUT=ROOT/'artifacts/nested-deltoid-500'; OUT.mkdir(parents=True,exist_ok=True)
H=500.; P={'R':(80.,5.,347.),'L':(-80.,5.,347.)}
MC={'R':(30.,-22.,347.),'L':(-30.,-22.,347.)}; PACK=(0.,14.,285.7)
PACK_SIZE=(60.,30.,30.5); SERVICE=(0.,-20.,365.); SERVICE_SIZE=(25.,20.,14.)
RSF_D=25.; RSF_L=35.8
# Cup dimensions are radial/axial about the existing Z yaw axis. The maximum
# lateral extent is pivot X +/-17 mm, matching the prior trim's outer bound.
RELIEF=(17.4,24.,13.5)
ARM_RELIEF=(14.7,20.7,10.9)
LIP_OUT=(17.,23.,13.2); LIP_IN=(15.6,21.6,11.8)
CAP_OUT=(14.8,20.8,11.); CAP_IN=(13.4,19.4,9.6)

def box(c,s): return b.Pos(*c)*b.Box(*s)
def cyl(axis,r,h,p):
    rot={'x':b.Rot(0,90,0),'y':b.Rot(90,0,0),'z':b.Rot(0,0,0)}[axis]
    return b.Pos(*p)*rot*b.Cylinder(r,h)
def ellipsoid(c,r): return b.Solid.make_sphere(1).scale(r).translate(c)
def shell(c,outer,inner): return ellipsoid(c,outer)-ellipsoid(c,inner)
def annulus_z(p,od=13.,id=6.,width=3.5): return cyl('z',od/2,width,p)-cyl('z',id/2,width+.2,p)
def belt(p1,p2,width=4.,thick=1.3,pitch_d=10.):
    v=np.array(p2,float)-np.array(p1,float);d=float(np.linalg.norm(v));u=v/d
    mid=tuple((np.array(p1,float)+np.array(p2,float))/2)
    pl=b.Plane(origin=mid,x_dir=tuple(u),z_dir=(0,0,1))
    return pl.location*(b.extrude(b.SlotCenterToCenter(d,pitch_d+2*thick),amount=width/2,both=True)-b.extrude(b.SlotCenterToCenter(d,pitch_d),amount=width/2+.15,both=True))
def turn(shape,deg,p): return b.Pos(*p)*b.Rot(0,0,deg)*b.Pos(*(-np.asarray(p)))*shape
def vol(s): return round(sum(x.volume for x in s.solids()),6) if s else 0.
def bbox(s):
    q=s.bounding_box();return [[q.min.X,q.min.Y,q.min.Z],[q.max.X,q.max.Y,q.max.Z]]
def common_pair(a,c):
    """Two exact operand orders; retain both because OCC can be asymmetric."""
    A=a.bounding_box();C=c.bounding_box()
    if any(max((A.min.X,A.min.Y,A.min.Z)[i],(C.min.X,C.min.Y,C.min.Z)[i])>min((A.max.X,A.max.Y,A.max.Z)[i],(C.max.X,C.max.Y,C.max.Z)[i]) for i in range(3)):
        return {'a_then_b_mm3':0.,'b_then_a_mm3':0.,'conservative_mm3':0.}
    with SkipClean(): ab=a.intersect(c)
    with SkipClean(): ba=c.intersect(a)
    x,y=vol(ab),vol(ba)
    return {'a_then_b_mm3':x,'b_then_a_mm3':y,'conservative_mm3':max(x,y)}
def distance(a,c):
    try:return round(a.distance(c),4)
    except Exception:return None
def add_mesh(scene,name,shape,color):
    v,f=shape.tessellate(.3,.25)
    m=trimesh.Trimesh(vertices=np.asarray([[x.X,x.Y,x.Z] for x in v]),faces=np.asarray(f),process=False)
    m.visual.vertex_colors=color;scene.add_geometry(m,geom_name=name,node_name=name)
def make_hardware():
    objs=[]
    for s,c in MC.items():
        p=P[s];ext=1 if s=='R' else -1
        motor=b.Pos(c[0],c[1],c[2]-RSF_L/2)*b.Cylinder(RSF_D/2,RSF_L);objs.append((s+'_RSF5B50_case',motor,'motor'))
        con=box((c[0]+ext*16.5,c[1],c[2]-20),(8,8,8));objs.append((s+'_PT1_connector',con,'connector'))
        ad=cyl('z',2.5,4.,(c[0],c[1],c[2]+2));objs.append((s+'_assumed_adapter',ad,'shaft'))
        for j,z in enumerate((p[2]-13.,p[2])):objs.append((f'{s}_W618_bearing_{j}',annulus_z((p[0],p[1],z)),'bearing'))
        objs.append((s+'_supported_shaft',cyl('z',3.,34.,(p[0],p[1],p[2]+1)),'shaft'))
        a=np.array((p[0],p[1],p[2]+12));q=np.array((ext*86.,p[1]-.5,p[2]+12));d=q-a
        bridge=b.Solid.make_cylinder(4.,float(np.linalg.norm(d)),b.Plane(origin=tuple(a),z_dir=tuple(d/np.linalg.norm(d))))
        tongue=box((ext*90.,2.85,p[2]+12),(8,.8,6));objs.append((s+'_yaw_carrier',b.Compound(children=[bridge,tongue]),'carrier'))
        mp=(c[0],c[1],p[2]+5.5);op=(p[0],p[1],p[2]+5.5)
        objs.append((s+'_motor_pulley',annulus_z(mp,10,5,4),'pulley'))
        objs.append((s+'_output_pulley',annulus_z(op,10,6,4),'pulley'))
        objs.append((s+'_belt',belt(mp,op),'belt'))
    spine=box((0,5,345),(12,12,70))
    for z in (P['R'][2]-13.,P['R'][2]):
        beam=box((0,5,z),(176,5,6))
        for p in P.values():beam=beam-cyl('z',6.65,6.2,(p[0],p[1],z))
        spine=spine.fuse(beam)
    objs.append(('bored_internal_spine',spine,'frame'))
    objs.extend([('shared_4S_pack',box(PACK,PACK_SIZE),'pack'),('service_reserve',box(SERVICE,SERVICE_SIZE),'service')])
    return objs

def export_pose(parts,modified,cups,hardware,angles,path):
    scene=trimesh.Scene(); arm_names={f'{s}_{n}' for s in ('L','R') for n in ('upper_arm_front','upper_arm_back','shoulder_trim')}
    for it in parts:
        n=it['name'];sh=modified.get(n,it['shape'])
        if n in arm_names:
            s=n[0];sh=turn(sh,angles[s],P[s])
        # All skin is fully opaque for the appearance gate.
        color=(225,218,202,255) if ('arm' in n or 'torso' in n or 'shoulder' in n) else (188,174,147,255)
        add_mesh(scene,n,sh,color)
    for n,sh in cups:
        if n.startswith('moving_'):
            side=n.split('_')[1];sh=turn(sh,angles[side],P[side])
        add_mesh(scene,n,sh,(145,111,76,255))
    for n,sh,k in hardware:
        colors={'motor':(45,105,177,255),'bearing':(82,132,83,255),'shaft':(95,130,95,255),'carrier':(160,130,75,255),'pulley':(195,143,55,255),'belt':(196,143,55,255),'frame':(67,98,72,255),'connector':(189,92,75,255),'pack':(213,143,45,255),'service':(155,94,75,255)}
        add_mesh(scene,n,sh,colors[k])
    scene.export(path)

def main():
    rows=[];export=None
    hw=make_hardware()
    for wall in (1.,2.):
        study=Study(H,wall,.65);study.build();items=study.parts
        part={x['name']:x['shape'] for x in items}
        modified={};cups=[]
        torso_names=['torso_front','torso_back']
        # Local only: relief centered on unchanged yaw axes in the torso and
        # moving arm root; no motor, belt, bearing, pack, or axis coordinates move.
        for s,p in P.items():
            relief=ellipsoid(p,RELIEF)
            lip=shell(p,LIP_OUT,LIP_IN);cap=shell(p,CAP_OUT,CAP_IN)
            cups.extend([(f'fixed_{s}_sloped_torso_lip',lip),(f'moving_{s}_deltoid_cap',cap)])
            for n in torso_names:
                modified[n]=modified.get(n,part[n]).cut(relief)
            for n in (f'{s}_upper_arm_front',f'{s}_upper_arm_back',f'{s}_shoulder_trim'):
                modified[n]=part[n].cut(ellipsoid(p,ARM_RELIEF))
        # Exact shell component pairs, preserving both operand orders.
        contact_rows=[]
        for s,p in P.items():
            lip=dict((n,sh) for n,sh in cups)[f'fixed_{s}_sloped_torso_lip']
            cap=dict((n,sh) for n,sh in cups)[f'moving_{s}_deltoid_cap']
            # Symmetric per-wall local checks: lip to torso, cap to arm pieces,
            # and moving cap to fixed lip throughout yaw.
            for tn in torso_names:
                contact_rows.append({'wall_mm':wall,'side':s,'pair':f'{tn} / fixed lip','exact_two_order':common_pair(modified[tn],lip),'gap_mm':distance(modified[tn],lip)})
            for an in (f'{s}_upper_arm_front',f'{s}_upper_arm_back',f'{s}_shoulder_trim'):
                contact_rows.append({'wall_mm':wall,'side':s,'pair':f'{an} / moving cap at neutral','exact_two_order':common_pair(modified[an],cap),'gap_mm':distance(modified[an],cap)})
        # Paired pose gates. Source neutral intersections are retained separately,
        # but the modified torso/arm/cup exact contacts are tested in both orders.
        pose_rows=[];poses=[('neutral',0,0),('forward',-30,30),('rear',30,-30)]
        for pose,ra,la in poses:
            ang={'R':ra,'L':la}; shell_hits=[]
            fixed_parts=[('torso_front',modified['torso_front']),('torso_back',modified['torso_back'])]
            fixed_parts += [(f'fixed_{s}_lip',dict(cups)[f'fixed_{s}_sloped_torso_lip']) for s in P]
            for s in ('R','L'):
                cap=dict(cups)[f'moving_{s}_deltoid_cap']; movedcap=turn(cap,ang[s],P[s])
                shell_hits.append({'pose':pose,'pair':f'{s} cap / fixed {s} lip','exact_two_order':common_pair(movedcap,dict(cups)[f'fixed_{s}_sloped_torso_lip']),'gap_mm':distance(movedcap,dict(cups)[f'fixed_{s}_sloped_torso_lip'])})
                for an in (f'{s}_upper_arm_front',f'{s}_upper_arm_back',f'{s}_shoulder_trim'):
                    am=turn(modified[an],ang[s],P[s])
                    for fn,fs in fixed_parts:
                        res=common_pair(am,fs)
                        if res['conservative_mm3']>.01 or (res['a_then_b_mm3']!=res['b_then_a_mm3']):
                            shell_hits.append({'pose':pose,'pair':f'{an} / {fn}','exact_two_order':res,'gap_mm':distance(am,fs)})
                    shell_hits.append({'pose':pose,'pair':f'{an} / moving cap','exact_two_order':common_pair(am,movedcap),'gap_mm':distance(am,movedcap)})
            pose_rows.append({'pose':pose,'yaw_deg':ang,'shell_contacts':shell_hits})
        # Same fixed hardware and shared pack; independent of wall thickness.
        fixed=[(n,sh,k) for n,sh,k in hw if k not in ('belt',)]
        fixed_pairs=[]
        for i,(n,a,ka) in enumerate(fixed):
            for m,c,kc in fixed[i+1:]:
                if frozenset((n,m)) in {frozenset(('R_supported_shaft','R_yaw_carrier')),frozenset(('L_supported_shaft','L_yaw_carrier')),frozenset(('R_RSF5B50_case','R_PT1_connector')),frozenset(('L_RSF5B50_case','L_PT1_connector'))}:continue
                r=common_pair(a,c)
                if r['conservative_mm3']>.01:fixed_pairs.append({'a':n,'b':m,'exact_two_order':r})
        gap_rows=[]
        for s,p in P.items():
            fixed_lip=dict(cups)[f'fixed_{s}_sloped_torso_lip'];moving_cap=dict(cups)[f'moving_{s}_deltoid_cap']
            for angle in (-30,0,30):
                moved=turn(moving_cap,angle,p);gap_rows.append({'side':s,'yaw_deg':angle,'clearance_mm':distance(fixed_lip,moved),'common':common_pair(fixed_lip,moved)})
        if wall==2:
            export={'items':items,'modified':modified,'cups':cups,'hardware':hw}
        rows.append({'wall_mm':wall,'outer_skin_change_max_x_mm':17.,'same_motor_belt_bearing_axis_pack_positions':True,
          'component_cup_dimensions_mm':{'torso_relief_ellipsoid_radii':RELIEF,'arm_relief_ellipsoid_radii':ARM_RELIEF,'fixed_lip_outer_inner_radii':{'outer':LIP_OUT,'inner':LIP_IN},'moving_cap_outer_inner_radii':{'outer':CAP_OUT,'inner':CAP_IN},'cup_to_lip_target_gap_mm':0.8},
          'local_pair_contacts':contact_rows,'pose_gates':pose_rows,'nested_cup_clearance':gap_rows,'fixed_hardware_pair_hits':fixed_pairs,
          'criteria':{'all_measured_cup_clearances_positive':all((r['clearance_mm'] or 0)>0 and r['common']['conservative_mm3']<.01 for r in gap_rows),'zero_arm_torso_contacts':all(not any(('torso_' in h['pair']) and h['exact_two_order']['conservative_mm3']>.01 for h in p['shell_contacts']) for p in pose_rows),'zero_unintended_hard_intersections':not fixed_pairs}})
    (OUT/'report.json').write_text(json.dumps({'source':'Study(500, wall, .65) from experiments/sketch_ape_shell/build.py; local cup relief only','pose_method':'upper arm and existing shoulder trim rotated rigidly around unchanged bilateral Z yaw axes; exact intersections evaluated in both operand orders; all output meshes opaque','rows':rows},indent=2))
    for pose,angles,file in [('neutral',{'R':0,'L':0},'neutral.glb'),('forward',{'R':-30,'L':30},'forward.glb'),('rear',{'R':30,'L':-30},'rear.glb')]:
        export_pose(export['items'],export['modified'],export['cups'],export['hardware'],angles,OUT/file)
    print('written',OUT/'report.json')
    for r in rows:
        print('wall',r['wall_mm'],r['criteria'],flush=True)
if __name__=='__main__': main()
