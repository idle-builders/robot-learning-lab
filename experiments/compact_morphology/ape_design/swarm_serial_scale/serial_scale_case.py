"""Exact-envelope scale gate for a fresh 21-axis serial ape skeleton.

This uses the sketch shell's source station geometry and the manufacturer XL330 STEP.
Reference robots inform the joint graph only; no reference CAD is imported.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import build123d as b
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'experiments/sketch_ape_shell'))
from build import Study, cranial_loft
OUT=ROOT/'artifacts/serial-scale'
STEP=ROOT/'artifacts/round2-hip-350/vendor-cad/XL_XC330_official.stp'
SHELL_STATIONS={
 'torso':[(160,0,2,42,25),(173,0,2,49,29),(199,0,3,58,34),(229,0,4,63,35),(253,0,5,63,30),(264,0,5,60,27),(274,0,5,35,22),(278,0,5,23,18)],
 'pelvis':[(130,0,5,9,11),(135,0,5,20,17),(145,0,5,34,23),(155,0,4,43,26),(164,0,3,46,26),(178,0,3,49,29)],
 'thigh':[(87,42,3,16,18),(98,43,3,20,22),(124,39,3,24,25),(145,35,4,22,23),(157,33,4,16,18),(163,31,4,8,10)],
 'shin':[(23,46,3,21,20),(36,46,3,23,23),(57,45,4,22,23),(75,43,4,19,20),(83,42,3,16,17)],
 'boot':[(2,46,-8,30,34),(5,46,-8,31,35),(12,46,-7,29,33),(20,46,-1,24,25),(26,46,3,19,19)],
 'upper_arm':[(199,93,2,14,17),(207,94,2,17,20),(228,89,3,20,22),(249,80,4,19,21),(260,73,4,14,17),(265,70,4,6,9)],
 'forearm':[(123,99,1,15,16),(133,102,1,18,19),(155,103,1,21,22),(178,101,1,18,20),(190,97,1,13,15)]}
AXES=[]
for side,hand in [(1,'R'),(-1,'L')]:
    AXES += [(f'{hand}_hip_yaw','leg','z',(side*29,0,160)),(f'{hand}_hip_roll','leg','y',(side*32,0,158)),(f'{hand}_hip_pitch','leg','x',(side*32,4,149)),(f'{hand}_knee','leg','x',(side*42,3,86)),(f'{hand}_ankle_pitch','leg','x',(side*46,4,24))]
    AXES += [(f'{hand}_shoulder_yaw','arm','z',(side*63,4,245)),(f'{hand}_shoulder_roll','arm','y',(side*66,4,245)),(f'{hand}_shoulder_pitch','arm','x',(side*70,4,237)),(f'{hand}_elbow','arm','x',(side*96,2,194)),(f'{hand}_hand_close','hand','x',(side*99,1,122))]
AXES += [('head_yaw','head','z',(0,4,276))]
assert len(AXES)==21

def volume(s): return round(sum(x.volume for x in s.solids()),3) if s else 0.0
def bbox_overlap(a,c):
    # Do not cache by id: temporary OCC shapes are released and Python can
    # recycle ids, making a later shape inherit a stale bounding box.
    x=a.bounding_box();y=c.bounding_box()
    xmin=[x.min.X,x.min.Y,x.min.Z];xmax=[x.max.X,x.max.Y,x.max.Z]
    ymin=[y.min.X,y.min.Y,y.min.Z];ymax=[y.max.X,y.max.Y,y.max.Z]
    return all(xmin[i]<=ymax[i] and ymin[i]<=xmax[i] for i in range(3))
def axis_rot(axis): return {'x':b.Rot(0,90,0),'y':b.Rot(-90,0,0),'z':b.Rot(0,0,0)}[axis]
def axis_cyl(axis,r,h,p):
    rot={'x':b.Rot(0,90,0),'y':b.Rot(90,0,0),'z':b.Rot(0,0,0)}[axis]
    return b.Pos(*p)*rot*b.Cylinder(r,h)
def box_axis(axis,size,p):
    dims={'x':(size[0],size[1],size[2]),'y':(size[1],size[0],size[2]),'z':(size[1],size[2],size[0])}[axis]
    return b.Pos(*p)*b.Box(*dims)
def placed_motor(shape,axis,point):
    return b.Pos(*point)*axis_rot(axis)*b.Pos(0,0,-6.5)*shape
def placed_case(axis,point):
    # Exact STEP bounds normalized about its output face: X20 × Y34 × Z29 mm.
    raw=b.Pos(0,-7.5,-8)*b.Box(20,34,29)
    return placed_motor(raw,axis,point)
def mirror_x(shape): return shape.mirror(b.Plane.YZ)
def shell_shapes(height,wall,relief=0):
    study=Study(height,wall,.65); scale=height/350
    result={}
    for name,stations in SHELL_STATIONS.items():
        st=[tuple(float(v) for v in row) for row in stations]
        if relief and name=='torso':
            # 10 mm total front/back cover relief, blended from z=199 to 264 mm.
            st=[(z,x,y,rx,ry+(5 if z in (229,253) else 2.5 if z in (199,264) else 0)) for z,x,y,rx,ry in st]
        if relief and name=='pelvis':
            # 10 mm total lateral/depth relief around the covered hip bowl only.
            st=[(z,x,y,rx+(5 if z in (145,155,164) else 2.5 if z==135 else 0),ry+(5 if z in (145,155,164) else 2.5 if z==135 else 0)) for z,x,y,rx,ry in st]
        hollow,outer=study.hollow_loft(st)
        result[name]=(outer.scale(scale),hollow.scale(scale))
    # Source head cranial shell and inner hollow are made in source coordinates then scaled.
    head_outer=cranial_loft((0,0,313),(39,35,37))
    head_inner=cranial_loft((0,0,313),(39-study.t,35-study.t,37-study.t))
    result['head']=(head_outer.scale(scale),(head_outer-head_inner).scale(scale))
    # Mirror other half links for the source left/right body.
    for key in ('thigh','shin','boot','upper_arm','forearm'):
        outer,skin=result[key]
        result[key+'_L']=(mirror_x(outer),mirror_x(skin))
        result[key+'_R']=(outer,skin)
    # A rounded palm envelope is the body for the hand-closing actuator.
    palm_o=b.Solid.make_sphere(1).scale((16*scale,13*scale,20*scale)).translate((99*scale,-1*scale,106*scale))
    palm_i=b.Solid.make_sphere(1).scale(((16-study.t)*scale,(13-study.t)*scale,(20-study.t)*scale)).translate((99*scale,-1*scale,106*scale))
    result['hand_R']=(palm_o,palm_o-palm_i)
    result['hand_L']=(mirror_x(palm_o),mirror_x(palm_o-palm_i))
    for side,label in [(1,'R'),(-1,'L')]:
        hip_cuff=b.Solid.make_sphere(1).scale((17*scale,22*scale,12*scale)).translate((side*30*scale,0,158*scale))
        shoulder_cuff=b.Solid.make_sphere(1).scale((14*scale,21*scale,13*scale)).translate((side*65*scale,4*scale,245*scale))
        knee_cuff=b.Solid.make_sphere(1).scale((18*scale,20*scale,12*scale)).translate((side*42*scale,3*scale,86*scale))
        elbow_cuff=b.Solid.make_sphere(1).scale((16*scale,18*scale,13*scale)).translate((side*96*scale,2*scale,194*scale))
        wrist_cuff=b.Solid.make_sphere(1).scale((14*scale,16*scale,10*scale)).translate((side*99*scale,1*scale,122*scale))
        result['hip_cuff_'+label]=(hip_cuff,hip_cuff)
        result['shoulder_cuff_'+label]=(shoulder_cuff,shoulder_cuff)
        result['knee_cuff_'+label]=(knee_cuff,knee_cuff)
        result['elbow_cuff_'+label]=(elbow_cuff,elbow_cuff)
        result['wrist_cuff_'+label]=(wrist_cuff,wrist_cuff)
    # All silhouettes/links scale from the approved 350 mm stations; covers stay source-faithful.
    return result,scale
def axis_lobes(name,kind):
    side=name[0] if name[0] in ('L','R') else ''
    if kind=='leg':
        if 'hip_yaw' in name:return ['torso','pelvis']
        if 'hip_' in name:return [f'thigh_{side}','pelvis','torso']
        if 'knee' in name:return [f'thigh_{side}',f'shin_{side}']
        return [f'shin_{side}',f'boot_{side}']
    if kind=='arm':
        if 'shoulder' in name:return ['torso',f'upper_arm_{side}']
        return [f'upper_arm_{side}',f'forearm_{side}']
    if kind=='hand':return [f'forearm_{side}',f'hand_{side}']
    return ['head','torso']
def motor_part_records(height,wall,relief,step_shape,hand_drive='palm'):
    shells,scale=shell_shapes(height,wall,relief); motors=[]; supports=[]; support_details=[]; transfers=[]; skin_hits=[]; containment=[]
    # Retain independent shell solids. Sequential cuts avoid invalid multi-solid
    # fuse results (the official XL330 STEP itself is not a valid fused solid).
    outer_cache={}
    for i,(name,kind,axis,basepoint) in enumerate(AXES):
        pt=tuple(v*scale for v in basepoint); side=-1 if name.startswith('L_') else 1
        mount_pt=(abs(pt[0]),pt[1],pt[2]) if side<0 else pt
        # Parent-link drives keep the round package inside the limb and transfer
        # output to the anatomical joint with a conceptual 1:1 timing-belt path.
        # Motor placement is a geometric screen; belt ratios/load capacity remain
        # unresolved and are not counted as solved transmission hardware.
        motor_pt=mount_pt
        if 'knee' in name: motor_pt=(mount_pt[0],3*scale,130*scale)
        if 'elbow' in name: motor_pt=((100 if height>=500 else 105)*scale,3*scale,220*scale)
        if 'hand_close' in name and hand_drive=='forearm': motor_pt=(99*scale,8.5*scale,175*scale)
        drive_axis='z' if 'hand_close' in name and hand_drive=='forearm' else axis
        visual=placed_motor(step_shape,drive_axis,motor_pt);mot=placed_case(drive_axis,motor_pt)
        if side<0:visual=mirror_x(visual);mot=mirror_x(mot)
        lobes=axis_lobes(name,kind)
        regions=[]
        for key in lobes:
            if key not in shells: continue
            regions.append(shells[key][0])
        cuff_key=None
        if 'shoulder_' in name:cuff_key='shoulder_cuff_'+name[0]
        elif 'hip_' in name:cuff_key='hip_cuff_'+name[0]
        elif 'knee' in name:cuff_key='knee_cuff_'+name[0]
        elif 'elbow' in name:cuff_key='elbow_cuff_'+name[0]
        elif 'hand_close' in name:cuff_key='wrist_cuff_'+name[0]
        if cuff_key:
            regions.append(shells[cuff_key][0])
        union_key=tuple(sorted(lobes+([cuff_key] if cuff_key else [])))
        if union_key not in outer_cache:
            outer_cache[union_key]=regions
        cover_regions=outer_cache[union_key]
        total=volume(mot)
        remaining=mot
        for region in cover_regions:
            if bbox_overlap(remaining,region):
                remaining=remaining.cut(region)
        outv=max(0,round(volume(remaining),3))
        outer_inter=max(0,round(total-outv,3))
        per_lobe=[]
        for key in lobes:
            if key in shells:
                iv=volume(mot.intersect(shells[key][0])) if bbox_overlap(mot,shells[key][0]) else 0.0
                per_lobe.append({'lobe':key,'inside_mm3':iv,'outside_lobe_mm3':max(0,round(total-iv,3))})
        containment.append({'axis':name,'component':'XL330 conservative case envelope','package_volume_mm3':total,'candidate_outer_lobes':lobes,'small_cuff':cuff_key,'per_lobe':per_lobe,'inside_adjacent_covered_union_mm3':outer_inter,'outside_adjacent_covered_union_mm3':outv})
        motors.append((name,visual,mot,axis,pt,kind,lobes))
        # Independent 6 mm output shaft, two W618/6 envelopes, and cheek/seat plates.
        bearings=[]; cheeks=[]
        for d in (-8,8):
            q=list(mount_pt);q['xyz'.index(axis)]+=d
            bearing=axis_cyl(axis,6.5,3.5,q)-axis_cyl(axis,3,4,q)
            cheek=box_axis(axis,(4,16,16),q)
            if side<0:bearing=mirror_x(bearing);cheek=mirror_x(cheek)
            bearings.append(bearing);cheeks.append(cheek)
        shaft=axis_cyl(axis,3,22,mount_pt);support_env=box_axis(axis,(20,16,16),mount_pt)
        if side<0:shaft=mirror_x(shaft);support_env=mirror_x(support_env)
        supports.append((name+' support',support_env,axis,pt,kind,lobes))
        support_details.append((name+' support details',bearings+cheeks+[shaft],axis,pt,kind,lobes))
        support_remaining=support_env
        for region in cover_regions:
            if bbox_overlap(support_remaining,region):support_remaining=support_remaining.cut(region)
        support_out=max(0,round(volume(support_remaining),3))
        support_inside=max(0,round(volume(support_env)-support_out,3))
        containment.append({'axis':name,'component':'W618 pair plus output-seat conservative envelope','package_volume_mm3':volume(support_env),'candidate_outer_lobes':lobes,'small_cuff':cuff_key,'inside_adjacent_covered_union_mm3':support_inside,'outside_adjacent_covered_union_mm3':support_out})
        if motor_pt!=mount_pt and 'hand_close' in name and hand_drive=='forearm':
            # A straight 5 mm Bowden/tendon corridor from forearm spool to wrist.
            delta=b.Vector(*mount_pt)-b.Vector(*motor_pt); length=delta.length
            route=b.Solid.make_cylinder(2.5*scale,length,b.Plane(origin=motor_pt,z_dir=delta))
            if side<0:route=mirror_x(route)
            rem=route
            for region in cover_regions:
                if bbox_overlap(rem,region):rem=rem.cut(region)
            transfers.append((name+' tendon route',route))
            containment.append({'axis':name,'component':'forearm-to-wrist tendon corridor','package_volume_mm3':volume(route),'candidate_outer_lobes':lobes,'outside_adjacent_covered_union_mm3':round(volume(rem),3),'note':'5 mm straight corridor only; curved tendon, liner, attachment and closing force unresolved'})
            assigned=lobes[0]
            if assigned in shells and bbox_overlap(route,shells[assigned][1]):
                v=volume(route.intersect(shells[assigned][1]))
                if v>.01:skin_hits.append({'axis':name,'component':'forearm-to-wrist tendon corridor','skin':assigned,'intersection_mm3':v})
        elif motor_pt!=mount_pt:
            # Simplified non-geared belt corridor with endpoint pulley envelopes.
            # The route is inspectable, but belt tooth form, ratio, tensioner,
            # stiffness and transfer torque have not been engineered.
            dz=abs(motor_pt[2]-mount_pt[2]); mid=((motor_pt[0]+mount_pt[0])/2,(motor_pt[1]+mount_pt[1])/2,(motor_pt[2]+mount_pt[2])/2)
            # The cuboid conservatively spans both pulley circles plus tangent
            # belt legs; it is a routing corridor, not a pulley CAD design.
            belt=b.Pos(*mid)*b.Box(16,3,dz+13)
            if side<0:belt=mirror_x(belt)
            rem=belt
            for region in cover_regions:
                if bbox_overlap(rem,region):rem=rem.cut(region)
            transfers.append((name+' transfer',belt))
            containment.append({'axis':name,'component':'conceptual belt/pulley corridor envelope','package_volume_mm3':volume(belt),'candidate_outer_lobes':lobes,'outside_adjacent_covered_union_mm3':round(volume(rem),3),'note':'layout corridor only; transfer torque, ratio, belt and tensioner are unresolved'})
            assigned=lobes[0]
            if assigned in shells and bbox_overlap(belt,shells[assigned][1]):
                v=volume(belt.intersect(shells[assigned][1]))
                if v>.01:skin_hits.append({'axis':name,'component':'conceptual belt/pulley corridor envelope','skin':assigned,'intersection_mm3':v})
        # These are exact BRep intersections of conservative package/seat envelopes with the source skin.
        for component,label in ((mot,'XL330 case envelope'),(support_env,'output-seat envelope')):
            assigned=lobes[0]
            if assigned in shells and bbox_overlap(component,shells[assigned][1]):
                v=volume(component.intersect(shells[assigned][1]))
                if v>.01:skin_hits.append({'axis':name,'component':label,'skin':assigned,'intersection_mm3':v})
        if (i+1)%5==0:print(f"  measured {i+1}/21 motor and support axes at {height} mm",flush=True)
    return shells,motors,supports,support_details,transfers,containment,skin_hits,scale

def build_case(height,wall,step_shape,relief=0,hand_drive='palm'):
    shells,motors,supports,support_details,transfers,containment,skin_hits,scale=motor_part_records(height,wall,relief,step_shape,hand_drive)
    selected=AXES
    selected_names={a[0] for a in selected}
    motor_map={x[0]:x for x in motors};support_map={x[0].replace(' support',''):x for x in supports}
    records=[x for x in containment if x['axis'] in selected_names]
    battery_center=(0,0,200*scale); battery_size=(60,30,30.5)
    battery=b.Pos(*battery_center)*b.Box(*battery_size)
    # Broad, clearly labelled service/controller reserve alongside the sourced 4S pack.
    service=b.Pos(42*scale,0,210*scale)*b.Box(18,20,14)
    torso=shells['torso'][1]
    service_hits=[]
    for n,shape in [('battery',battery),('service reserve',service)]:
        v=volume(shape.intersect(torso)) if bbox_overlap(shape,torso) else 0
        service_hits.append({'component':n,'torso_wall_intersection_mm3':v})
    return dict(height_mm=height,wall_mm=wall,scale=scale,relief_mm=relief,hand_drive=hand_drive,axis_datums=[{'axis':n,'kind':k,'rotation_axis':a,'output_mm':tuple(v*scale for v in p)} for n,k,a,p in AXES if n in selected_names],containment=records,skin_intersections=[h for h in skin_hits if h['axis'] in selected_names],self_interference='skipped after the individual package/support envelope failures',service_intersections=service_hits,motors=motors,supports=supports,support_details=support_details,transfers=transfers,shells=shells,scale_factor=scale)

def export_viewer(case,all_axes=True):
    scene=trimesh.Scene(); col_shell=(225,218,202,95); col_motor=(55,111,159,255);col_support=(184,132,55,255)
    shells=case['shells']
    for key in ('torso','pelvis','hip_cuff_R','hip_cuff_L','shoulder_cuff_R','shoulder_cuff_L','knee_cuff_R','knee_cuff_L','elbow_cuff_R','elbow_cuff_L','wrist_cuff_R','wrist_cuff_L','thigh_R','thigh_L','shin_R','shin_L','boot_R','boot_L','upper_arm_R','upper_arm_L','forearm_R','forearm_L','hand_R','hand_L','head'):
        outer=shells[key][0]
        vv,ff=outer.tessellate(.7,.7)
        mesh=trimesh.Trimesh(np.array([[v.X,v.Y,v.Z] for v in vv]),np.asarray(ff),process=False)
        mesh.visual.vertex_colors=col_shell;scene.add_geometry(mesh,geom_name='skin_'+key,node_name='skin_'+key)
    for name,visual,shape,*_ in case['motors']:
        # Show the conservative, dimensioned package envelope; raw STEP is retained as the CAD source.
        vv,ff=shape.tessellate(.6,.6);mesh=trimesh.Trimesh(np.array([[v.X,v.Y,v.Z] for v in vv]),np.asarray(ff),process=False);mesh.visual.vertex_colors=col_motor;scene.add_geometry(mesh,geom_name=name,node_name=name)
    for name,shapes,*_ in case['support_details']:
        for i,shape in enumerate(shapes):
            vv,ff=shape.tessellate(.5,.5);mesh=trimesh.Trimesh(np.array([[v.X,v.Y,v.Z] for v in vv]),np.asarray(ff),process=False);mesh.visual.vertex_colors=col_support;scene.add_geometry(mesh,geom_name=f'{name}_{i}',node_name=f'{name}_{i}')
    for name,shape in case['transfers']:
        vv,ff=shape.tessellate(.5,.5);mesh=trimesh.Trimesh(np.array([[v.X,v.Y,v.Z] for v in vv]),np.asarray(ff),process=False);mesh.visual.vertex_colors=(194,72,68,255);scene.add_geometry(mesh,geom_name=name,node_name=name)
    scene.export(OUT/'serial-scale.glb')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scale',type=int,action='append',choices=[350,400,450,500]);ap.add_argument('--wall',type=int,choices=[1,2],default=1);ap.add_argument('--relief-mm',type=int,choices=[0,10],default=0);ap.add_argument('--hand-drive',choices=['palm','forearm'],default='palm');ap.add_argument('--append-report',action='store_true',help='append/replace matching height-wall-axis rows in existing report.json');args=ap.parse_args()
    if not STEP.exists():raise SystemExit(f'official XL330 STEP missing: {STEP}')
    shape=b.import_step(str(STEP));OUT.mkdir(parents=True,exist_ok=True)
    scales=args.scale or [350,400,450,500];reports=[]
    for h in scales:
        case=build_case(h,args.wall,shape,args.relief_mm,args.hand_drive)
        records=case['containment']; out=[r for r in records if r['outside_adjacent_covered_union_mm3']>.1]
        print(f"{h} mm: {len(out)}/{len(records)} motor/support envelopes outside adjacent covered envelope; {len(case['skin_intersections'])} exact envelope/skin intersections",flush=True)
        report={k:v for k,v in case.items() if k not in ('motors','supports','support_details','transfers','shells')}
        report['parent_link_drive_layout']={'hand_drive':args.hand_drive,'knee_motor_output_source_mm':[42,3,130],'knee_joint_output_source_mm':[42,3,86],'elbow_motor_output_source_mm':[100 if h>=500 else 105,3,220],'elbow_joint_output_source_mm':[96,2,194],'hand_forearm_motor_output_source_mm':[99,8.5,175] if args.hand_drive=='forearm' else None,'hand_wrist_output_source_mm':[99,1,122],'transfer':'knee/elbow use conceptual belt corridor; forearm hand option uses a 5 mm straight tendon corridor. Route geometry only, no force/ratio/tension/attachment proof; transfer mass excluded.'}
        report.update(axis_count=21,actuator='ROBOTIS XL330-M288-T x21',official_case_envelope_mm=[20,34,29],package_pose_center_from_output_mm=[0,-7.5,-14.5],motor_mass_g=378,bearing_count=42,bearing_mass_g=79.8,pack={'size_mm':[60,30,30.5],'center_mm':(0,0,200*case['scale_factor']),'mass_g':107,'nominal_v':14.8},power='XL330 requires regulated 5 V TTL; buck current/thermal envelope not selected',motor_specs={'mass_g_each':18,'stall_torque_Nm_at_5V':.52,'continuous_torque':'unpublished','voltage_V':[3.7,6.0],'bus':'TTL'},total_known_mass_g=564.8,unresolved_structure_mass='carrier cheek material, fasteners, service electronics, cable and wiring not included',mass_centres='output datums and pack center recorded; motor/case CGs and final frame/COM not solved',torque_note='0.52 Nm is stall torque; XL330 continuous duty is unpublished for all axes.')
        reports.append(report)
        reduced_names={a[0] for a in AXES if not a[0].endswith('shoulder_yaw')}
        reduced_records=[x for x in case['containment'] if x['axis'] in reduced_names]
        reduced_skin=[x for x in case['skin_intersections'] if x['axis'] in reduced_names]
        reports.append({'height_mm':h,'wall_mm':args.wall,'relief_mm':args.relief_mm,'axis_count':19,'removed_axes':['R_shoulder_yaw','L_shoulder_yaw'],'retained_axes':['R_hand_close','L_hand_close'],'containment':reduced_records,'skin_intersections':reduced_skin,'self_interference':case['self_interference'],'service_intersections':case['service_intersections'],'motor_mass_g':342,'bearing_count':38,'bearing_mass_g':72.2,'pack_mass_g':107,'total_known_mass_g':521.2,'note':'19-axis comparison removes bilateral shoulder yaw and retains both opening-paw axes; support/frame mass and COM remain unmodeled.'})
        # Use only source-loft tests as a quick decisive gate; do not run poses unless all static bounds clear.
        if h==scales[-1]: export_viewer(case)
    if args.append_report and (OUT/'report.json').exists():
        previous=json.loads((OUT/'report.json').read_text()).get('cases',[])
        new_keys={(r['height_mm'],r['wall_mm'],r['relief_mm'],r['axis_count']) for r in reports}
        reports=[r for r in previous if (r['height_mm'],r['wall_mm'],r.get('relief_mm',0),r['axis_count']) not in new_keys]+reports
    static_pass=all(not any(r['outside_adjacent_covered_union_mm3']>.1 for r in rcase['containment']) and not rcase['skin_intersections'] for rcase in reports)
    (OUT/'report.json').write_text(json.dumps({'cases':reports,'individual_package_and_skin_envelope_gate_clear':static_pass,'assembly_self_interference':'not run after individual containment/skin gates; do not interpret a clear envelope as an assembly pass','pose_sweep':'skipped because all component and assembly static gates are not cleared'},indent=2)+'\n')
    rows=['height_mm,wall_mm,relief_mm,axis_count,outside_envelopes,total_envelopes,skin_intersections,max_outside_component,max_outside_mm3']
    for r in reports:
        if 'containment' not in r:continue
        failures=[x for x in r['containment'] if x.get('outside_adjacent_covered_union_mm3',0)>.1]
        worst=max(r['containment'],key=lambda x:x.get('outside_adjacent_covered_union_mm3',0),default={})
        rows.append(f"{r['height_mm']},{r['wall_mm']},{r.get('relief_mm',0)},{r['axis_count']},{len(failures)},{len(r['containment'])},{len(r['skin_intersections'])},{worst.get('axis','')+' '+worst.get('component','')},{worst.get('outside_adjacent_covered_union_mm3',0)}")
    (OUT/'scale-summary.csv').write_text('\n'.join(rows)+'\n')
    viewer=(Path(__file__).with_name('viewer.html').read_text().replace('serial-scale.glb','serial-scale.glb'))
    (OUT/'index.html').write_text(viewer)
if __name__=='__main__':main()
