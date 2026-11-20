"""Removable torso IMU proposal on a drilled derivative of the exact V5 spine."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import numpy as np
import trimesh
from build123d import Axis, Box, Cylinder, Compound, Location, RegularPolygon, extrude, export_step, export_stl, import_step
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopTools import TopTools_ListOfShape
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.gp import gp_Trsf

ROOT=Path(__file__).resolve().parents[5]
OUT=ROOT/'artifacts/cove50-sensor-integration/v2'
CONFIG=ROOT/'robot/sensor-integration/cad/imu-mount-v2.json'

def read(p):return json.loads(p.read_text())
def rel(p):return str(p.relative_to(ROOT))
def sha(p):
    with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def bounds(s):
    b=s.bounding_box();return np.array([list(b.min),list(b.max)])
def cyl_y(r,h,center):return Cylinder(r,h).rotate(Axis.X,90).translate(center)
def ring(r,inner,h,center):return cyl_y(r,h,center)-cyl_y(inner,h+2,center)
def common(a,b):
    args,tools=TopTools_ListOfShape(),TopTools_ListOfShape();args.Append(a.wrapped);tools.Append(b.wrapped)
    op=BRepAlgoAPI_Common();op.SetArguments(args);op.SetTools(tools);op.SetNonDestructive(True);op.Build()
    if not op.IsDone():raise RuntimeError('Native Common did not finish')
    props=GProp_GProps();BRepGProp.VolumeProperties_s(op.Shape(),props);return float(props.Mass())
def intersects_bounds(a,b):return bool(np.all(a[0]<=b[1]+1e-6)and np.all(b[0]<=a[1]+1e-6))

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'parts').mkdir(exist_ok=True);c=read(CONFIG)
    manifest=read(ROOT/c['baseline_manifest']);upper=read(ROOT/c['upper_manifest'])
    frame_row=next(p for p in upper['parts']if p['name']==c['frame_source_part']);frame_path=ROOT/frame_row['native_step']
    if sha(frame_path)!=frame_row['native_step_sha256']:raise ValueError('V5 frame drift')
    source=import_step(frame_path);spine=source
    for x,y,z in c['placement']['mount_holes_world_mm']:spine=spine-cyl_y(1.35,8,(x,34,z))
    board_raw=import_step(ROOT/c['board_step'])
    placed=[]
    for component in board_raw.children:
        # Transform each component explicitly: a moved assembly root does not
        # rewrite the source tree's child shapes in build123d.
        q=Compound(children=list(component.solids()),label=component.label).rotate(Axis.X,-90).translate(c['placement']['translation_mm']);placed.append(q)
    board=Compound(children=placed)
    context=trimesh.load(ROOT/'artifacts/cove50-sensor-integration/v1/assembly.glb',force='scene',process=False)
    rows=[];shapes={};scene=trimesh.Scene()
    def add(name,s,role,material,color,physical=True,mass_g=None,basis=None):
        s=Compound(children=list(s.solids()));s.label=name;shapes[name]=s;path=OUT/'parts'/f'{name}.step';meshpath=OUT/'parts'/f'{name}.stl';export_step(s,path);export_stl(s,meshpath,tolerance=.06,angular_tolerance=.2)
        mesh=trimesh.load(meshpath,force='mesh',process=False);mesh.visual.face_colors=color;mesh.apply_scale(.001);scene.add_geometry(mesh,node_name=name,geom_name=name)
        row=dict(name=name,owning_link='torso',role=role,material=material,native_step=rel(path),native_step_sha256=sha(path),mesh=rel(meshpath),mesh_sha256=sha(meshpath),mesh_units='mm',bounds_mm=bounds(s).tolist(),native_volume_mm3=s.volume,native_validity=[bool(q.is_valid)for q in s.solids()],physical_part=physical,inspection_reference=not physical,color=color,physical_part_id=name,mass_kg=mass_g/1000 if mass_g is not None else None,mass_basis=basis)
        rows.append(row)
    add('sensor_spine_two_hole_proposal',spine,'frame','Inherited bent aluminium V5',[155,168,181,255],basis='Replacement source frame, not a second spine purchase; inherited full frame mass not recomputed as a new sensor')
    # Retain complete populated source model and component identities, including pins below substrate.
    for i,part in enumerate(board.children):
        add('imu_board_'+str(i)+'_'+part.label,part,'sensor'if i==0 else'board_component','Retained component/PCB proxy',[45,130,95,255]if 'PCB'in part.label else[70,75,80,255],basis='Board/component mass unweighed; source geometry is a library model')
    for i,(x,y,z)in enumerate(c['placement']['mount_holes_world_mm']):
        add(f'imu_spacer_{i}',ring(2.5,1.35,3,(x,37.5,z)),'spacer','6061 aluminium planning',[70,155,210,255],mass_g=math.pi*(2.5**2-1.35**2)*3*.0027,basis='Nominal volume times2700kg/m3; excludes finish/tolerance')
        add(f'imu_washer_{i}',ring(3,1.35,.5,(x,40.85,z)),'washer','Zinc plated steel',[175,177,180,255],mass_g=math.pi*(3**2-1.35**2)*.5*.00785,basis='Nominal volume times7850kg/m3; not catalog mass')
        add(f'imu_M2p5x12_{i}',cyl_y(1.25,12,(x,35.1,z))+cyl_y(2.25,2.5,(x,42.35,z)),'fastener','A2 stainless steel',[181,185,190,255],mass_g=.58,basis='Accu catalog0.58g per screw; no helical threads or socket cavity modeled')
        nut=extrude(RegularPolygon(5/math.sqrt(3),6),amount=2).rotate(Axis.X,90).translate((x,32,z))-cyl_y(1.25,4,(x,31,z))
        add(f'imu_M2p5_nut_{i}',nut,'nut','A2 stainless steel, black',[100,107,115,255],mass_g=.28,basis='Accu catalog0.28g per nut; nominal smooth bore')
    # Explicit access reservation, excluded from installed BOM/mass and from source-solid fit claims.
    header=next(p for p in board.children if 'JST_'in p.label);b=bounds(header);keep=Box(b[1,0]-b[0,0]+1,10,b[1,2]-b[0,2]+1).translate(((b[0,0]+b[1,0])/2,b[1,1]+5,(b[0,2]+b[1,2])/2))
    add('imu_connector_service_reservation',keep,'service_reservation','Reservation only',[230,160,40,100],physical=False,basis='Conservative header footprint plus10mm, not actual PHR-5 housing')
    # Native package-only export; context is added separately without a duplicate frame.
    export_step(Compound(children=[s for n,s in shapes.items()if n!='imu_connector_service_reservation']),OUT/'mount-package.step')
    scene.export(OUT/'mount-package.glb')
    write(OUT/'parts.json',dict(schema='cove50.imu-mount-package.v1',version='v2',root_link='torso',parts=rows,joints=[],scope='Exact source-board placement plus nominal fastening on a proposed drilled V5 spine. No source accepted assembly modified.'))
    sources=[dict(path=rel(p),sha256=sha(p))for p in[CONFIG,Path(__file__),frame_path,ROOT/c['board_step'],ROOT/c['board_contract'],ROOT/c['upper_manifest'],ROOT/c['baseline_manifest']]]
    write(OUT/'source-seal.json',dict(sources=sources))
    # Inspect nearby source-native occupancy. AABB is only a broadphase; invalid/missing operands stay unknown.
    relevant={};unknown=[]
    bb=np.array([[-30,0,285],[5,90,350]],float)
    for p in manifest['parts']:
        if p.get('source_name')==c['frame_source_part']:continue
        b=p.get('bounds_mm')
        if b is None:
            tr,key=context.graph[p['name']];mesh=context.geometry[key].copy();mesh.apply_transform(tr);b=(mesh.bounds*1000).tolist()
        if not intersects_bounds(np.array(b),bb):continue
        path=p.get('native_step')
        if path and (ROOT/path).is_file():
            if p.get('native_step_sha256')and sha(ROOT/path)!=p['native_step_sha256']:raise ValueError('Source drift '+p['name'])
            relevant[p['name']]=import_step(ROOT/path)
        else:unknown.append(dict(name=p['name'],reason='No individual native STEP; reference mesh or unresolved source',inspection_reference=p.get('inspection_reference')))
    trace=[]
    def query(label,s,exclude=()):
        for name,t in relevant.items():
            if name in exclude:continue
            a,b=bounds(s),bounds(t);gap=float(np.max(np.maximum(a[0]-b[1],b[0]-a[1])))
            if gap>1e-6:
                trace.append(dict(case=label,other=name,method='disjoint_native_bounds',aabb_separation_mm=gap,native_overlap_mm3=0.0));continue
            valid=all(q.is_valid for q in s.solids())and all(q.is_valid for q in t.solids())
            trace.append(dict(case=label,other=name,method='native_common',aabb_separation_mm=gap,native_overlap_mm3=common(s,t)if valid else None,operand_validity=bool(valid)))
    for n,s in shapes.items():
        if n=='sensor_spine_two_hole_proposal':continue
        query('installed/'+n,s)
    for i,(x,y,z)in enumerate(c['placement']['mount_holes_world_mm']):
        query(f'rear_driver_{i}',cyl_y(2,35,(x,61.1,z)))
        query(f'front_nut_tool_{i}',cyl_y(3.5,25,(x,17.5,z)))
    for travel in np.linspace(0,25,11):query(f'board_withdraw_{travel:g}mm',board.translate((0,travel,0)))
    # Reuse the exact completed V5 representative motion configurations.
    audit_path=ROOT/'artifacts/cove50-engineering-competition/upper-body-20261023/lead/v5/audit-config.json'
    poses=read(audit_path)['poses'];sources.append(dict(path=rel(audit_path),sha256=sha(audit_path)))
    spec=importlib.util.spec_from_file_location('sensor_screen',ROOT/'robot/electronics/sensing/studies/screen.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    new_subjects={n:s for n,s in shapes.items()if n!='sensor_spine_two_hole_proposal'}
    loaded={};pose_stats=[]
    for pose in poses:
        tf=module.transforms(manifest['joints'],pose['commands_deg']);broad=0;disjoint=0
        for p in manifest['parts']:
            if p['source_half']!='upper'or p['owning_link']=='torso':continue
            tr,key=context.graph[p['name']];mesh=context.geometry[key]
            b=np.array(p.get('bounds_mm')or (trimesh.transform_points(mesh.bounds,tr)*1000).tolist())
            corners=np.array([[x,y,z]for x in b[:,0]for y in b[:,1]for z in b[:,2]])
            t=tf[p['owning_link']];points=corners@t[:3,:3].T+t[:3,3];bb=np.array([points.min(0),points.max(0)])
            for name,s in new_subjects.items():
                if not intersects_bounds(bounds(s),bb):disjoint+=1;continue
                broad+=1;path=p.get('native_step')
                row=dict(case='motion/'+pose['name']+'/'+name,other=p['name'],commands_deg=pose['commands_deg'],method='native_common',native_overlap_mm3=None)
                if path and (ROOT/path).is_file():
                    if path not in loaded:
                        if p.get('native_step_sha256')and sha(ROOT/path)!=p['native_step_sha256']:raise ValueError('Moving source drift')
                        loaded[path]=import_step(ROOT/path)
                    source_shape=loaded[path];g=gp_Trsf();g.SetValues(*map(float,t[:3,:].reshape(-1)));moved=source_shape.moved(Location(g))
                    row['operand_validity']=all(q.is_valid for q in moved.solids())
                    if row['operand_validity']:row['native_overlap_mm3']=common(s,moved)
                else:row.update(method='reference_or_missing_native',limitation='Mesh-reference source, not native material evidence')
                trace.append(row)
        pose_stats.append(dict(pose=pose['name'],commands_deg=pose['commands_deg'],bounds_disjoint_pairs=disjoint,broadphase_queries=broad))
    write(OUT/'motion-configurations.json',dict(source=rel(audit_path),source_sha256=sha(audit_path),configurations=pose_stats))
    (OUT/'mount-fit-trace.jsonl').write_text(''.join(json.dumps(t)+'\n'for t in trace))
    bearing=[]
    for x,y,z in c['placement']['mount_holes_world_mm']:
        annulus=ring(2.5,1.35,.01,(x,35.995,z));bearing.append(dict(center_world_mm=[x,36,z],available_spine_annulus_area_mm2=common(spine,annulus)/.01,nominal_spacer_annulus_area_mm2=math.pi*(2.5**2-1.35**2)))
    distances=[]
    for name,t in relevant.items():
        if all(q.is_valid for q in t.solids()):
            query_distance=BRepExtrema_DistShapeShape(board.wrapped,t.wrapped);query_distance.Perform()
            distances.append(dict(other=name,distance_mm=float(query_distance.Value())if query_distance.IsDone()else None))
    distances.sort(key=lambda r:r['distance_mm']if r['distance_mm']is not None else math.inf)
    report=dict(motion_configurations=len(poses),motion_pairs=sum(p['bounds_disjoint_pairs']+p['broadphase_queries']for p in pose_stats),motion_native_candidates=sum(p['broadphase_queries']for p in pose_stats),nearest_board_native_surfaces=distances,method='Native STEP board/component placement and nominal purchased/produced fasteners. Non-destructive OCP common after native-AABB broadphase. Source reference skins without native parts stay explicitly unresolved.',sources=sources,board_bounds_world_mm=bounds(board).tolist(),new_mount_holes=c['placement']['mount_holes_world_mm'],removed_frame_volume_mm3=source.volume-spine.volume,ideal_two_hole_volume_mm3=2*math.pi*1.35**2*4,bearing_annuli=bearing,
        source_board_thickness_mm=1.51,nominal_stack_board_thickness_mm=1.6,proxy_thickness_discrepancy_mm=.09,nominal_thread_beyond_nut_mm=.9,checks=len(trace),nearby_native_sources=len(relevant),unknown_context=unknown,native_material_witnesses=[r for r in trace if r['native_overlap_mm3']and r['native_overlap_mm3']>1e-7],full_trace=rel(OUT/'mount-fit-trace.jsonl'),
        board_frame= c['frame_contract'],sensor_package_center_world_mm=c['placement']['sensor_package_center_world_mm'],
        grades=dict(source_package=dict(grade=2.5,confidence='medium',trend='retained real PCB layout, generic component models'),mount_geometry=dict(grade=2.5,confidence='medium',trend='new proposal'),service=dict(grade=1.5,confidence='low',trend='sampled reservations only'),physical_registration=dict(grade=.5,confidence='high',trend='unmeasured die axes/calibration')),
        limitations=['Spine derivative is a proposal, not accepted V5 or final one-piece center-body geometry.','Exact source board uses generic component models and1.51mm substrate versus1.6mm contract. Actual PHR-5/lead, crimp strain relief, driver/spanner shape and fingers unresolved.','V5 finite representative upper motions are screened against new mount; no continuous, loaded, walking or real harness motion qualification.','No board stress, preload, vibration, physical mounting/calibration or installed timing measured.'])
    write(OUT/'report.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
