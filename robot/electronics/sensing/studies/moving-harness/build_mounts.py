"""Native head/torso strain-relief proposal and explicit attachment changes."""
from pathlib import Path
import hashlib,importlib.util,json,math
import numpy as np
import trimesh
from build123d import Axis,Box,Cylinder,Compound,RegularPolygon,extrude,import_step,export_step,export_stl
ROOT=Path(__file__).resolve().parents[5];OUT=ROOT/'artifacts/cove50-sensor-integration/v7'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def rel(p):return str(p.relative_to(ROOT))
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
spec=importlib.util.spec_from_file_location('native_helpers',ROOT/'robot/electronics/sensing/studies/imu-mount/build.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
def box(x0,x1,y0,y1,z0,z1):return Box(x1-x0,y1-y0,z1-z0).translate(((x0+x1)/2,(y0+y1)/2,(z0+z1)/2))
def cy(r,y0,y1,x,z):return Cylinder(r,y1-y0).rotate(Axis.X,90).translate((x,(y0+y1)/2,z))
def cx(r,x0,x1,y,z):return Cylinder(r,x1-x0).rotate(Axis.Y,90).translate(((x0+x1)/2,y,z))
def nut_y(x,top,z):return extrude(RegularPolygon(5/math.sqrt(3),6),amount=2).rotate(Axis.X,90).translate((x,top,z))-cy(1.25,top-3,top+1,x,z)
def nut_x(front,y,z):return extrude(RegularPolygon(5/math.sqrt(3),6),amount=2).rotate(Axis.Y,90).translate((front,y,z))-cx(1.25,front-1,front+3,y,z)
def local_common(a,b):
 from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
 from OCP.TopTools import TopTools_ListOfShape
 bounds=native.bounds(a);room=Box(*(bounds[1]-bounds[0])).translate((bounds[0]+bounds[1])/2);aa,bb=TopTools_ListOfShape(),TopTools_ListOfShape();aa.Append(b.wrapped);bb.Append(room.wrapped);op=BRepAlgoAPI_Common();op.SetArguments(aa);op.SetTools(bb);op.SetNonDestructive(True);op.Build()
 if not op.IsDone():raise RuntimeError('Native clipped region incomplete')
 region=Compound.cast(op.Shape())
 return native.common(a,region)if region.solids()else 0.
def main():
 (OUT/'mount-parts').mkdir(exist_ok=True);c=read(ROOT/'robot/sensor-integration/cad/moving-head-harness-v7.json');fixed_x=c['torso_anchor']['origin_world_mm'][0];fixed_dx=fixed_x-15.55;m=read(ROOT/'artifacts/cove50-engineering-competition/upper-body-20261023/lead/v5/parts.json');index={p['name']:p for p in m['parts']};source_head=ROOT/index['integrated_engineered_head']['native_step'];source_spine=ROOT/'artifacts/cove50-sensor-integration/v2/parts/sensor_spine_two_hole_proposal.step';head=import_step(source_head);spine=import_step(source_spine);old_head=head;old_spine=spine;rows=[];shapes={};scene=trimesh.Scene()
 # Oval rear port:10mm X width and18mm Z height, through only the rear head
 # region. No optical/window or camera mounting datum is translated.
 cutter=box(10.55,20.55,20,70,464.25,472.25)+cy(5,20,70,15.55,464.25)+cy(5,20,70,15.55,472.25);head=head-cutter
 port_only_volume=head.volume
 # Two boss recesses accept the new2mm strap without moving the camera datum.
 for x0,x1 in [(-41.7,-31.3),(31.3,41.7)]:head=head-box(x0,x1,12.225,14.425,440.3,448.7)
 for x in [14,20]:spine=spine-cy(1.35,29,40,x,371)
 spine=spine-cy(3.2,29,40,fixed_x,356)
 def add(name,s,link,role,material,color,mass_g=None,identity=None):
  s=Compound(children=list(s.solids()),label=name);shapes[name]=s;path=OUT/'mount-parts'/f'{name}.step';stl=path.with_suffix('.stl');export_step(s,path);export_stl(s,stl,tolerance=.06,angular_tolerance=.15);mesh=trimesh.load(stl,force='mesh',process=False);mesh.apply_scale(.001);mesh.visual.face_colors=color;scene.add_geometry(mesh,node_name=name,geom_name=name)
  rows.append(dict(name=name,source_name=name,owning_link=link,role=role,material=material,native_step=rel(path),native_step_sha256=sha(path),mesh=rel(stl),mesh_sha256=sha(stl),mesh_units='mm',bounds_mm=native.bounds(s).tolist(),native_volume_mm3=s.volume,native_validity=[bool(q.is_valid)for q in s.solids()],physical_part=True,physical_part_id=name,inspection_reference=False,mass_kg=mass_g/1000 if mass_g is not None else None,part_identity=identity,qualification='Proposed native geometry; no manufactured grip, retention or physical qualification'))
 add('harness_rear_port_head',head,'head','proposed_source_derivative','Inherited source PA12',[232,223,201,255]);add('harness_four_hole_spine',spine,'torso','proposed_source_derivative','Inherited4mm aluminium plus existing V2 IMU holes',[155,168,181,255])
 # Flat2mm plate seats under the two source camera screw heads. New M4x10
 # screws shift the head seat by2mm and retain the original tipY4.225.
 bar=box(-41.5,41.5,12.225,14.225,440.5,448.5)+box(25,31,12.225,14.225,448,478.5)
 for x in [-37.5,37.5]:bar=bar-cy(2.15,11,16,x,444.5)
 for z in [462.25,474.25]:bar=bar-cy(1.35,11,16,28,z)
 add('harness_camera_mount_bar',bar,'head','produced_bracket','5052-H32 aluminium2mm planning',[90,160,195,255],mass_g=bar.volume*.00268)
 # Side-split clamps open radially around a pre-terminated lead. Nominal bore
 #4.5mm against source4.6mm jacket is intentional compression, not clearance.
 def split_clamp(y0,y1,zc):
  left=box(8.55,15.55,y0,y1,zc-8,zc+8);right=box(15.55,22.55,y0,y1,zc-8,zc+8);bore=cy(2.25,y0-1,y1+1,15.55,zc);left=left-bore;right=right-bore
  for z in [zc-6,zc+6]:
   hole=cx(1.35,5,26,(y0+y1)/2,z);left=left-hole;right=right-hole
  return left,right
 left,right=split_clamp(24,32,468.25)
 bridge=box(8.55,32,20,24,460.25,464.5)+box(24,32,14.225,24,459.25,477.25);left=left+bridge
 for z in [462.25,474.25]:left=left-cy(1.35,12,28,28,z)
 add('harness_head_clamp_base',left,'head','produced_clamp','PA12 planning',[236,220,175,255]);add('harness_head_clamp_cap',right,'head','produced_clamp','PA12 planning',[210,190,150,255])
 left,right=split_clamp(39,47,356);left=left.translate((fixed_dx,0,0));right=right.translate((fixed_dx,0,0));plate=box(9.55,31.55,36,39,341,375)-cy(3.2,35,39.0001,fixed_x,356);left=left+plate
 for x in [14,20]:
  left=left-cy(1.35,34,41,x,371);left=left-cy(2.5,39,44,x,371)
 add('harness_fixed_clamp_base',left,'torso','produced_clamp','PA12 planning',[236,220,175,255]);add('harness_fixed_clamp_cap',right,'torso','produced_clamp','PA12 planning',[210,190,150,255])
 for i,x in enumerate([-37.5,37.5]):
  screw=cy(2,4.225,14.225,x,444.5)+cy(3.5,14.225,18.225,x,444.5);add(f'harness_camera_M4x10_{i}',screw,'head','fastener','A2 stainless nominal',[175,180,185,255],identity='M4x10 DIN912/ISO4762, nominal7x4mm head; no threads/socket modeled')
 for link,ym,zc in [('head',28,468.25),('fixed',43,356)]:
  for i,z in enumerate([zc-6,zc+6]):
   dx=fixed_dx if link=='fixed'else 0
   screw=cx(1.25,2.55+dx,22.55+dx,ym,z)+cx(2.25,22.55+dx,25.05+dx,ym,z);add(f'harness_{link}_cap_M2p5x20_{i}',screw,'head'if link=='head'else'torso','fastener','A2 stainless nominal',[175,180,185,255],identity='M2.5x20 cap screw,4.5x2.5head; smooth shank proxy');add(f'harness_{link}_cap_nut_{i}',nut_x(6.55+dx,ym,z),'head'if link=='head'else'torso','nut','A2 stainless nominal',[115,120,125,255],mass_g=.28,identity='Accu HPN-M2.5-A2-BL,AF5mm,t2mm; inherited V2 catalog source')
 for i,z in enumerate([462.25,474.25]):
  spacer=cy(2.5,24,26,28,z)-cy(1.35,23,27,28,z);add(f'harness_head_mount_spacer_{i}',spacer,'head','produced_spacer','6061 aluminium planning',[75,150,200,255],mass_g=spacer.volume*.0027)
  screw=cy(1.25,10,26,28,z)+cy(2.25,26,28.5,28,z);add(f'harness_head_mount_M2p5x16_{i}',screw,'head','fastener','A2 stainless nominal',[175,180,185,255],identity='M2.5x16 cap screw, smooth shank proxy');add(f'harness_head_mount_nut_{i}',nut_y(28,12.225,z),'head','nut','A2 stainless nominal',[115,120,125,255],mass_g=.28,identity='Accu HPN-M2.5-A2-BL,AF5mm,t2mm')
 for i,x in enumerate([14,20]):
  screw=cy(1.25,29,39,x,371)+cy(2.25,39,41.5,x,371);add(f'harness_fixed_mount_M2p5x10_{i}',screw,'torso','fastener','A2 stainless nominal',[175,180,185,255],identity='M2.5x10 cap screw, smooth shank proxy');add(f'harness_fixed_mount_nut_{i}',nut_y(x,32,371),'torso','nut','A2 stainless nominal',[115,120,125,255],mass_g=.28,identity='Accu HPN-M2.5-A2-BL,AF5mm,t2mm')
 export_step(Compound(children=list(shapes.values())),OUT/'mounts.step');scene.export(OUT/'mounts.glb');write(OUT/'mount-parts.json',dict(version='v7',parts=rows,root_link='torso',joints=[],scope='Strain-relief mount and rear-port proposal only. Exact body-source derivatives retained; no global selection or physical grip qualification.'))
 # Neutral added-part/source material observations. Intended mating/contact
 # interfaces are named; overlap volumes remain visible and uncombined.
 targets=[];context_path=ROOT/'artifacts/cove50-sensor-integration/v6/assembly-parts.json';input_sources=[context_path];upper_context=read(context_path)
 for p in upper_context['parts']:
  if p.get('inspection_reference')or p.get('owning_link')not in ['head','neck_yaw','torso']or not p.get('native_step'):continue
  if p['name']in ['upper__integrated_engineered_head','sensor_spine_two_hole_proposal','upper__camera_M4x8_0','upper__camera_M4x8_1']:continue
  b=np.array(p.get('bounds_mm',[[0,0,0],[0,0,0]]))
  if b[1,2]<330:continue
  path=ROOT/p['native_step'];input_sources.append(path)
  if p.get('native_step_sha256')and sha(path)!=p['native_step_sha256']:raise ValueError('Native target source drift '+p['name'])
  s=import_step(path);targets.append((p['name'],s));
 subjects=[(n,s)for n,s in shapes.items()if n not in ['harness_rear_port_head','harness_four_hole_spine']];trace=[]
 for i,(name,s)in enumerate(subjects):
  for other,t in [('harness_rear_port_head',head),('harness_four_hole_spine',spine)]+targets+subjects[i+1:]:
   a,b=native.bounds(s),native.bounds(t);gap=float(np.max(np.maximum(a[0]-b[1],b[0]-a[1])));valid=all(q.is_valid for q in s.solids())and all(q.is_valid for q in t.solids());value=0. if gap>1e-6 else local_common(s,t)if valid else None
   trace.append(dict(part=name,other=other,method='disjoint_native_bounds'if gap>1e-6 else'native_common'if valid else'source_invalid_unknown',overlap_mm3=value))
  print('Mount material subject complete: '+name,flush=True)
 (OUT/'mount-fit-trace.jsonl').write_text(''.join(json.dumps(q)+'\n'for q in trace));report=dict(parts=len(rows),head_port_removed_volume_mm3=old_head.volume-port_only_volume,camera_strap_recess_removed_volume_mm3=port_only_volume-head.volume,head_source_solids=len(old_head.solids()),head_proposal_solids=len(head.solids()),added_spine_removed_volume_mm3=old_spine.volume-spine.volume,new_spine_holes=[dict(center_mm=[x,36,371],diameter_mm=2.7)for x in [14,20]]+[dict(center_mm=[fixed_x,36,356],diameter_mm=6.4)],head_M4_tip_Y_mm=4.225,head_M4_new_seat_Y_mm=14.225,source_M4_seat_Y_mm=12.225,nominal_grip_diameter_interference_mm=.1,neutral_observations=len(trace),material_observations=[q for q in trace if q['overlap_mm3']is not None and q['overlap_mm3']>1e-6],unknown_observations=[q for q in trace if q['overlap_mm3']is None],limitations=['Proposed port/fastener/mount geometry; no physical print, plate/tool, preload, clamp friction, jacket compression/abrasion or durability measured.','New port/head mount and spine holes change source load paths. No inherited FEM/physical release or automatic clearance verdict.','Cap bore is deliberately0.1mm smaller than source jacket diameter; actual grip and stiffness are unmeasured.','Source parts/fasteners may need local revisions after native observations and exact cable/host/service reconciliation.'])
 write(OUT/'mount-report.json',report);write(OUT/'mount-source-seal.json',dict(files=[dict(path=rel(p),sha256=sha(p),bytes=p.stat().st_size)for p in dict.fromkeys([Path(__file__),source_head,source_spine,ROOT/'robot/sensor-integration/cad/moving-head-harness-v7.json']+input_sources)]));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
