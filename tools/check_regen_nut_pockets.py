"""Per-robot exact check of recessed M2 nuts; reuse retained service via subset proof."""
import argparse,json,hashlib,shutil,itertools,traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
from build123d import import_step,export_step,export_stl,Pos,Rot,Compound
from lab.artifact_store import snapshot_file
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import Part,transform_shape,overlap
from tools.check_arm_episode import parts_from_step
from tools.compose_forearm_enclosure import verify_report
from tools.check_neck_collar import ExactQueries
from tools.regen_nut_pocket_geometry import pocket_geometry

def run(name,out):
 out.mkdir(parents=True,exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 cfg,m,parts,cuts,pockets,interface=pocket_geometry();a=m['arches'];h=m['hardware']
 prior=Path('artifacts/regen-mount-v4');r4=json.loads((prior/'report.json').read_text())
 r6=json.loads(Path('artifacts/regen-mount-v6/combined.json').read_text());assert r6['robots'][name]['status']=='pass'
 for p,expected in r4['source_bindings'].items():
  candidate=Path(p) if p!='electronics/regen-mount.yaml' else prior/'source'/p
  assert sha(candidate)==expected,p
 promotion=json.loads(Path('artifacts/regen-mount-v6/promotion.json').read_text());assert sha(cfg['mount_source'])==promotion['new_sha256']
 for p,expected in r6['evidence_files'].items():assert sha(p)==expected,p
 sources=['electronics/regen-nut-pockets.yaml',cfg['mount_source'],'tools/regen_nut_pocket_geometry.py',__file__,'tools/regen_mount_geometry.py','lab/leg_mechanism.py','lab/packaging.py','tools/check_neck_collar.py',prior/'report.json',prior/name/'proposed-front.step','artifacts/regen-mount-v6/combined.json','artifacts/regen-mount-v6/promotion.json']
 r={'status':'partial','execution_status':'running','robot':name,'physical_status':'not_run','strength_status':'not_run','source_bindings':{str(Path(p).absolute().relative_to(Path.cwd())):sha(p) for p in sources}}
 for p in sources:
  relative=Path(p).absolute().relative_to(Path.cwd())
  if relative.parts[0] in ['electronics','tools','lab']:
   dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);snapshot_file(p,dest)
 try:
  oldfront=import_step(prior/name/'proposed-front.step');front=oldfront
  for p in pockets:front=front-p
  assert front.is_valid and len(front.solids())==1
  export_step(front,out/'front.step');export_stl(front,out/'front.stl')
  rb=import_step(out/'front.step');assert rb.is_valid and len(rb.solids())==1 and abs(rb.volume-front.volume)<1e-5
  r['front']={'step_roundtrip':'pass','valid_solids':1,'additional_removed_mm3':oldfront.volume-front.volume,'backing_thickness_mm':cfg['pocket']['nut_bearing_face_x_mm']-a['beam_rear_face_x_mm']}
  r['nominal_thread_engagement_mm']=h['nut_height_mm'];r['screw_projection_mm']=a['screw_seat_x_mm']+cfg['beam_screw_length_mm']-cfg['pocket']['nut_bearing_face_x_mm']-h['nut_height_mm']
  # New obstacle union is contained by the old beam and fastener union.
  # This transfers earlier sampled board paths; it is not continuous motion proof.
  r['service_subset_proof']={'front_added_volume_mm3':max(0,front.volume-overlap(front,oldfront)),'nuts':[],'screws':[]}
  for i,y in enumerate(a['y_centers_mm']):
   nut=parts[f'beam_nut_{i}'];r['service_subset_proof']['nuts'].append(max(0,nut.volume-overlap(nut,oldfront)))
   # Same axis/head and shorter shank: verify actual B-rep containment as well.
   from tools.regen_mount_geometry import geometry
   _,oldparts,_,_=geometry();old=oldparts[f'beam_screw_{i}'];new=parts[f'beam_screw_{i}']
   r['service_subset_proof']['screws'].append(max(0,new.volume-overlap(new,old)))
  assert max([r['service_subset_proof']['front_added_volume_mm3'],*r['service_subset_proof']['screws']])<1e-5
  # Nuts occupy the small annulus left by the original 2.2 mm screw bores.
  # They are not wholly contained by the old stock, so check them directly
  # against every earlier board-removal sample instead of weakening the proof.
  qpath=ExactQueries();nut_targets=[Part(k,'torso',s,'candidate','power',(1,0,0,1)) for k,s in parts.items() if k.startswith('beam_nut_')]
  nut_path_hits=[];waypoints=m['service_paths']['board_waypoints_mm'];sample_count=0
  for start,end in zip(waypoints,waypoints[1:]):
   for t in np.linspace(0,1,int(np.ceil(np.linalg.norm(np.array(end)-start)))+1):
    pos=[float(x+(y-x)*t) for x,y in zip(start,end)];sample_count+=1
    hits=qpath.collide(Pos(*pos)*parts['module'],nut_targets)
    if hits:nut_path_hits.append({'translation_mm':pos,'collisions':hits})
  r['service_subset_proof']['new_nut_board_path']={'samples':sample_count,'collisions':nut_path_hits}
  assert not nut_path_hits
  r['service_subset_proof']['status']='pass'
  r['nut_rotation_probes']=[]
  for i,y in enumerate(a['y_centers_mm']):
   for angle in [0,5,10,-10,30]:
    nut=Pos(0,y,a['bolt_z_mm'])*Rot(angle,0,0)*Pos(0,-y,-a['bolt_z_mm'])*parts[f'beam_nut_{i}']
    r['nut_rotation_probes'].append({'nut':i,'angle_deg':angle,'interference_mm3':overlap(nut,front)})
   assert overlap(parts[f'beam_nut_{i}'],oldfront)>1
  assert all(v['interference_mm3']<.02 if v['angle_deg'] in [0,5] else v['interference_mm3']>.02 for v in r['nut_rotation_probes'])
  folder=Path(f'artifacts/{name}-shin-cover-composed-'+('v3' if name=='pip' else 'v2'));retained=verify_report(folder)
  assert retained['assembly_step_roundtrip']=='pass'
  for p in ['assembly.step','report.json','resolved-config.json']:r['source_bindings'][str(folder/p)]=sha(folder/p)
  robot=make_robot(RobotConfig.model_validate_json((folder/'resolved-config.json').read_text()));frame=robot.forward_kinematics()['torso']
  print(name,'loading whole assembly',flush=True)
  raw=parts_from_step(list(import_step(folder/'assembly.step').children),retained['parts'])
  targets=[replace(p,shape=transform_shape(front,frame)) if p.name=='torso_front' else p for p in raw]
  q=ExactQueries();changed=[k for k in parts if k.startswith('beam_')]
  r['neutral_checks']={k:q.collide(transform_shape(parts[k],frame),targets) for k in changed};r['source_assembly_status']=retained['status'];r['groups_checked']=len(raw)
  r['internal_checks']=[]
  for k in changed:
   for other,s in parts.items():
    if other==k or other in ['vent','capacitor_working_space']:continue
    v=overlap(parts[k],s)
    if v>.02:r['internal_checks'].append({'a':k,'b':other,'overlap_mm3':v})
  r['nut_removal']=[];route=cfg['service']['nut_waypoints_mm']
  service=[p for p in targets if p.name not in ['torso_rear','carrier']]+[Part('tray','torso',transform_shape(parts['tray'],frame),'candidate','power',(1,0,0,1))]
  for i in range(2):
   hits=[];count=0
   for start,end in zip(route,route[1:]):
    for t in np.linspace(0,1,int(np.ceil(np.linalg.norm(np.array(end)-start)/cfg['service']['step_mm']))+1):
     shift=[float(x+(y-x)*t) for x,y in zip(start,end)];count+=1
     found=q.collide(transform_shape(Pos(*shift)*parts[f'beam_nut_{i}'],frame),service)
     if found:hits.append({'translation_mm':shift,'collisions':found})
   r['nut_removal'].append({'nut':i,'status':'fail' if hits else 'pass','route_mm':route,'samples':count,'collisions':hits,'prerequisites':cfg['service']['prerequisites'],'continuous_sweep':'not_run'})
  for k in changed:export_stl(parts[k],out/(k+'.stl'))
  r['neutral_status']='fail' if any(r['neutral_checks'].values()) or r['internal_checks'] else 'pass';r['execution_status']='pass'
  print(name,r['neutral_status'],[x['status'] for x in r['nut_removal']],flush=True)
 except Exception as exc:
  r['execution_status']='fail';r['error']=str(exc);(out/'error.log').write_text(traceback.format_exc());raise
 finally:(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--robot',choices=['pip','scout'],required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.robot,args.output)
