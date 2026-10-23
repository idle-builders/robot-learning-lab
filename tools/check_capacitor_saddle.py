"""Native CAD checks for the capacitor saddle overlay; no physical qualification."""
import argparse,itertools,json,traceback,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
from build123d import import_step,export_step,export_stl,Box,Pos
from lab.artifact_store import snapshot_file,retain_dependency
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import Part,transform_shape,overlap
from tools.check_arm_episode import parts_from_step
from tools.compose_forearm_enclosure import verify_report
from tools.check_neck_collar import ExactQueries
from tools.capacitor_saddle_geometry import geometry

def run(name,out,config):
 out.mkdir(parents=True,exist_ok=False)
 write=lambda n,v:(out/n).write_text(json.dumps(v,indent=2)+'\n')
 sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 r={'status':'partial','execution_status':'running','robot':name,'physical_status':'not_run','strength_thermal_insulation':'not_run','wire_and_cutter_access':'not_run','board_and_tray_removal':'not_run'}
 try:
  c,m,parts,*_=geometry(config)
  sources=[str(config),c['parent'],'electronics/regen-mount.yaml','electronics/regen-clamp.yaml','electronics/motor-capacitor.yaml','tools/capacitor_saddle_geometry.py','tools/check_capacitor_saddle.py','tools/regen_mount_geometry.py','tools/regen_nut_pocket_geometry.py','tools/check_neck_collar.py','lab/packaging.py','lab/leg_mechanism.py']
  r['source_bindings']={f:sha(f) for f in sources}
  for f in sources:snapshot_file(f,out/'source'/f)
  r['cad']={}
  for k in ['tray','strap','strap_head']:
   s=parts[k];assert s.is_valid and len(s.solids())==1,k
   export_step(s,out/(k+'.step'));export_stl(s,out/(k+'.stl'))
   b=import_step(out/(k+'.step'));assert b.is_valid and len(b.solids())==1 and abs(b.volume-s.volume)<1e-5
   r['cad'][k]={'status':'pass','volume_mm3':float(s.volume),'solids':1,'roundtrip_volume_error_mm3':float(abs(b.volume-s.volume))}
  r['added_print_volume_mm3']=float(parts['tray'].volume-parts['original_tray'].volume)
  candidates={k:v for k,v in parts.items() if k not in ['original_tray','capacitor_working_space']}
  q=ExactQueries();r['internal_collisions']=[]
  for a,b in itertools.combinations(candidates,2):
   hits=q.collide(candidates[a],[Part(b,'torso',candidates[b],'candidate','power',(1,0,0,1))])
   if hits:r['internal_collisions'].append({'a':a,'b':b,'hits':hits})
  r['closed_strap_lift_negative_control_mm3']=float(overlap(Pos(0,0,2)*parts['capacitor'],parts['strap']))
  assert r['closed_strap_lift_negative_control_mm3']>.02
  folder=Path(f'artifacts/{name}-shin-cover-composed-'+('v3' if name=='pip' else 'v2'))
  previous=Path('artifacts/regen-nut-pockets-v3')/name
  retained=verify_report(folder);assert retained['assembly_step_roundtrip']=='pass'
  r['source_assembly_status']=retained['status']
  for f in [folder/'assembly.step',folder/'report.json',folder/'resolved-config.json',previous/'front.step',previous/'report.json']:
   r['source_bindings'][str(f)]=sha(f)
  robot=make_robot(RobotConfig.model_validate_json((folder/'resolved-config.json').read_text()));frame=robot.forward_kinematics()['torso']
  print(name,'loading assembly',flush=True)
  raw=parts_from_step(list(import_step(folder/'assembly.step').children),retained['parts'])
  front=import_step(previous/'front.step')
  targets=[replace(p,shape=transform_shape(front,frame)) if p.name=='torso_front' else p for p in raw]
  targets.append(Part('battery_reference_plus_1mm','torso',transform_shape(Pos(8,0,39)*Box(61.04,87.12,24.39),frame),'allocation','power',(1,.5,0,1)))
  r['groups_checked']=len(raw);r['neutral_checks']={}
  for k in ['tray','strap','strap_head','capacitor','vent']:
   print(name,'checking',k,flush=True)
   r['neutral_checks'][k]=q.collide(transform_shape(parts[k],frame),targets)
  r['neutral_status']='fail' if r['internal_collisions'] or any(r['neutral_checks'].values()) else 'pass'
  write('progress.json',r)
  service=[p for p in targets if p.name not in ['torso_rear','carrier','battery_reference_plus_1mm']]
  service += [Part(k,'torso',transform_shape(s,frame),'candidate','power',(1,0,0,1)) for k,s in candidates.items() if k not in ['capacitor','vent','strap','strap_head']]
  hits=[];count=0
  for start,end in zip(c['service']['capacitor_waypoints_mm'],c['service']['capacitor_waypoints_mm'][1:]):
   for t in np.linspace(0,1,int(np.ceil(np.linalg.norm(np.array(end)-start)/c['service']['step_mm']))+1):
    shift=(np.array(start)+(np.array(end)-start)*t).tolist();count+=1
    found=q.collide(transform_shape(Pos(*shift)*parts['capacitor'],frame),service)
    if found:hits.append({'shift_mm':shift,'collisions':found})
  r['capacitor_removal']={'status':'fail' if hits else 'pass','samples':count,'collisions':hits,'waypoints_mm':c['service']['capacitor_waypoints_mm'],'prerequisites':c['service']['prerequisites'],'continuous_sweep':'not_run'}
  r['board_and_tray_removal']={}
  service_base=[p for p in targets if p.name not in ['torso_rear','carrier','battery_reference_plus_1mm']]
  for key in ['module','tray']:
   obstacles=list(service_base)
   if key=='module':
    obstacles += [Part(k,'torso',transform_shape(s,frame),'candidate','power',(1,0,0,1)) for k,s in candidates.items() if k=='tray' or k.startswith('beam_')]
   route=m['service_paths']['board_waypoints_mm' if key=='module' else 'tray_waypoints_mm'];hits=[];count=0
   for start,end in zip(route,route[1:]):
    for t in np.linspace(0,1,int(np.ceil(np.linalg.norm(np.array(end)-start)/m['service_paths']['sample_step_mm']))+1):
     shift=(np.array(start)+(np.array(end)-start)*t).tolist();count+=1
     found=q.collide(transform_shape(Pos(*shift)*parts[key],frame),obstacles)
     if found:hits.append({'shift_mm':shift,'collisions':found})
   r['board_and_tray_removal'][key]={'status':'fail' if hits else 'pass','samples':count,'collisions':hits,'waypoints_mm':route,'continuous_sweep':'not_run','prerequisites':'Rear cover, carrier, battery, capacitor, strap and board screws removed; tray removal additionally requires board and beam hardware removed.'}
  print(name,'board/tray',[(k,v['status']) for k,v in r['board_and_tray_removal'].items()],flush=True)
  r['dependency_manifests']=[retain_dependency(folder),retain_dependency(previous)]
  r['execution_status']='pass'
  print(name,r['neutral_status'],r['neutral_checks'],r['capacitor_removal']['status'],flush=True)
 except Exception as e:
  r['execution_status']='fail';r['error']=str(e);(out/'error.log').write_text(traceback.format_exc());raise
 finally:write('report.json',r)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--robot',choices=['pip','scout'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('electronics/capacitor-saddle.yaml'));a=p.parse_args();run(a.robot,a.output,a.config)
