"""Bounded floor-pickup posture/load preflight, not a learned cleanup rollout."""
import argparse,copy,json,sys,uuid,subprocess
from pathlib import Path
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
from scipy.optimize import differential_evolution
from lab.mujoco_learning import BUNDLE,digest
from lab.artifact_store import snapshot_file,plan_migration,apply_migration,verify_manifest


def make_models(out):
 # Kinematics generated in the existing CAD environment; numerical study stays
 # in the existing learning environment. No duplicate CAD or meshes are built.
 code="from pathlib import Path; from lab.config import load_config; from lab.robot import make_robot; from lab.export import mjcf; r=make_robot(load_config(Path('variants/pip-floor-feet-v0.yaml'))); mjcf(r,Path(__import__('sys').argv[1]))"
 subprocess.run([str(ROOT/'.venv/bin/python'),'-c',code,str(out/'feet.xml')],cwd=ROOT,check=True)
 old=ET.parse(BUNDLE/'model/robot.xml').getroot()
 for mesh in old.findall('./asset/mesh'):mesh.set('file',str((BUNDLE/'model'/mesh.get('file')).absolute()))
 ET.ElementTree(old).write(out/'wheels.xml')
 foot=ET.parse(out/'feet.xml').getroot();common=[]
 old_bodies={b.get('name'):b for b in old.findall('.//body')}
 for body in foot.findall('.//body'):
  source=old_bodies.get(body.get('name'))
  if source is None:continue
  # Reuse planning mass/tensor and actual CAD on unchanged shared links. New
  # foot/hip/ankle links retain declared generic allocations and envelope shapes.
  if body.find('inertial') is not None:body.remove(body.find('inertial'))
  body.insert(0,copy.deepcopy(source.find('inertial')))
  visuals=[g for g in source.findall('geom') if g.get('type')=='mesh']
  if visuals:
   for geom in body.findall('geom'):
    rgba=geom.get('rgba','1 1 1 1').split();rgba[-1]='0';geom.set('rgba',' '.join(rgba))
   for geom in visuals:body.append(copy.deepcopy(geom))
   common.append(body.get('name'))
 for mesh in old.findall('./asset/mesh'):foot.find('asset').append(copy.deepcopy(mesh))
 ET.ElementTree(foot).write(out/'feet.xml')
 return common


class Posture:
 def __init__(self,path,feet=False):
  self.model=mujoco.MjModel.from_xml_path(str(path));self.data=mujoco.MjData(self.model);self.feet=feet
  self.ids={self.model.joint(i).name:i for i in range(1,self.model.njnt)}
  self.torso=self.model.body('torso').id
 def set(self,name,value):self.data.qpos[self.model.jnt_qposadr[self.ids[name]]]=value
 def pose(self,x):
  knee,hip,roll,arm,elbow=x;m=self.model;d=self.data;mujoco.mj_resetData(m,d)
  for side in ['left','right']:
   self.set(side+'_thigh_joint',hip);self.set(side+'_shin_joint',knee)
   if self.feet:self.set(side+'_ankle_pitch_joint',-hip-knee)
  for name,value in [('left_shoulder_roll_joint',roll),('left_upper_arm_joint',arm),('left_forearm_joint',elbow),('left_gripper_joint',-.45),('right_shoulder_roll_joint',-.3),('right_forearm_joint',-.4)]:self.set(name,value)
  mujoco.mj_forward(m,d)
  support='left_foot' if self.feet else 'left_wheel'
  support_geom=m.geom('left_foot_0' if self.feet else 'left_wheel_0').id
  height=float(m.geom_size[support_geom,2]-m.geom_pos[support_geom,2]) if self.feet else float(m.geom_size[support_geom,0])
  d.qpos[2]+=height-d.body(support).xpos[2];mujoco.mj_forward(m,d)
  # Grasp reference defined by the shared forearm/finger recipe, not a new IK tree.
  point=d.body('left_forearm').xpos+d.body('left_forearm').xmat.reshape(3,3)@np.array([0,.002,m.geom_pos[m.geom('left_forearm_1').id,2]])
  com=d.subtree_com[self.torso];mass=m.body_subtreemass[self.torso]
  xs=[];ys=[];bodies=[]
  for side,sign in [('left',1),('right',-1)]:
   body=m.body(side+('_foot' if self.feet else '_wheel')).id;bodies.append(body)
   xs.append(d.xpos[body,0]);ys.append(float(d.geom(side+('_foot_0' if self.feet else '_wheel_0')).xpos[1]))
  if self.feet:
   geom=m.geom('left_foot_0').id;offset=m.geom_pos[geom,0];half=m.geom_size[geom,0]
   lo=max(xs)+offset-half;hi=min(xs)+offset+half;cop=np.clip(com[0],lo,hi);margin=min(com[0]-lo,hi-com[0]);balance=max(0.,-margin)
  else:
   cop=np.mean(xs);margin=-abs(com[0]-cop);balance=abs(com[0]-cop)
  left=(com[1]-ys[1])/(ys[0]-ys[1]);frc=np.zeros(m.nv)
  for body,y,fraction in zip(bodies,ys,[left,1-left]):
   jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,np.array([cop,y,0.]),body);frc+=jac[2]*mass*9.81*fraction
  efforts=d.qfrc_bias-frc
  tau={n:float(efforts[m.jnt_dofadr[i]]) for n,i in self.ids.items()}
  contacts=[]
  for c in d.contact:
   a,b=m.geom(c.geom1).name,m.geom(c.geom2).name
   if c.dist>=-.0005:continue
   contacts.append({'a':a,'b':b,'penetration_m':float(-c.dist)})
  # All detected non-support penetration stays visible, including unrelated
  # existing collisions; the optimizer specifically avoids hand/leg obstruction.
  active=[c for c in contacts if any(n.startswith(('left_gripper','left_forearm','left_upper_arm')) for n in [c['a'],c['b']])]
  penetration=max([c['penetration_m'] for c in active],default=0.)
  leg=[abs(v) for n,v in tau.items() if any(s in n for s in ['thigh','shin','ankle','hip_','foot_joint'])]
  return {'joints':{n:float(d.qpos[m.jnt_qposadr[i]]) for n,i in self.ids.items()},'grasp_point_m':point.tolist(),
   'base_position_m':d.qpos[:3].tolist(),'com_world_m':com.tolist(),'mass_kg':float(mass),'support_margin_x_m':float(margin),
   'balance_error_m':float(balance),'vertical_load_left_fraction':float(left),'required_static_effort_nm':tau,
   'peak_leg_effort_nm':float(max(leg)),'active_arm_penetration_m':penetration,'penetrations':contacts,
   'ankle_pitch_rad':float(-hip-knee) if self.feet else None,
   'base_equilibrium_residual':efforts[:6].tolist()}
 def loss(self,x):
  r=self.pose(x);p=np.array(r['grasp_point_m'])
  # The floor block may be anywhere in this reachable side/front patch. This
  # is an existence probe, not a navigation target silently moved by a policy.
  reach=abs(p[2]-.018)+max(0.,.015-p[0])+max(0.,p[0]-.12)+max(0.,.135-p[1])+max(0.,p[1]-.22)
  ankle=max(0.,abs(r['ankle_pitch_rad'])-.65) if self.feet else 0.
  return 100*reach+200*r['balance_error_m']+500*r['active_arm_penetration_m']+20*ankle+5*max(0.,r['peak_leg_effort_nm']-.7)


def assess_posture(r, feet=False):
 p=np.array(r['grasp_point_m'])
 checks={
  'reach_candidate':bool(abs(p[2]-.018)<.003 and .015<=p[0]<=.12 and .135<=p[1]<=.22),
  'static_load_within_current_cap':r['peak_leg_effort_nm']<=.7,
  'active_arm_clearance':r['active_arm_penetration_m']<.0005,
  'all_collision_clearance':not r['penetrations'],
  'support_candidate':r['balance_error_m']<.001 and 0<=r['vertical_load_left_fraction']<=1 and (not feet or abs(r['ankle_pitch_rad'])<=.65),
 }
 checks['status']='pass' if all(checks.values()) else 'fail'
 return checks


def run():
 out=ROOT/'artifacts'/('floor-cleanup-preflight-'+uuid.uuid4().hex[:12]);out.mkdir();print(out,flush=True)
 report={'status':'running','scope':'Kinematic pose search and quasistatic vertical-support load estimate. No robot controller, dynamics rollout, grasp or cleanup success is claimed.','models':{},'task_stages':{k:'not_run' for k in ['find_floor_target','move_to_target','bend_and_pick_up','carry_to_destination','set_down']}}
 try:
  report['feet_shared_cad_and_mass_links']=make_models(out)
  for rel in ['tools/probe_floor_cleanup.py','variants/pip-floor-feet-v0.yaml','lab/robot.py','lab/export.py']:
   snapshot_file(ROOT/rel,out/'source'/rel)
  snapshot_file(BUNDLE/'manifest.json',out/'source/bundle-manifest.json')
  for name in ['wheels','feet']:
   probe=Posture(out/(name+'.xml'),feet=name=='feet')
   opt=differential_evolution(probe.loss,[(0,1.8),(-1.2,.2),(0,.45),(-.5,.1),(-.5,.1)],seed=42,popsize=6,maxiter=45,polish=True,workers=1)
   r=probe.pose(opt.x);r.update({'optimizer_objective':float(opt.fun),'optimizer_converged':bool(opt.success),'search_evaluations':int(opt.nfev),'sampled_pose_not_global_impossibility_proof':True})
   p=np.array(r['grasp_point_m']);r.update(assess_posture(r,probe.feet))
   r['static_load_scope']='Unloaded robot with assumed vertical wheel/sole support; no payload, dynamics or validated contact forces. Invalid as a held posture when other geometry penetrates.'
   r['joint_count']=len(probe.ids);r['visual_fidelity']='retained CAD' if name=='wheels' else 'retained shared-link CAD plus new foot/hip/ankle envelopes; no mechanism fit qualification'
   # Overlay a real 30-mm floor block in the diagnostic rendering. Robot pose
   # was assigned by the posture solver, not reached dynamically.
   model=probe.model;data=probe.data;renderer=mujoco.Renderer(model,height=480,width=800)
   cam=mujoco.MjvCamera();cam.lookat[:]=[.02,.07,.16];cam.distance=.75;cam.azimuth=40;cam.elevation=-10
   renderer.update_scene(data,camera=cam);scene=renderer.scene
   g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_BOX,np.array([.015,.015,.015]),np.array([p[0],p[1],.015]),np.eye(3).ravel(),np.array([.95,.4,.05,1.]));scene.ngeom+=1
   from PIL import Image,ImageDraw
   im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.text((10,10),name.upper()+' — KINEMATIC FLOOR-REACH PROBE; NOT A CLEANUP ROLLOUT',fill='black');im.save(out/(name+'.png'));renderer.close()
   report['models'][name]=r
   print(name,r['status'],r['grasp_point_m'],r['peak_leg_effort_nm'],r['active_arm_penetration_m'],flush=True)
  report['status']='preflight_complete_cleanup_not_run'
  report['control_compatibility']={'wheels':'Existing frozen motor commands only ±0.025 rad leg residuals around upright; no crouch/grasp/transport policy.','feet':'22 actuators versus 16; existing 47-observation/4-residual wheeled actor is incompatible; a foot-contact controller/policy and robot-specific training are required.'}
 except Exception as e:
  import traceback
  report['status']='fail';report['error']=str(e);(out/'error.log').write_text(traceback.format_exc());raise
 finally:
  (out/'report.json').write_text(json.dumps(report,indent=2));migration=plan_migration(ROOT,[out]);apply_migration(ROOT,migration);verify_manifest(ROOT,migration)
 return out

if __name__=='__main__':run()
