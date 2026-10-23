"""Bounded PPO skill adaptation, frozen motor control, image observations and replay."""
import argparse,json,os,sys,time,uuid,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from lab.household_learning import InteractionEnv,build_scene,SKILLS
from lab.mujoco_learning import BUNDLE,ISAAC_CHECKPOINT,digest
from lab.control_assets import verify_control_bundle
from lab.artifact_store import snapshot_file,plan_migration,apply_migration,verify_manifest
PARENT=ROOT/'artifacts/pip-realistic-gym-v7/evidence/control/camera/skills/selector.zip'
CASES={'far':{'forward_m':.72,'left_m':.04},'near':{'forward_m':.38,'left_m':-.03},
       'settle':{'forward_m':.51,'left_m':.01},'vision_loss':{'forward_m':.67,'left_m':-.05,'blackout':True},
       'target_moves_closer':{'forward_m':.65,'left_m':.04,'intrusion':True}}
SCOPE='Gentle interaction proxy: camera-driven pause/approach/retreat over frozen Isaac motor policy. Known-size colour-marker vision and steering/safety are authored. Full CAD, free base, delayed actuator/battery proxy. No cat recognition, comfort qualification, object manipulation or learned-world-model control.'

class Progress(BaseCallback):
 def __init__(self,out):super().__init__();self.out=out;self.episodes=[]
 def _on_step(self):
  for info in self.locals['infos']:
   if 'episode' in info:self.episodes.append(info['episode'])
  if self.num_timesteps%256==0:
   (self.out/'progress.json').write_text(json.dumps({'steps':self.num_timesteps,'episodes':self.episodes[-20:]}))
   print('PPO steps',self.num_timesteps,flush=True)
  return True


def evaluate(model,scene,out,*,render=True,cases=None):
 import imageio.v2 as imageio
 from PIL import Image,ImageDraw
 env=InteractionEnv(scene,capture=True);reports={};all_rows=[]
 try:
  for i,(case,options) in enumerate((cases or CASES).items()):
   dest=out/case;dest.mkdir();obs,_=env.reset(seed=4100+i,options=options);total=0.;frames=[]
   for step in range(60):
    with torch.no_grad():
     tensor=torch.as_tensor(obs[None],dtype=torch.float32)
     probabilities=model.policy.get_distribution(tensor).distribution.probs[0].numpy().tolist()
     value=float(model.policy.predict_values(tensor).item())
    action,_=model.predict(obs,deterministic=True)
    obs,reward,t,tr,info=env.step(action);total+=reward
    row=env.rows[-1];row.update({'action_probabilities':probabilities,'predicted_discounted_return':value,'case':case,
      'body_names':[env.motor.model.body(b).name for b in range(1,env.motor.model.nbody)]})
    if render:
     camera=Image.fromarray(env.rgb).resize((320,180));draw=ImageDraw.Draw(camera)
     if row['estimate']:
      x,y,w,h=row['estimate']['bbox_xywh'];draw.rectangle((x,y,x+w,y+h),outline='lime',width=2)
     camera.save(dest/f'camera-{step:03}.png');row['image']=f'{case}/camera-{step:03}.png'
     canvas=Image.new('RGB',(960,440),'#102027');canvas.paste(Image.fromarray(env.render()),(0,0));canvas.paste(camera,(640,0));draw=ImageDraw.Draw(canvas)
     draw.text((650,195),'HEAD CAMERA -> VISION -> PPO SKILL',fill='white')
     draw.text((650,218),'Policy: '+row['skill']+' | override: '+row['override'],fill='white')
     est=row['estimate']['forward_m'] if row['estimate'] else None
     draw.text((650,245),'Range estimate: '+(f'{est:.3f} m' if est else 'MISSING'),fill='white')
     draw.text((650,266),f"Simulator range: {row['truth_forward_m']:.3f} m",fill='white')
     draw.text((650,290),f"Motor command: {row['command'][0]:+.3f} m/s",fill='white')
     draw.text((15,375),f"Gentle interaction PROXY | {case} | {row['time_s']:.1f} s | learned skill: {row['skill']}",fill='white')
     draw.text((15,398),'Marker vision / safety are authored. Electrical values estimated. No real-cat qualification.',fill='white')
     frames.append(np.asarray(canvas))
    if t or tr:break
   (dest/'trace.json').write_text(json.dumps(env.rows))
   if frames:imageio.mimwrite(dest/'replay.mp4',frames,fps=5,codec='libx264',quality=7,macro_block_size=1)
   errors=[r['vision_error_m'] for r in env.rows if r['vision_error_m'] is not None]
   reports[case]={**info,'return':total,'elapsed_sim_s':float(env.motor.data.time),'seed':4100+i,'reset':options,
     'final_range_m':env.rows[-1]['truth_forward_m'],'minimum_range_m':min(r['truth_forward_m'] for r in env.rows),
     'range_estimate_rmse_m':float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
     'vision_missing_frames':sum(r['estimate'] is None for r in env.rows),
     'skill_counts':{n:sum(r['skill']==n for r in env.rows) for n in SKILLS},
     'lost_vision_stop':all(r['command']==[0.,0.,0.] for r in env.rows if r['override']=='missing_vision'),
     'energy_j':env.motor.pack.energy_j}
   all_rows.extend(env.rows)
   print(case,reports[case],flush=True)
 finally:env.close()
 return reports,all_rows


def run(steps,parent=None,render=True,mode='train',case='far',distance=None,lateral=None):
 torch.set_num_threads(2)
 if steps<256 or steps>8192 or steps%256:raise ValueError('Use 256–8192 PPO steps, multiples of 256')
 out=ROOT/'artifacts'/('household-interaction-'+uuid.uuid4().hex[:12]);out.mkdir();print(str(out),flush=True)
 report={'schema_version':1,'status':'running','scope':SCOPE,'task':'gentle-interaction-proxy','execution_status':'running',
   'training_steps':0,'motor_checkpoint_sha256':digest(ISAAC_CHECKPOINT),'robot_bundle':str(BUNDLE.relative_to(ROOT)),
   'learning_layers':{'vision':'authored known-size RGB marker detector','skill':'SB3 PPO, selected from pause/approach/retreat',
    'motor':'frozen existing Isaac RSL-RL checkpoint; authored balance underneath','physics':'MuJoCo, 500 Hz',
    'electrical':'unmeasured delay/torque-speed/battery RC proxy','world_prediction':'not in policy path; Cosmos transfer failed qualification'},
   'acceptance':'12 s, no fall/overspeed, minimum optical forward separation >= 0.30 m, final >=1 s within 0.50±0.06 m and |lateral|<0.08 m at |forward speed|<0.025 m/s. Provisional proxy criterion, not animal-safety claim.',
   'policy_observation':'[marker visible, camera forward metres, camera left metres, 0]; no clock/seed/truth/path',
   'rates_hz':{'physics':500,'motor':50,'camera_and_skill':5},'seed':73}
 begin=time.monotonic();env=None
 try:
  report['source_verification']=verify_control_bundle(BUNDLE)
  for rel in ['tools/household_experiment.py','lab/household_learning.py','lab/mujoco_learning.py','lab/electromechanical.py','learning/pip-control-v1.json','learning/pip-electromechanical-v1.json','tools/write_household_rrd.py']:
   p=ROOT/rel
   if p.exists():snapshot_file(p,out/'source'/rel)
  snapshot_file(BUNDLE/'manifest.json',out/'source/bundle-manifest.json')
  (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
  build_scene(out/'scene.xml');env=InteractionEnv(out/'scene.xml')
  if parent:
   model=PPO.load(parent,env=env,device='cpu');report['parent_checkpoint']=str(parent);report['parent_sha256']=digest(parent)
  else:
   model=PPO('MlpPolicy',env,n_steps=256,batch_size=64,n_epochs=8,learning_rate=.001,gamma=.95,gae_lambda=.95,
             ent_coef=.025,policy_kwargs={'net_arch':[32,32]},seed=73,device='cpu',verbose=0)
   old=PPO.load(PARENT,device='cpu')
   model.policy.mlp_extractor.load_state_dict(old.policy.mlp_extractor.state_dict())
   report['initialization']='Retained Isaac skill-selector actor/critic feature layers; new heads because third skill changed from align to retreat.'
   report['parent_checkpoint']=str(PARENT);report['parent_sha256']=digest(PARENT)
  if mode=='train':
   start=time.monotonic();callback=Progress(out);model.learn(total_timesteps=steps,callback=callback,reset_num_timesteps=not bool(parent))
   report['training_seconds']=time.monotonic()-start;report['training_steps']=steps
   (out/'training-episodes.json').write_text(json.dumps(callback.episodes))
  model.save(out/'policy.zip');probe=env.reset(seed=4001)[0];before=model.predict(probe,deterministic=True)[0]
  restored=PPO.load(out/'policy.zip',device='cpu');report['reload_action_match']=bool(before==restored.predict(probe,deterministic=True)[0]);report['checkpoint_sha256']=digest(out/'policy.zip')
  env.close();env=None
  cases=None
  if mode=='evaluate':
   opts=dict(CASES[case])
   if distance is not None:opts['forward_m']=distance
   if lateral is not None:opts['left_m']=lateral
   cases={case:opts}
  report['cases'],rows=evaluate(restored,out/'scene.xml',out,render=render,cases=cases)
  report['execution_status']='pass';report['status']='pass' if all(c['success'] for c in report['cases'].values()) else 'fail'
  if render:
   from lab.rerun_scene import export_scene
   export_scene(out/'scene.xml',out/'scene.json')
   (out/'replay.json').write_text(json.dumps({'rows':rows,'interface':json.loads((BUNDLE/'model/interfaces.json').read_text()),'scope':SCOPE}))
   with (out/'rerun-export.log').open('w') as log:
    subprocess.run([str(ROOT/'.venv-observe/bin/python'),'tools/write_household_rrd.py',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    subprocess.run([str(ROOT/'.venv-observe/bin/rerun'),'rrd','verify',str(out/'episode.rrd')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
 except Exception as exc:
  import traceback
  report['execution_status']='fail';report['status']='fail';report['error']=str(exc);(out/'error.log').write_text(traceback.format_exc());raise
 finally:
  if env:env.close()
  report['elapsed_s']=time.monotonic()-begin;(out/'report.json').write_text(json.dumps(report,indent=2))
  manifest=plan_migration(ROOT,[out]);apply_migration(ROOT,manifest);verify_manifest(ROOT,manifest)
 print(str(out),flush=True);return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=2048);p.add_argument('--parent',type=Path);p.add_argument('--no-render',action='store_true');p.add_argument('--mode',choices=['train','evaluate'],default='train');p.add_argument('--case',choices=CASES,default='far');p.add_argument('--distance',type=float);p.add_argument('--lateral',type=float)
 a=p.parse_args()
 if a.mode=='evaluate' and not a.parent:p.error('Evaluation needs a retained checkpoint')
 if a.distance is not None and not .34<=a.distance<=.85:p.error('distance outside proxy envelope')
 if a.lateral is not None and abs(a.lateral)>.12:p.error('lateral outside proxy envelope')
 run(a.steps,a.parent,not a.no_render,a.mode,a.case,a.distance,a.lateral)
